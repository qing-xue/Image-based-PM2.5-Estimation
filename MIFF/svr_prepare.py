"""训练 MIFF 模型准备：提取训练集图像的特征"""

import os
from PIL import Image
import numpy as np

from adapter_ffa import ffa_net
from adapter_monodepth import monodepth_net
from common import data_load
import utils


def f1f2f3(haze):
    """注意返回值顺序！！！"""
    dehaze = ffa_net.test(haze).convert('RGB')

    # f1: Normalizing Entropy ----------------------------------
    entropy1 = utils._entropy(haze)
    entropy2 = utils._entropy(dehaze)
    rate = entropy1 / entropy2

    # f2: Depth Similarity -------------------------------------
    disp1 = monodepth_net.test(haze)
    disp2 = monodepth_net.test(dehaze)
    depth_sim = utils.similarity(np.asarray(disp1), np.asarray(disp2))

    # f3: Structure Loss ---------------------------------------
    image_residual = np.asarray(haze) - np.asarray(dehaze)
    residual_sim = utils.similarity(np.asarray(haze), image_residual)

    return rate, depth_sim, residual_sim


def extract_features(df, save_name='out.csv'):
    """df 包含图片名和对应的 PM2.5 值"""
    df.insert(df.shape[1], 'rate', 0.5)       # normalizing entropy
    df.insert(df.shape[1], 'depth_sim', 0.5)  # depth similarity
    df.insert(df.shape[1], 'residual', 0.5)   # color similarity
    delete_row = []

    for i, row in df.iterrows():
        img_name = row['file_Id']
        if not os.path.exists(img_name):
            delete_row.append(i)  # 一定要删除，不然会被默认值影响！！！
            continue

        print('Processing {}/{}: '.format(i, len(df)), img_name)
        haze = Image.open(img_name).convert('RGB')
        (width, height) = (haze.width // 2, haze.height // 2)
        haze = haze.resize((width, height))  # PIL 先 Resize 减少计算量
        rate, depth_sim, residual_sim = f1f2f3(haze)

        df.loc[i, 'rate'] = rate
        df.loc[i, 'depth_sim'] = depth_sim
        df.loc[i, 'residual'] = residual_sim

    print('Done. -----------------------------------------------')
    df.drop(index=df.index[delete_row], inplace=True)
    df.to_csv(df.to_csv(save_name, index=False))


if __name__ == '__main__':
    # ['file_Id', 'PM2.5']
    (df_train_Xys, df_test_Xys), dataset_name = data_load.get_PM_Dataset(r'D:\workplace\dataset\BLH_Photo_Beijing\Daytime')
    extract_features(df_train_Xys, 'df_info_train_miff.csv')
    extract_features(df_test_Xys, 'df_info_valid_miff.csv')


# root_train = '/home/jljl/file/cas/df_info_train_FFA.csv'
# root_valid = '/home/jljl/file/cas/df_info_valid_FFA.csv'
# features = ['depth_sim', 'residual', 'rate']



