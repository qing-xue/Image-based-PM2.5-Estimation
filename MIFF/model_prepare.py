"""训练 MIFF 模型准备：提取训练集图像的特征"""

import os
from PIL import Image
from adapter_ffa import ffa_net
from common import data_load
import utils


def extract_features(df):
    """df 包含图片名和对应的 PM2.5 值"""
    df.insert(df.shape[1], 'rate', 0.5)  # normalizing entropy

    for i, row in df.iterrows():
        img_name = row['file_Id']
        haze = Image.open(img_name).convert('RGB')
        (width, height) = (haze.width // 2, haze.height // 2)
        haze = haze.resize((width, height))  # PIL 先 Resize 减少计算量
        dehaze = ffa_net.test(haze).convert('RGB')

        # f1: normalizing entropy ----------------------------------
        entropy1 = utils._entropy(haze)
        entropy2 = utils._entropy(dehaze)
        rate = entropy1 / entropy2
        df.loc[i, 'rate'] = rate


# ['file_Id', 'PM2.5']
(df_train_Xys, df_test_Xys), dataset_name = data_load.get_PM_Dataset(r'D:\workplace\dataset\BLH_Photo_Beijing\Daytime')
extract_features(df_train_Xys)


# root_train = '/home/jljl/file/cas/df_info_train_FFA.csv'
# root_valid = '/home/jljl/file/cas/df_info_valid_FFA.csv'
# features = ['depth_sim', 'residual', 'rate']



