import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
from skimage.color import rgb2ycbcr
from skimage.measure import shannon_entropy


import torch
from PIL import Image
import time
import pandas as pd
import os


'''
返回图片的信息熵
输入图片格式为torch.tensor
'''
def _entropy_tensor(image_src):
    image_src = image_src * torch.log2(image_src)
    sum = torch.sum(image_src)
    return sum * -1.0


'''
返回图片的信息熵
输入图片格式为np.array
'''
def _entropy(image_src):
    # img_gray = rgb2gray(image_src)
    img_gray = image_src.convert('L')
    entropy = shannon_entropy(img_gray)
    return entropy


'''
返回两张图片的相似度
输入图片格式为np.array
'''
def similarity(image_a, image_b):
    multichannel = True if len(image_a.shape) == 3 and image_a.shape[2] == 3 else False
    return ssim(image_a, image_b, multichannel=multichannel)


'''
返回两张图片的相似度
输入图片格式为torch.tensor
'''
def similarity_tensor(image_a, image_b):
    epsilon = 1e-9
    up = 2 * torch.sum(image_a * image_b) + epsilon
    down = torch.sum(image_a * image_a + image_b * image_b) + epsilon


'''
返回原图的深度图和去雾图深度图的相似度
'''
def _depth():
    import torch
    from PIL import Image
    import time
    import pandas as pd
    import os

    root_df = '/mnt/nvme1n1p1/jljl/cas/csv/Heshan_Daytime_train.csv'
    root_path1 = '/mnt/nvme1n1p1/jljl/cas/Heshan_imgset_depth'
    # root_path2 = '/public/home/ac2ws7ph6q/cas/Photo_Beijing_ld_depth'
    root_path3 = '/mnt/nvme1n1p1/jljl/cas/Heshan_imgset_FFA_depth'
    df = pd.read_csv(root_df)
    df = df['file_Id']
    list_file = []
    list_sim = []

    count = 0
    total = len(df)
    print('start==========')
    for file in df:
        image_path1 = os.path.join(root_path1, file.split('.')[0]+'_disp.jpeg')
        # image_path2 = os.path.join(root_path2, 'dehaze_'+file.split('.')[0]+'_disp.jpeg')
        image_path2 = os.path.join(root_path3, file.split('.')[0]+'_FFA_disp.jpeg')

        image_src1 = Image.open(image_path1).convert('RGB')
        image_src1 = np.asarray(image_src1)
        image_src1 = image_src1.astype('float64') / 255 # 除以255没区别
        # image_src1 = torch.tensor(data=image_src1, device='cuda')


        image_src2 = Image.open(image_path2).convert('RGB')
        image_src2 = np.asarray(image_src2)
        image_src2 = image_src2.astype('float64') / 255 # 除以255没区别
        # image_src2 = torch.tensor(data=image_src2, device='cuda')

        out1 = similarity(image_src1, image_src2)
        list_file.append(file)
        list_sim.append(out1)
        count = count + 1
        print(f'{count}/{total} success=====')

    df_out = pd.DataFrame({'file_Id':list_file, 'depth_sim': list_sim})
    df_out.to_csv('df_depth_train_FFA.csv', sep=',', index=False)


'''
返回原图和去雾图相减后的相似度
'''
def _residual():
    import torch
    from PIL import Image
    import time
    import pandas as pd
    import os

    print('start========================')
    root_df = '/mnt/nvme1n1p1/jljl/cas/csv/Heshan_Daytime_train.csv'
    root_path1 = '/mnt/nvme1n1p1/jljl/cas/Heshan_imgset'
    # root_path2 = '/public/home/ac2ws7ph6q/Light-DehazeNet-main/visual_results/train'
    root_path3 = '/mnt/nvme1n1p1/jljl/cas/Heshan_imgset_FFA'
    df = pd.read_csv(root_df)
    df = df['file_Id']
    list_file = []
    list_residual = []

    count = 0
    for file in df:
        print(file)
        image_path1 = os.path.join(root_path1, file)
        # image_path2 = os.path.join(root_path2, 'dehaze_' + file)
        image_path2 = os.path.join(root_path3, file.split('.')[0]+'_FFA.png')

        image_src1 = Image.open(image_path1).convert('RGB')
        image_src1 = np.asarray(image_src1)
        image_src1 = image_src1.astype('float64') / 255
        # image_tensor = torch.tensor(data=image_src, device='cuda')
        # out = _entropy_tensor(image_tensor)


        image_src2 = Image.open(image_path2).convert('RGB')
        image_src2 = np.asarray(image_src2)
        image_src2 = image_src2.astype('float64') / 255
        # image_tensor = torch.tensor(data=image_src, device='cuda')
        # out = _entropy_tensor(image_tensor)

        # residual
        image_residual = image_src1 - image_src2
        residual_sim = similarity(image_src2, image_residual)

        list_file.append(file)
        list_residual.append(residual_sim)
        count = count + 1
        print(count)


    df_out = pd.DataFrame({'file_Id': list_file, 'out1': list_residual})
    df_out.to_csv('df_residual_train_FFA.csv', sep=',', index=False)


if __name__=="__main__":
    entropy()