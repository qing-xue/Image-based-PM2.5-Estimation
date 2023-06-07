from PIL import Image
import os
import torch
import copy
import numpy as np
import pandas as pd

from model import classifier
from feature import BLH_BJ_min, BLH_BJ_max


def get_RGBs(im):
    width, height = im.width , im.height
    sky = im.crop((0, 0, width, height // 2))
    ground = im.crop((0, height // 2, width, height))

    sky_r, sky_g, sky_b = Image.Image.split(sky)
    sky_r, sky_g, sky_b = map(np.asarray, (sky_r, sky_g, sky_b))
    sky_r, sky_g, sky_b = map(np.mean, (sky_r, sky_g, sky_b))  # 取均值而不是求和

    grd_r, grd_g, grd_b = Image.Image.split(ground)
    grd_r, grd_g, grd_b = map(np.asarray, (grd_r, grd_g, grd_b))
    grd_r, grd_g, grd_b = map(np.mean, (grd_r, grd_g, grd_b))

    B_R_sky = sky_b * 1.0 / sky_r
    B_R_grd = grd_b * 1.0 / grd_r
    RGB_sky = sky_r + sky_g + sky_b
    RGB_grd = grd_r + grd_g + grd_b

    return B_R_sky, B_R_grd, RGB_sky, RGB_grd


def old2keys(state_dict):
    """Match the key-values"""
    state_dict_v2 = copy.deepcopy(state_dict)
    for key in state_dict:
        if 'model' in key:
            pre, post = key.split('.', maxsplit=1)
            state_dict_v2[post] = state_dict_v2.pop(key)

    return state_dict_v2


def process_input(im):
    """将 PIL 图片处理为 Tensor 类型以适用于模型
      KE_850, GE_500, R_B_Sky, R_B_Ground, RGB_Sky, RGB_Ground, T2m, BLH, U10, V10, TP, SP
    """
    (width, height) = (im.width // 2, im.height // 2)
    im = im.resize((width, height))  # PIL 先 Resize 减少计算量
    B_R_sky, B_R_grd, RGB_sky, RGB_grd = get_RGBs(im)

    B_R_sky = (B_R_sky - BLH_BJ_min[2]) / (BLH_BJ_max[2] - BLH_BJ_min[2])
    B_R_grd = (B_R_grd - BLH_BJ_min[3]) / (BLH_BJ_max[3] - BLH_BJ_min[3])
    RGB_sky = (RGB_sky - BLH_BJ_min[4]) / (BLH_BJ_max[4] - BLH_BJ_min[4])
    RGB_grd = (RGB_grd - BLH_BJ_min[5]) / (BLH_BJ_max[5] - BLH_BJ_min[5])

    info = np.full((1, 12), 0.5, dtype=np.float32)
    info[0][2], info[0][3] = B_R_sky, B_R_grd
    info[0][4], info[0][5] = RGB_sky, RGB_grd
    print(info)

    x = torch.from_numpy(np.array(info, dtype=np.float32))
    # x = torch.unsqueeze(x, 0)        # add dim [B, C, ...]
    return x


# TODO: Only for Beijing Dataset ------------------------------
Daytime_PM_MAX = 262.0
Daytime_PM_MIN = 1.0

# 模型：-------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(r'./best_beijing.pth', map_location=device)

net = classifier()
state_dict = old2keys(ckp['state_dict'])
net.load_state_dict(state_dict, strict=True)
net.eval()

# 数据：图片所在文件夹 ---------------------------------------------
test_imgs = r'D:\workplace\dataset\Various-Pic\V-picture\imgs'
img_dir = test_imgs + '/'
output_dir = img_dir
print("pred_dir:", output_dir)

# 开始测试：----------------------------------------------------------
all_pred = np.empty((0, 1), float)
img_names = []

for img_name in os.listdir(img_dir):
    if not img_name.endswith(('jpg', 'png', 'jpeg')):
        continue
    print(f'\r {img_name}', end='\n', flush=True)
    img_names += [img_name]

    im = Image.open(img_dir + img_name).convert('RGB')
    x = process_input(im)

    with torch.no_grad():
        pred = net(x)
        pred = torch.squeeze(pred.clamp(0, 1).cpu())
        pred_nor = pred * (Daytime_PM_MAX - Daytime_PM_MIN) + Daytime_PM_MIN
        print("Estimated PM2.5: ", pred_nor.flatten())

    all_pred = np.vstack((all_pred, pred_nor))

df = pd.DataFrame({'IMG': img_names, 'Preds': all_pred.flatten()})
df.to_excel(os.path.join(output_dir, 'results.xlsx'), index=False)