import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import pandas as pd
import numpy as np

from networks_v2 import PM_Single_Net


def process_input(im):
    """将 PIL 图片处理为 Tensor 类型以适用于模型"""
    im = tfs.Resize((256, 256))(im)  # magic number: 256
    t_im = tfs.ToTensor()(im)[None, ::]
    return t_im


# 数据：测试图片 ------------------------------------------
Daytime_PM_MAX = 262.0  # Only for Beijing Dataset
Daytime_PM_MIN = 1.0

# TODO：test_imgs
test_imgs = r'../imgs'
img_dir = test_imgs + '/'
output_dir = img_dir
print("pred_dir:", output_dir)

# 模型 ---------------------------------------------------
# TODO：exp_dir
exp_dir = r""
model_dir = exp_dir + r"./MobileNetV2.pk"  # ['ResNet18.pk', MobileNetV2.pk']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(model_dir, map_location=device)
print("Load model: ", model_dir)

# TODO：net 还要改对应的模型文件路劲
net = PM_Single_Net(Body='mobilev2')  # ['resnet18', 'mobilev2']
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()

# 开始测试 ------------------------------------------------
all_pred = np.empty((0, 1), float)
img_names = []

for im in os.listdir(img_dir):
    if not im.endswith(('jpg', 'jpeg', 'png')):
        continue
    print(f'\r {im}', end='\n', flush=True)
    img_names += [im]

    haze = Image.open(img_dir + im)
    haze1 = process_input(haze)  # [H,W,C], 而 haze.size: W,H. 顺序不一样

    with torch.no_grad():
        pred = net(haze1)

    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    pred_nor = pred * (Daytime_PM_MAX - Daytime_PM_MIN) + Daytime_PM_MIN
    print("Estimated PM2.5: ", pred_nor.flatten())
    all_pred = np.vstack((all_pred, pred_nor))

df = pd.DataFrame({'IMG': img_names, 'Preds': all_pred.flatten()})
df.to_excel(os.path.join(output_dir, 'results.xlsx'), index=False)
