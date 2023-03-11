from PIL import Image
import os
import torch
import numpy as np
import copy
import pandas as pd

from net import IAWD
from IA_Features import IA_Features


def old2keys(state_dict):
    """Match the key-values"""
    state_dict_v2 = copy.deepcopy(state_dict)
    for key in state_dict:
        if 'module' in key:
            pre, post = key.split('.', maxsplit=1)
            state_dict_v2[post] = state_dict_v2.pop(key)

    return state_dict_v2


def process_input(im):
    """将 PIL 图片处理为 Tensor 类型以适用于模型"""
    (width, height) = (im.width // 2, im.height // 2)
    im = im.resize((width, height))  # PIL 先 Resize 减少计算量
    info = IA_Features().getInfo(im)
    x = torch.from_numpy(np.array(info, dtype=np.float32))
    x = torch.unsqueeze(x, 0)        # add dim [B, C, ...]
    return x


# TODO: Only for Beijing Dataset ------------------------------
Daytime_PM_MAX = 262.0
Daytime_PM_MIN = 1.0

# 模型：-------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(r'PM_IAWD_Beijing.pk', map_location=device)

net = IAWD()
state_dict = old2keys(ckp['model'])
net.load_state_dict(state_dict, strict=True)
net.eval()

# 数据：图片所在文件夹 ---------------------------------------------
test_imgs = r'D:\workplace\dataset\V-picture-2'
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