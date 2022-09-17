from PIL import Image
import torch
import numpy as np
import copy

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


# TODO: Only for Beijing Dataset ------------------------------
Daytime_PM_MAX = 262.0
Daytime_PM_MIN = 1.0

# -------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(r'PM_IAWD_Beijing.pk', map_location=device)

net = IAWD()
state_dict = old2keys(ckp['model'])
net.load_state_dict(state_dict, strict=True)
net.eval()

# -------------------------------------------------------------
img_name = r'../imgs/Beijing_20190530051212642_PM=22.jpg'
im = Image.open(img_name).convert('RGB')

(width, height) = (im.width // 2, im.height // 2)
im = im.resize((width, height))  # PIL 先 Resize 减少计算量

info = IA_Features().getInfo(im)

# -------------------------------------------------------------
with torch.no_grad():
    x = torch.from_numpy(np.array(info, dtype=np.float32))
    x = torch.unsqueeze(x, 0)  # add dim [B, C, ...]
    pred = net(x)
    pred_nor = pred * (Daytime_PM_MAX - Daytime_PM_MIN) + Daytime_PM_MIN
    print("Estimated PM2.5: ", pred_nor.flatten())