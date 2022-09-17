from PIL import Image
import torch
import numpy as np

from net import IAWD
from IA_Features import IA_Features


# -------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(r'PM_IAWD_Beijing.pk', map_location=device)

net = IAWD()
net.load_state_dict(ckp['model'], strict=False)
net.eval()

# -------------------------------------------------------------
img_name = r'../imgs/Beijing_20190530051212642.jpg'
im = Image.open(img_name).convert('RGB')

(width, height) = (im.width // 2, im.height // 2)
im = im.resize((width, height))  # PIL 先 Resize 减少计算量

info = IA_Features().getInfo(im)

# -------------------------------------------------------------
with torch.no_grad():
    x = torch.from_numpy(np.array(info, dtype=np.float32))
    # add dim [B, C, H, W]...
    pred = net(info)
    print("Estimated PM2.5: ", pred)