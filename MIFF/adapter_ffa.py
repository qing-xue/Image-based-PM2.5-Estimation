"""
Python 的模块就是天然的单例模式，因为模块在第一次导入时，会生成 .pyc 文件，
当第二次导入时，就会直接加载 .pyc 文件，而不会再次执行模块代码。
"""
import torch
import torchvision.transforms as tfs
from PIL import Image
from common.FFANet.FFA import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class FFANet_Adapter(object):

    def __init__(self):
        model_dir = f'../common/FFANet/ots_train_ffa_3_19.pk'
        ckp = torch.load(model_dir, map_location=DEVICE)
        net = FFA(gps=3, blocks=19)
        net = nn.DataParallel(net)
        net.load_state_dict(ckp['model'])
        net.eval()  # 开启验证模式
        self.net = net
        self.transform = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
        ])

    def test(self, haze):
        haze1 = self.transform(haze)[None, ::]
        # haze_no = tfs.ToTensor()(haze)[None, ::]

        with torch.no_grad():
            pred = self.net(haze1)

        ts = torch.squeeze(pred.clamp(0, 1).cpu())
        ndarr = ts.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)

        return im


ffa_net = FFANet_Adapter()

# 将上面的代码保存在文件 mysingleton.py 中，要使用时，直接在其他文件中导入此文件中的对象，这个对象即是单例模式的对象
# from a import singleton