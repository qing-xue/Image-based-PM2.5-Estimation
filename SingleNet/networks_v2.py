import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

import backbones


class PM_Single_Net(nn.Module):
    def __init__(self, Body=None, pretrained=True):
        """天空 + 地面块共同提取特征，直接输出"""
        super(PM_Single_Net, self).__init__()
        self._get_Body(Body, pretrained)

    def forward(self, x):
        y = self.Body(x)
        y = self.fc(y)
        return y

    def _get_Body(self, name='vgg16', pretrained=True):
        if 'vgg16' == name:
            self.Body = backbones.VGG16_15(pretrained)  # Body output: 4096 feature vector
            self.fc = nn.Linear(4096, 1)      # output normalized PM2.5 value)
        elif 'resnet18' == name:
            self.Body = backbones.Resnet18_17(pretrained)  # Body output: 512 feature vector
            self.fc = nn.Linear(512, 1)
        elif 'mobilev2' == name:
            self.Body = backbones.MobileNetv2(pretrained)
            self.fc = nn.Linear(1280, 1)


if __name__ == '__main__':
    net = PM_Single_Net(Body='mobilev2')
    summary(net, input_size=[(3, 256, 256), ])

    x1 = torch.rand(1, 3, 256, 256)
    flops, params = profile(net, inputs=(x1,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')