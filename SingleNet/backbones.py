import torch
import torch.nn as nn
import torchvision.models as models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class VGG16_Body(nn.Module):
    """抽 avgpool 前的特征提取层"""
    def __init__(self, pretrained=True):
        super(VGG16_Body, self).__init__()

        vgg16 = models.vgg16_bn(pretrained)
        # vgg16.classifier = Identity()  # remove fcs

        self.vgg16 = vgg16.features
        self.avg_pool = nn.AdaptiveAvgPool2d(7)

    def forward(self, x):
        y = self.vgg16(x)
        y = self.avg_pool(y)
        return y


class VGG16_15(nn.Module):
    """仅去掉最后一层全连接层"""
    def __init__(self, pretrained=True):
        super(VGG16_15, self).__init__()

        vgg16 = models.vgg16_bn(pretrained)
        vgg16.classifier[-1] = Identity()  # remove the last fc layer

        self.vgg16_15 = vgg16

    def forward(self, x):
        y = self.vgg16_15(x)
        return y


class Resnet18_17(nn.Module):
    """仅去掉最后一层全连接层"""
    def __init__(self, pretrained=True, del_type='fc'):
        super(Resnet18_17, self).__init__()

        m = models.resnet18(pretrained)
        if del_type == 'fc':
            # m = nn.Sequential(*list(m.children())[:-1])
            m.fc = Identity()
        elif del_type == 'avgpool':
            m = nn.Sequential(
                *list(m.children())[:-2],
                nn.AdaptiveAvgPool2d((1, 1))
            )  # lsit 避免调用了原始的 flatten()

        self.model = m

    def forward(self, x):
        y = self.model(x)
        return y


class MobileNetv2(nn.Module):
    def __init__(self, pretrained=True, del_type='fc'):
        super(MobileNetv2, self).__init__()

        m = models.mobilenet_v2(pretrained)
        if del_type == 'fc':
            m.classifier[-1] = Identity()
        elif del_type == 'avgpool':
            m = nn.Sequential(
                *list(m.features),
                nn.AdaptiveAvgPool2d((1, 1))
            )  # lsit 避免调用了原始的 flatten()

        self.model = m  # output: 1280

    def forward(self, x):
        y = self.model(x)
        return y



if __name__ == '__main__':
    # VGG16_Body, VGG16_15, Resnet18_17
    net = MobileNetv2(del_type='avgpool')
    print(net)

    x = torch.rand(4, 3, 256, 256)
    net(x)