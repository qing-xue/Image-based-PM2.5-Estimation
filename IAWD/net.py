"""2021 IAWD

Gu K, Liu H, Xia Z, et al. PM₂. ₅ Monitoring: Use Information Abundance Measurement and Wide and Deep Learning[J].
IEEE Transactions on Neural Networks and Learning Systems, 2021, 32(10): 4278-4290.

no officail code, or if you have found the official source code, please tell us.
"""

import torch
from torch import nn


class IAWD(torch.nn.Module):

    def __init__(self):
        super(IAWD, self).__init__()
        self.deep = BPNNModel_DNN()
        self.wide = BPNNModel_WNN()
        self.layer = nn.Linear(1, 1, bias=True)
        self.bn = nn.BatchNorm1d(num_features=6, affine=False)

    def forward(self, x):
        x = self.bn(x)                     # simulate the normalization of input data
        x1 = self.deep(x)
        x2 = self.wide(x)
        y = 0.5 * x1 + 0.5 * x2            # Equation (17)
        y = self.layer(y)
        return y


class BPNNModel_DNN(torch.nn.Module):
    """Deep neural network

    Page 7: Three hidden layers (essentially three fully connected layers) with two hidden neurons in each layer
    in the deep neural network.
    """

    def __init__(self):
        super(BPNNModel_DNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(6, 2), nn.Sigmoid())
        self.layer2 = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())
        self.layer3 = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())
        self.layer4 = nn.Sequential(nn.Linear(2, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        o = self.layer4(x)
        return o


class BPNNModel_WNN(torch.nn.Module):
    """Wide neural network

    Page 7: One hidden layer (essentially one fully connected layer) with eleven hidden neurons in the wide neural network.
    """

    def __init__(self):
        super(BPNNModel_WNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(6, 11), nn.Sigmoid())
        self.layer4 = nn.Sequential(nn.Linear(11, 1))

    def forward(self, x):
        x = self.layer1(x)
        o = self.layer4(x)
        return o
