import torch
import torch.nn as nn


class classifier(nn.Module):
    def __init__(self, in_channels=12, out_channels=1):
        super(classifier, self).__init__()
        self.stem = nn.Linear(in_channels, 3, bias=True)
        self.fc1 = nn.Linear(3, 3, bias=True)
        self.fc2 = nn.Linear(3, 3, bias=True)
        self.fc3 = nn.Linear(3, out_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
