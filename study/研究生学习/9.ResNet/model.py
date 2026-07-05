import torch
from torch import nn
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self,in_channels,num_channels,use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.Relu = nn.ReLU()
        self.Conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, stride=stride)
        self.Conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            self.Conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=stride)
        else:
            self.Conv3 = None

    def forward(self, X):
        Y = self.Relu(self.bn1(self.Conv1(X)))
        Y = self.bn2(self.Conv2(Y))
        if self.Conv3:
            X = self.Conv3(X)
        return self.Relu(Y + X)


class ResNet(nn.Module):
    def __init__(self, Residual):
        super(ResNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            Residual(64, 64),
            Residual(64, 64),
        )
        self.block3 = nn.Sequential(
            Residual(64, 128, use_1x1conv=True, stride=2),
            Residual(128, 128),
        )
        self.block4 = nn.Sequential(
            Residual(128, 256, use_1x1conv=True, stride=2),
            Residual(256, 256),
        )
        self.block5 = nn.Sequential(
            Residual(256, 512, use_1x1conv=True, stride=2),
            Residual(512, 512),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 10)
        )
    def forward(self, X):
        X = self.block1(X)
        X = self.block2(X)
        X = self.block3(X)
        X = self.block4(X)
        X = self.block5(X)
        X = self.fc(X)
        return X
