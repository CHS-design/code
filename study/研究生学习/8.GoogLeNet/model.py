import torch
from torch import nn
from torchsummary import summary

class Inception(nn.Module):
    '''Inception模块 
    c1:1x1卷积输出通道数 
    c2:1x1卷积->3x3卷积输出通道数
    c3:1x1卷积->5x5卷积输出通道数
    c4:3x3最大池化->1x1卷积输出通道数
    '''
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.Relu = nn.ReLU()

        #路线1:1x1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1)
        )
        #路线2:1x1卷积->3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        #路线3:1x1卷积->5x5卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        #路线4:3x3最大池化->1x1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1)
        )

    def forward(self, x):
        #路线1
        out1 = self.Relu(self.branch1(x))
        #路线2
        out2 = self.branch2(x)
        #路线3
        out3 = self.branch3(x)
        #路线4
        out4 = self.Relu(self.branch4(x))
        #将四条路线的输出结果在通道维度上进行拼接
        outputs = torch.cat((out1, out2, out3, out4), dim=1)
        return outputs   


class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
class GoogLeNet(nn.Module):
    def __init__(self, Inception, num_classes=2, aux_logits=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inception4a = Inception(480, 192, (96, 208), (16, 48), 64)
        self.aux1 = AuxClassifier(512, num_classes)
        self.inception4b = Inception(512, 160, (112, 224), (24, 64), 64)
        self.inception4c = Inception(512, 128, (128, 256), (24, 64), 64)
        self.inception4d = Inception(512, 112, (144, 288), (32, 64), 64)
        self.aux2 = AuxClassifier(528, num_classes)
        self.inception4e = Inception(528, 256, (160, 320), (32, 128), 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block3(x)

        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.training and self.aux_logits else None

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if self.training and self.aux_logits else None

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.block5(x)
        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNet(Inception).to(device)
    model.eval()
    summary(model, (3, 224, 224), device=device.type)
