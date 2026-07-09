import torch
from torch import nn
from torchsummary import summary

class DenseLayer(nn.Module):
    def __init__(self,in_channels,growth_rate):
        super().__init__()

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels,growth_rate,kernel_size=3,padding=1,bias=False)

    def forward(self,X):
        Y = self.bn(X)
        Y = self.relu(Y)
        Y = self.conv(Y)
        return Y
    
class DenseBlock(nn.Module):
    def __init__(self, in_channels,num_layers,growth_rate):
        super().__init__()

        self.net = nn.ModuleList()
        for layer in range(num_layers):
            layer_in_channels = in_channels + layer*growth_rate
            self.net.append(DenseLayer(layer_in_channels,growth_rate=growth_rate))

    def forward(self,x):
        for layer in self.net:
            y = layer(x)
            x = torch.cat((x , y) , dim = 1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False,)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, in_channels = 3, num_layers= 12, growth_rate=12,  num_classes=10):
        super().__init__()

        channels = 16
        self.conv0 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)

        self.denseblock1 = DenseBlock(in_channels=channels, num_layers=num_layers, growth_rate=growth_rate)
        channels = channels + num_layers * growth_rate
        self.trans1 = TransitionLayer(in_channels=channels, out_channels=channels)

        self.denseblock2 = DenseBlock(in_channels=channels, num_layers=num_layers, growth_rate=growth_rate)
        channels = channels + num_layers * growth_rate
        self.trans2 = TransitionLayer(in_channels=channels, out_channels=channels)

        self.denseblock3 = DenseBlock(in_channels=channels, num_layers=num_layers, growth_rate=growth_rate)
        channels = channels + num_layers * growth_rate

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(channels, num_classes)
        )

    def forward(self, x):
        x = self.conv0(x)

        x = self.denseblock1(x)
        x = self.trans1(x)

        x = self.denseblock2(x)
        x = self.trans2(x)

        x = self.denseblock3(x)

        x = self.classifier(x)
        return x
    
if __name__ == '__main__':
    model = DenseNet().to(device='cuda')
    summary(model, (3, 32, 32), device='cuda')
    print(model)








    
