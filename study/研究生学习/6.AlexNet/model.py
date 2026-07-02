import torch
from torch import nn
from torchsummary import summary # 查看模型结构


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__() # 调用父类的构造函数
        self.ReLu = nn.ReLU() 
        self.c1 = nn.Conv2d(1,96,11,4) # 输入通道1，输出通道96，卷积核大小11，步长4
        self.p1 = nn.MaxPool2d(3,2) # 最大池化，池化核大小3，步长2
        self.c2 = nn.Conv2d(96,256,5,1,2) # 输入通道96，输出通道256，卷积核大小5，步长1，padding=2
        self.p2 = nn.MaxPool2d(3,2) # 最大池化，池化核大小3，步长2
        self.c3 = nn.Conv2d(256,384,3,1,1) # 输入通道256，输出通道384，卷积核大小3，步长1，padding=1
        self.c4 = nn.Conv2d(384,384,3,1,1) # 输入通道384，输出通道384，卷积核大小3，步长1，padding=1
        self.c5 = nn.Conv2d(384,256,3,1,1) # 输入通道384，输出通道256，卷积核大小3，步长1，padding=1
        self.p3 = nn.MaxPool2d(3,2) # 最大池化，池化核大小3，步长2
        self.flatten = nn.Flatten() # 展平
        self.fc1 = nn.Linear(256*6*6,4096) # 全连接层，输入256*6*6，输出4096
        self.fc2 = nn.Linear(4096,4096) # 全连接层，输入4096，输出4096
        self.fc3 = nn.Linear(4096,10) # 全连接层，输入4096，输出10
        self.dropout = nn.Dropout(0.5)


    def forward(self,x):
        x = self.ReLu(self.c1(x))
        x = self.p1(x)
        x = self.ReLu(self.c2(x))
        x = self.p2(x)
        x = self.ReLu(self.c3(x))
        x = self.ReLu(self.c4(x))
        x = self.ReLu(self.c5(x))
        x = self.p3(x)

        x = self.flatten(x)
        x = self.ReLu(self.fc1(x))
        x = self.dropout(x)
        x = self.ReLu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    summary(model,(1,227,227)) # 查看模型结构
    print(model)
    
    
