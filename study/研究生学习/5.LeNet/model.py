import torch
from torch import nn
from torchsummary import summary #torchsummary是一个第三方库，可以用来显示模型的结构和参数数量等信息

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()

        self.c1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2)
        self.sig = nn.Sigmoid()
        self.p1 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.c2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.p2 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=400,out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=84)
        self.fc3 = nn.Linear(in_features=84,out_features=10)
        


    def forward(self,x):
        x = self.sig(self.c1(x))
        x = self.p1(x)
        x = self.sig(self.c2(x))
        x = self.p2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
    
if __name__ == "__main__":
    device= torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device=device)
    print(summary(model=model,input_size=(1,28,28)))