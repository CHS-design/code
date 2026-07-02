import copy
from pathlib import Path
# 导入pandas库，并将其简写为pd
# pandas是Python中强大的数据分析工具库，提供了DataFrame等数据结构
import pandas as pd
from model import LeNet
import torch
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"
MODEL_SAVE_PATH = BASE_DIR / "best_model.pth"

#1.加载数据集函数
def test_data_process():
    test_data = FashionMNIST(root = DATA_ROOT,
                             train = False,
                             transform = transforms.Compose([transforms.Resize(28),
                                                             transforms.ToTensor()]),
                             download = True)
    test_dataloader = Data.DataLoader(dataset = test_data,
                                      batch_size = 1,
                                      shuffle = True,
                                      num_workers = 4)
    return test_dataloader,test_data
#2.测试模型函数
def test_model_process(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    test_corrects = 0.0
    test_num = 0.0
    
    with torch.no_grad():
        for test_x,test_y in test_dataloader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            test_output = model(test_x)
            pre_lab = torch.argmax(test_output,dim = 1)
            test_corrects += (pre_lab == test_y).sum().item()
            test_num += test_x.size(0)
    
    
    test_acc = test_corrects / test_num
    print('测试准确率: {:.4f}'.format(test_acc))


if __name__ == '__main__':
    test_dataloader,test_data = test_data_process()
    model = LeNet()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    # test_model_process(model, test_dataloader)


    #推理流程
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    #直接获得数据集标签
    # print(test_data.classes)

    with torch.no_grad():
        for test_x,test_y in test_dataloader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            model.eval()
            test_output = model(test_x)
            pre_lab = torch.argmax(test_output,dim = 1)
            print('真实种类：',test_data.classes[test_y.item()])
            print('预测种类：',test_data.classes[pre_lab.item()])
            # print('真实标签：',test_y.item())
            # print('预测标签：',pre_lab.item())
            print('------------------------')







