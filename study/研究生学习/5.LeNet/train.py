# 导入必要的库和模块
import copy
import time
from pathlib import Path

import pandas as pd
from torchvision.datasets import FashionMNIST  # 导入FashionMNIST数据集
import numpy as np  # 导入NumPy库，用于科学计算
from torchvision import transforms  # 导入transforms模块，用于数据预处理
import torch
import torch.utils.data as Data  # 导入PyTorch的数据工具,主要用dataloader
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化
from model import LeNet  # 导入自定义的LeNet模型


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
NUM_WORKERS = 4
PIN_MEMORY = DEVICE.type == "cuda"
DATA_ROOT = Path(__file__).resolve().parent / "data"
MODEL_SAVE_PATH = Path(__file__).resolve().parent / "best_model.pth"


def train_val_data_process():
    # 数据预处理
    train_data = FashionMNIST(root=DATA_ROOT,
                              train=True,
                              transform=transforms.Compose([transforms.Resize(28),
                                                            transforms.ToTensor()]),
                              download=True)
    
    train_data,val_data = Data.random_split(train_data,[round(len(train_data)*0.8),round(len(train_data)*0.2)])

    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=NUM_WORKERS,
                                   pin_memory=PIN_MEMORY,
                                   persistent_workers=NUM_WORKERS > 0)

    val_loader = Data.DataLoader(dataset=val_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=PIN_MEMORY,
                                 persistent_workers=NUM_WORKERS > 0)
    
    return train_loader,val_loader

def train_model_process(model,train_loader,val_loader,num_epochs):
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = DEVICE
    #优化器选择：Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数选择：交叉熵损失函数
    critertion = torch.nn.CrossEntropyLoss()
    # 讲模型放入到训练设备中
    model = model.to(device)
    #复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表,保存每个epoch的训练集损失
    train_loss_all = []
    # 验证集损失列表,保存每个epoch的验证集损失
    val_loss_all = []
    # 训练集准确度列表,保存每个epoch的训练集准确度
    train_acc_all = []
    # 验证集准确度列表,保存每个epoch的验证集准确度
    val_acc_all = []
    # 当前时间,方便计算每轮训练时间
    since = time.time()
    best_epoch = 0

    # 进行num_epochs轮训练,每轮训练包括一个训练阶段和一个验证阶段
    for epoch in range(num_epochs):
        print("当前训练轮次: {}, 共 {} 轮".format(epoch, num_epochs-1))
        print('-' * 10)

        # 训练集和验证集的损失值和正确数初始化
        train_loss = 0.0      # 训练集损失值初始化为0
        train_corrects = 0    # 训练集预测正确数量初始化为0

        val_loss = 0.0        # 验证集损失值初始化为0
        val_corrects = 0      # 验证集预测正确数量初始化为0
        
        train_num = 0        # 训练集样本数量初始化为0,用于计算平均损失和准确度
        val_num = 0          # 验证集样本数量初始化为0,用于计算平均损失和准确度

        # 训练阶段
        for step, (train_data_x, train_data_y) in enumerate(train_loader):
            # 将特征放入到训练设备中
            train_data_x = train_data_x.to(device)
            # 将标签放入到训练设备中
            train_data_y = train_data_y.to(device)
            # 设置模型为训练模式
            model.train()

            # 前向传播过程，输入为训练数据集特征，输出为对每个样本的预测值
            output = model(train_data_x)
            #argmax函数返回指定维度上最大值的索引,这里dim=1表示在每行中取最大值的索引,是softmax函数的输出结果中概率最大的类别索引
            pre_lab = torch.argmax(output,dim=1) 
            # 计算训练集损失函数值,critertion是交叉熵损失函数,输入为模型的输出和真实标签
            loss = critertion(output,train_data_y)

            #梯度清零,在每次反向传播之前需要将梯度清零,否则梯度会累积
            optimizer.zero_grad()
            # 反向传播过程,利用loss.backward()计算损失函数关于模型参数的梯度
            loss.backward()
            # 更新模型参数,optimizer是Adam优化器,根据计算得到的梯度更新模型参数
            optimizer.step()

            # 计算训练集损失值和正确数量
            train_loss += loss.item() * train_data_x.size(0)  # 累加每个batch的损失值,乘以batch大小是为了得到总损失值
            train_corrects += torch.sum(pre_lab == train_data_y.data)  # 累加预测正确的数量,pre_lab是模型预测的标签,train_data_y.data是真实标签
            train_num += train_data_x.size(0)  # 累加训练集样本数量,train_data_x.size(0)是当前batch的样本数量   
            
        # 计算该轮次训练集的平均损失和准确度
        train_loss = train_loss / train_num
        train_acc = train_corrects.double().item() / train_num
        # 将该轮次训练集的损失和准确度添加到列表中
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)


        # 验证阶段
        model.eval()
        with torch.no_grad():
            for step, (val_data_x, val_data_y) in enumerate(val_loader):
                # 将特征放入到验证设备中
                val_data_x = val_data_x.to(device)
                # 将标签放入到验证设备中
                val_data_y = val_data_y.to(device)

                # 前向传播过程，输入为验证数据集特征，输出为对每个样本的预测值
                output = model(val_data_x)
                pre_lab = torch.argmax(output,dim=1)  # 获取预测标签,与训练阶段相同,获取每行最大值的索引作为预测标签
                loss = critertion(output,val_data_y)  # 计算验证集损失函数值

                # 计算验证集损失值和正确数量
                val_loss += loss.item() * val_data_x.size(0)  # 累加每个batch的损失值,乘以batch大小是为了得到总损失值
                val_corrects += torch.sum(pre_lab == val_data_y.data)  # 累加预测正确的数量
                val_num += val_data_x.size(0)  # 累加验证集样本数量,val_data_x.size(0)是当前batch的样本数量

        # 计算该轮次验证集的平均损失和准确度
        val_loss = val_loss / val_num
        val_acc = val_corrects.double().item() / val_num
        # 将该轮次验证集的损失和准确度添加到列表中
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)

        #打印该轮次的训练和验证结果
        print("第{}轮训练集损失: {:.4f}, 训练集准确率: {:.4f}".format(epoch, train_loss, train_acc))
        print("第{}轮验证集损失: {:.4f}, 验证集准确率: {:.4f}".format(epoch, val_loss, val_acc))

        # 如果当前验证集准确度大于之前保存的最高准确度,则更新最高准确度,并保存当前模型参数
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        # 该轮次计算训练总时间
        time_elapsed = time.time() - since
        print("第{}轮训练时间: {:.0f}m {:.0f}s".format(epoch, time_elapsed // 60, time_elapsed % 60))

    # 训练结束后，保存验证集准确率最高的模型参数
    torch.save(best_model_wts, MODEL_SAVE_PATH)
    print("训练结束，验证集准确率最高在{}轮，最高准确率为{:.4f}".format(best_epoch, best_acc))
    # 保存训练过程数据
    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                        "train_loss_all": train_loss_all,
                                        "train_acc_all": train_acc_all,
                                        "val_loss_all": val_loss_all,
                                        "val_acc_all": val_acc_all
                                    })
    return train_process

def matplot_train_process(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)  #1行2列第1个子图
    plt.plot(train_process["epoch"], train_process["train_loss_all"],"ro-", label="Train Loss")
    plt.plot(train_process["epoch"], train_process["val_loss_all"], "bo-", label="Val Loss")
    plt.legend()

    plt.subplot(1, 2, 2)  #1行2列第2个子图
    plt.plot(train_process["epoch"], train_process["train_acc_all"], "ro-", label="Train Acc")
    plt.plot(train_process["epoch"], train_process["val_acc_all"], "bo-", label="Val Acc")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #模型实例化
    model = LeNet()
    #加载数据集
    train_data, val_data = train_val_data_process()
    #训练模型
    train_process = train_model_process(model, train_data, val_data,10)
    #绘制训练过程
    matplot_train_process(train_process)
