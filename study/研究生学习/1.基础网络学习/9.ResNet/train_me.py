import copy
import os
import time

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import ResNet, Residual
import torch.nn as nn
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pth')
NUM_CLASSES = 2
BATCH_SIZE = 128


def build_model(num_classes=NUM_CLASSES):
    model = ResNet(Residual)
    model.fc[-1] = nn.Linear(model.fc[-1].in_features, num_classes)
    return model


def load_compatible_weights(model, model_path):
    if not os.path.exists(model_path):
        print("没有找到 best_model.pth，从头训练")
        return

    state_dict = torch.load(model_path, map_location='cpu')
    model_state = model.state_dict()
    compatible_state = {}
    skipped_keys = []

    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped_keys.append(key)

    if not compatible_state:
        print("找到 best_model.pth，但结构不匹配，已从头训练")
        return

    model_state.update(compatible_state)
    model.load_state_dict(model_state)
    if skipped_keys:
        print("已加载 best_model.pth 中的兼容参数，跳过 {} 个不匹配参数".format(len(skipped_keys)))
    else:
        print("已加载 best_model.pth，继续训练")


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4879, 0.4545, 0.4167],
            std=[0.2596, 0.2524, 0.2552]
        )
    ])


def train_val_data_process():
    # 数据集路径
    ROOT_TRAIN_DATA = os.path.join(BASE_DIR, 'data_me', 'train')
    ROOT_VAL_DATA = os.path.join(BASE_DIR, 'data_me', 'test')

    # 数据集预处理方法
    transform = get_transform()
    
    # 加载训练集和验证集
    train_data = ImageFolder(ROOT_TRAIN_DATA, transform=transform)
    val_data = ImageFolder(ROOT_VAL_DATA, transform=transform)

    if train_data.class_to_idx != val_data.class_to_idx:
        raise ValueError("训练集和验证集类别映射不一致：{} vs {}".format(train_data.class_to_idx, val_data.class_to_idx))

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=0)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=BATCH_SIZE,
                                       shuffle=False,
                                       num_workers=0)

    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入到训练设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)

        # 初始化参数
        # 训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入到训练设备中
            b_x = b_x.to(device)
            # 将标签放入到训练设备中
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入到验证设备中
            b_x = b_x.to(device)
            # 将标签放入到验证设备中
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()
            with torch.no_grad():
                # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
                output = model(b_x)
                # 查找每一行中最大值对应的行标
                pre_lab = torch.argmax(output, dim=1)
                # 计算每一个batch的损失函数
                loss = criterion(output, b_y)


            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于验证的样本数量
            val_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        # 计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60, time_use%60))

    # 选择最优参数，保存最优参数的模型
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, BEST_MODEL_PATH)


    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":val_acc_all,})

    return train_process


def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载需要的模型
    model = build_model()
    load_compatible_weights(model, BEST_MODEL_PATH)

    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 利用现有的模型进行模型的训练
    train_process = train_model_process(model, train_data, val_data, num_epochs=50)
    matplot_acc_loss(train_process)
