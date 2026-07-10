import argparse
import copy
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from DenSet_40_model import DenseNet


# 让 Matplotlib 在 Windows 上优先使用常见中文字体，避免图表中的中文显示成方块。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# CIFAR-10 三个颜色通道的统计量。训练集、验证集和测试集必须使用同一组值归一化。
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def format_elapsed_time(elapsed_seconds):
    """将秒数格式化为便于终端阅读的中文时长。"""
    if elapsed_seconds < 60:
        return f"{elapsed_seconds:.1f}秒"

    total_seconds = round(elapsed_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}小时{minutes}分{seconds}秒"
    return f"{minutes}分{seconds}秒"


def train_val_data_process(data_dir, batch_size, num_workers, validation_ratio, seed):
    """从 CIFAR-10 训练集划分训练集和验证集，并构建数据加载器。"""
    if not 0 < validation_ratio < 1:
        raise ValueError("验证集比例必须大于 0 且小于 1。")

    # 只对训练图像做随机裁剪和翻转，帮助模型适应轻微的位置、方向变化。
    # 验证集不能使用随机增强，否则同一模型多次评估的结果会不一致。
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # 验证阶段只做格式转换和标准化，保持每张图片的原始内容不变。
    validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # 两个数据集实例共用同一批 50 000 张训练图片，但分别使用训练和验证的预处理方式。
    # download=True 会在 data_dir 中没有数据集时自动下载；官方测试集不在训练脚本中使用。
    train_data = CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    validation_data = CIFAR10(
        root=data_dir,
        train=True,
        transform=validation_transform,
        download=True,
    )

    # 固定随机种子后，每次运行都会得到相同的 45 000 / 5 000 训练验证划分。
    validation_size = int(len(train_data) * validation_ratio)
    split_indices = torch.randperm(
        len(train_data),
        generator=torch.Generator().manual_seed(seed),
    ).tolist()
    validation_indices = split_indices[:validation_size]
    train_indices = split_indices[validation_size:]
    train_data = Subset(train_data, train_indices)
    validation_data = Subset(validation_data, validation_indices)

    # CUDA 训练时使用锁页内存，能加快 CPU 到 GPU 的数据传输；CPU 训练时不启用它。
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    # 训练集每轮打乱，减少批次顺序对优化过程的影响；验证集固定顺序即可。
    train_dataloader = DataLoader(train_data, shuffle=True, **loader_options)
    validation_dataloader = DataLoader(validation_data, shuffle=False, **loader_options)
    return train_dataloader, validation_dataloader


def evaluate_model(model, dataloader, criterion, device):
    """在不更新参数的情况下，计算一个数据集的平均损失和准确率。"""
    # eval() 会让 BatchNorm 使用已学习的统计量，并关闭 Dropout 的随机失活。
    model.eval()
    loss_sum = 0.0
    correct_count = 0
    sample_count = 0

    # 验证时不需要梯度，关闭梯度计算可减少显存占用并提高速度。
    with torch.no_grad():
        for images, labels in dataloader:
            # non_blocking 与 pin_memory 配合时可异步传输数据到 GPU。
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # DenseNet 的输出是未经过 Softmax 的 logits，正好可直接传给交叉熵损失。
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            # loss.item() 是当前批次的平均损失，要乘回批大小后才能正确累计。
            loss_sum += loss.item() * batch_size
            # 每行 logits 的最大值下标就是模型预测的类别编号。
            correct_count += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    # 按样本总数求平均，避免最后一个较小批次影响统计结果。
    return loss_sum / sample_count, correct_count / sample_count


def train_model_process(model, train_dataloader, validation_dataloader, num_epochs, learning_rate):
    """按 DenseNet-40 的 CIFAR-10 配置训练模型，并返回最佳权重和训练记录。"""
    # 优先使用 GPU；没有 CUDA 环境时自动退回 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # DenseNet-40 的复现配置：SGD、动量 0.9、Nesterov 和 L2 正则化。
    optimizer = SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )

    # 第 150、225 轮结束后将学习率各缩小为原来的 0.1 倍。
    scheduler = MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # 训练过程中保存验证准确率最高时的权重，训练结束后恢复该版本。
    best_model_wts = copy.deepcopy(model.state_dict())
    best_validation_acc = float("-inf")
    history = []
    training_start_time = time.perf_counter()

    for epoch in range(num_epochs):
        # 单轮计时从进入训练模式开始，包含本轮训练和验证评估。
        epoch_start_time = time.perf_counter()

        # train() 会启用 BatchNorm 的批次统计更新，适合参数训练。
        model.train()
        train_loss_sum = 0.0
        train_correct_count = 0
        train_sample_count = 0

        for images, labels in train_dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 每个 mini-batch 都要清空上一批残留的梯度。
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)

            # 反向传播计算梯度，再根据 SGD 更新模型参数。
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct_count += (logits.argmax(dim=1) == labels).sum().item()
            train_sample_count += batch_size

        train_loss = train_loss_sum / train_sample_count
        train_acc = train_correct_count / train_sample_count

        # 一轮训练完成后，用没有参与反向传播的验证集记录泛化表现并选择最佳模型。
        validation_loss, validation_acc = evaluate_model(
            model,
            validation_dataloader,
            criterion,
            device,
        )

        # 先记录本轮实际使用的学习率，再更新下一轮的学习率。
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_elapsed_time = time.perf_counter() - epoch_start_time
        total_elapsed_time = time.perf_counter() - training_start_time

        # DataFrame 便于后续保存成 CSV，也能直接用于绘制训练曲线。
        history.append({
            "轮次": epoch + 1,
            "学习率": current_lr,
            "训练损失": train_loss,
            "训练准确率": train_acc,
            "验证损失": validation_loss,
            "验证准确率": validation_acc,
            "本轮耗时（秒）": epoch_elapsed_time,
            "累计耗时（秒）": total_elapsed_time,
        })
        print(
            f"第 {epoch + 1:03d}/{num_epochs} 轮 ，学习率：{current_lr:.4g}\n"
            f"训练损失：{train_loss:.4f} ，训练准确率：{train_acc:.4f}\n"
            f"验证损失：{validation_loss:.4f} ，验证准确率：{validation_acc:.4f}\n"
            f"本轮用时：{format_elapsed_time(epoch_elapsed_time)} ，"
            f"累计用时：{format_elapsed_time(total_elapsed_time)}"
        )

        if validation_acc > best_validation_acc:
            # deepcopy 防止后续训练继续修改已保存的最佳权重。
            best_validation_acc = validation_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    total_elapsed_time = time.perf_counter() - training_start_time
    finish_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"训练结束时间：{finish_time} | 总训练用时：{format_elapsed_time(total_elapsed_time)} | "
        f"最佳验证准确率：{best_validation_acc:.4f}"
    )

    # 返回验证准确率最高的模型，而非最后一轮模型。
    model.load_state_dict(best_model_wts)
    return model, pd.DataFrame(history)


def matplot_acc_loss(train_process, figure_path):
    """绘制损失和准确率曲线，并保存为 PNG 图片。"""
    plt.figure(figsize=(12, 4))

    # 左图显示损失是否稳定下降；训练损失与验证损失差距过大时可能出现过拟合。
    plt.subplot(1, 2, 1)
    plt.plot(train_process["轮次"], train_process["训练损失"], "ro-", label="训练损失")
    plt.plot(train_process["轮次"], train_process["验证损失"], "bs-", label="验证损失")
    plt.xlabel("训练轮次")
    plt.ylabel("损失")
    plt.legend()

    # 右图显示准确率变化，便于定位最佳验证准确率所在的训练阶段。
    plt.subplot(1, 2, 2)
    plt.plot(train_process["轮次"], train_process["训练准确率"], "ro-", label="训练准确率")
    plt.plot(train_process["轮次"], train_process["验证准确率"], "bs-", label="验证准确率")
    plt.xlabel("训练轮次")
    plt.ylabel("准确率")
    plt.legend()
    plt.tight_layout()

    # 先保存图片，再显示窗口，避免关闭窗口后丢失训练曲线。
    plt.savefig(figure_path, dpi=150)
    plt.show()


def parse_args():
    """读取命令行参数，默认值对应 DenseNet-40 的完整复现训练。"""
    parser = argparse.ArgumentParser(description="训练 DenseNet-40 (k=12) 复现模型")
    parser.add_argument("--data-dir", default="./data", help="CIFAR-10 数据集目录")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-ratio", type=float, default=0.1, help="从训练集划出的验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="固定训练集和验证集划分的随机种子")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 所有训练产物都保存到脚本所在目录，避免受运行终端当前目录影响。
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "best_densenet40_cifar10.pth")
    history_path = os.path.join(base_dir, "densenet40_cifar10_history.csv")
    figure_path = os.path.join(base_dir, "densenet40_cifar10_curve.png")

    # DenseNet-40：3 个 Dense Block，每个 Block 12 层，每层增长 12 个通道。
    model = DenseNet(num_layers=12, growth_rate=12, num_classes=10)

    # 数据加载器会在首次运行时自动下载 CIFAR-10 训练集，并固定划分训练集和验证集。
    train_dataloader, validation_dataloader = train_val_data_process(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.validation_ratio,
        args.seed,
    )

    # 完整训练结束后，model 已恢复为验证准确率最高时的权重。
    model, train_process = train_model_process(
        model,
        train_dataloader,
        validation_dataloader,
        args.epochs,
        args.learning_rate,
    )

    # 分别保存模型参数、每轮指标和可视化曲线，便于后续复现实验结果。
    torch.save(model.state_dict(), model_path)
    train_process.to_csv(history_path, index=False, encoding="utf-8-sig")
    matplot_acc_loss(train_process, figure_path)
