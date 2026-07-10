import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from DenSet_40_model import DenseNet


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


def create_test_dataloader(data_dir, batch_size, num_workers):
    """构建官方 CIFAR-10 测试集的数据加载器，不对图片做随机增强。"""
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_data = CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)
    return DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_model(model, dataloader, criterion, device):
    """计算模型在官方测试集上的平均损失和准确率。"""
    model.eval()
    loss_sum = 0.0
    correct_count = 0
    sample_count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct_count += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    return loss_sum / sample_count, correct_count / sample_count


def parse_args():
    """读取测试所需的命令行参数。"""
    parser = argparse.ArgumentParser(description="测试 DenseNet-40 (k=12) 复现模型")
    parser.add_argument("--data-dir", default="./data", help="CIFAR-10 数据集目录")
    parser.add_argument("--model-path", default=None, help="待测试的模型参数文件路径")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model_path or os.path.join(base_dir, "best_densenet40_cifar10.pth")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"未找到模型参数文件：{model_path}。请先运行 train.py 完成训练。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet(num_layers=12, growth_rate=12, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_dataloader = create_test_dataloader(args.data_dir, args.batch_size, args.num_workers)

    test_start_time = time.perf_counter()
    test_loss, test_acc = evaluate_model(model, test_dataloader, nn.CrossEntropyLoss(), device)
    test_elapsed_time = time.perf_counter() - test_start_time

    print("官方测试集评估完成")
    print(f"测试损失：{test_loss:.4f} | 测试准确率：{test_acc:.4f}")
    print(f"测试用时：{format_elapsed_time(test_elapsed_time)}")
