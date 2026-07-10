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
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from DenSet_40_model import DenseNet


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def train_test_data_process(data_dir, batch_size, num_workers):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_data = CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_data = CIFAR10(root=data_dir, train=False, transform=test_transform, download=True)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_dataloader = DataLoader(train_data, shuffle=True, **loader_options)
    test_dataloader = DataLoader(test_data, shuffle=False, **loader_options)
    return train_dataloader, test_dataloader


def evaluate_model(model, dataloader, criterion, device):
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


def train_model_process(model, train_dataloader, test_dataloader, num_epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    scheduler = MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = float("-inf")
    history = []
    since = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct_count = 0
        train_sample_count = 0

        for images, labels in train_dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct_count += (logits.argmax(dim=1) == labels).sum().item()
            train_sample_count += batch_size

        train_loss = train_loss_sum / train_sample_count
        train_acc = train_correct_count / train_sample_count
        test_loss, test_acc = evaluate_model(model, test_dataloader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })
        print(
            f"Epoch {epoch + 1:03d}/{num_epochs} | lr: {current_lr:.4g} | "
            f"train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"test loss: {test_loss:.4f}, acc: {test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    elapsed = time.time() - since
    print(f"训练完成，用时 {elapsed // 60:.0f}m{elapsed % 60:.0f}s，最佳测试准确率: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, pd.DataFrame(history)


def matplot_acc_loss(train_process, figure_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss"], "ro-", label="Train loss")
    plt.plot(train_process["epoch"], train_process["test_loss"], "bs-", label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc"], "ro-", label="Train accuracy")
    plt.plot(train_process["epoch"], train_process["test_acc"], "bs-", label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="训练 DenseNet-40 (k=12) 复现模型")
    parser.add_argument("--data-dir", default="./data", help="CIFAR-10 数据集目录")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "best_densenet40_cifar10.pth")
    history_path = os.path.join(base_dir, "densenet40_cifar10_history.csv")
    figure_path = os.path.join(base_dir, "densenet40_cifar10_curve.png")

    model = DenseNet(num_layers=12, growth_rate=12, num_classes=10)
    train_dataloader, test_dataloader = train_test_data_process(
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )
    model, train_process = train_model_process(
        model,
        train_dataloader,
        test_dataloader,
        args.epochs,
        args.learning_rate,
    )
    torch.save(model.state_dict(), model_path)
    train_process.to_csv(history_path, index=False)
    matplot_acc_loss(train_process, figure_path)