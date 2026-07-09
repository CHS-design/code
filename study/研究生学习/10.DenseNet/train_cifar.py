from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import (
    count_parameters,
    densenet100_k12,
    densenet100_k24,
    densenet40_k12,
    densenet_bc100_k12,
    densenet_bc190_k40,
    densenet_bc250_k24,
)


MODEL_BUILDERS = {
    "densenet40_k12": densenet40_k12,
    "densenet100_k12": densenet100_k12,
    "densenet100_k24": densenet100_k24,
    "densenet_bc100_k12": densenet_bc100_k12,
    "densenet_bc250_k24": densenet_bc250_k24,
    "densenet_bc190_k40": densenet_bc190_k40,
}


def build_transforms(dataset: str, augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    train_ops: list[object] = []
    if augment:
        train_ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])

    test_ops = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int]:
    train_transform, test_transform = build_transforms(args.dataset, args.augment)
    root = Path(args.data_dir)

    dataset_class = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
    train_set = dataset_class(root=root, train=True, download=True, transform=train_transform)
    test_set = dataset_class(root=root, train=False, download=True, transform=test_transform)
    num_classes = 10 if args.dataset == "cifar10" else 100

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    return train_loader, test_loader, num_classes


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: SGD,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits, targets) * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_acc / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits, targets) * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_acc / total_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DenseNet on CIFAR-10/CIFAR-100.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), default="densenet_bc100_k12")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./runs")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--drop-rate", type=float, default=None)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(augment=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.pin_memory = device.type == "cuda"
    drop_rate = args.drop_rate if args.drop_rate is not None else (0.0 if args.augment else 0.2)

    train_loader, test_loader, num_classes = build_loaders(args)
    model = MODEL_BUILDERS[args.model](num_classes=num_classes, drop_rate=drop_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[args.epochs // 2, args.epochs * 3 // 4],
        gamma=0.1,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    print(f"model={args.model} params={count_parameters(model):,} device={device} drop_rate={drop_rate}")
    for epoch in range(1, args.epochs + 1):
        start = time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                "model": args.model,
                "dataset": args.dataset,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(checkpoint, output_dir / f"{args.model}_{args.dataset}_best.pt")

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"best_acc={best_acc:.4f} time={time() - start:.1f}s"
        )


if __name__ == "__main__":
    main()
