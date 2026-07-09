# DenseNet 复现说明

这个目录里的复现代码优先复现论文在 CIFAR-10/CIFAR-100 上的 DenseNet 配置。

## 文件

- `model.py`：DenseNet / DenseNet-BC 模型定义。
- `train_cifar.py`：CIFAR-10 / CIFAR-100 训练脚本。
- `requirements.txt`：运行训练所需依赖。

## 环境

当前机器有 NVIDIA GeForce RTX 5070 Ti，但本机现有 Python 环境还没有安装 `torch` 和 `torchvision`。

安装 PyTorch 时建议打开官方安装页，选择：

- OS：Windows
- Package：Pip
- Language：Python
- Compute Platform：CUDA 12.8 或官方页面当前推荐的较新 CUDA 版本

常见命令形式如下，具体以官方页面生成为准：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 推荐先跑的小实验

先用论文里的轻量配置 DenseNet-BC-100, k=12：

```powershell
python train_cifar.py --dataset cifar10 --model densenet_bc100_k12 --epochs 300 --batch-size 64
```

如果没有显卡，可以先短跑确认流程：

```powershell
python train_cifar.py --dataset cifar10 --model densenet40_k12 --epochs 1 --batch-size 16 --device cpu
```

## 论文配置对应关系

| 论文配置 | 命令参数 |
| --- | --- |
| DenseNet, L=40, k=12 | `--model densenet40_k12` |
| DenseNet, L=100, k=12 | `--model densenet100_k12` |
| DenseNet, L=100, k=24 | `--model densenet100_k24` |
| DenseNet-BC, L=100, k=12 | `--model densenet_bc100_k12` |
| DenseNet-BC, L=250, k=24 | `--model densenet_bc250_k24` |
| DenseNet-BC, L=190, k=40 | `--model densenet_bc190_k40` |

## 训练设置

默认训练设置按论文 CIFAR 实验靠拢：

- optimizer：SGD
- initial learning rate：0.1
- momentum：0.9, Nesterov
- weight decay：1e-4
- epochs：300
- batch size：64
- learning rate decay：总 epoch 的 50% 和 75% 处各除以 10
- CIFAR+ 数据增强：random crop + horizontal flip

默认使用数据增强，因此 dropout 为 0。如果加 `--no-augment`，默认 dropout 会改成 0.2，对应论文中无数据增强设置。
