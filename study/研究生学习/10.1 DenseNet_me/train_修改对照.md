# `train.py` 修改对照

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| 模型导入 | `from model import ResNet, Residual` | `from DenSet_40_model import DenseNet` |
| 数据集 | `FashionMNIST` | `CIFAR10` |
| 图片尺寸 | 强制缩放到 `224×224` | 保持 CIFAR-10 的 `32×32` |
| 训练数据处理 | 仅 `ToTensor()` | `RandomCrop(32, padding=4)`、`RandomHorizontalFlip()`、标准化 |
| 测试数据处理 | 从训练集随机划分验证集 | 使用官方 CIFAR-10 测试集，仅标准化 |
| 批大小 | `128` | `64` |
| 训练轮数 | `50` | `300` |
| 优化器 | `Adam(lr=0.001)` | `SGD(lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4)` |
| 学习率调整 | 无 | 在第 `150`、`225` 轮各缩小为原来的 `0.1` 倍 |
| 模型保存 | `best_model.pth`，启动时自动加载后重新训练 | 保存最佳测试准确率权重到 `best_densenet40_cifar10.pth` |
| 训练记录 | 仅内存中的 DataFrame 和弹窗曲线 | 额外保存 `densenet40_cifar10_history.csv` 与 `densenet40_cifar10_curve.png` |
| 小规模验证 | 无 | 可用 `--epochs 1` 先跑一轮，例如 `python train.py --epochs 1` |

## 关键代码变化

### 数据集

```python
# 修改前
train_data = FashionMNIST(..., transforms.Resize(size=224), ...)

# 修改后
train_data = CIFAR10(..., train=True, transform=train_transform, download=True)
test_data = CIFAR10(..., train=False, transform=test_transform, download=True)
```

### 优化器和学习率

```python
# 修改前
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 修改后
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9,
                nesterov=True, weight_decay=1e-4)
scheduler = MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
```

### 模型创建

```python
# 修改前
model = ResNet(Residual)

# 修改后
model = DenseNet(num_layers=12, growth_rate=12, num_classes=10)
```
