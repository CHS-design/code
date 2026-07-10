# Lessons

- DenseNet-40 (k=12) 的训练脚本必须和复现目标保持一致：使用 CIFAR-10 的 `32×32` 数据增强、SGD 和 `150/225` epoch 学习率衰减；不要直接沿用其他网络实验中的 FashionMNIST、`224×224` 输入或 Adam 配置。
