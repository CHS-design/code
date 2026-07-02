# LESSONS.md

- 训练/测试脚本里的数据集目录和模型权重路径应基于脚本所在目录生成，避免使用 `./data` 这类依赖启动目录的相对路径，也不要写个人电脑上的绝对路径。
- 在模型里使用 Dropout 时优先用 `nn.Dropout` 模块；如果使用 `F.dropout`，必须传入 `training=self.training`，否则 `model.eval()` 后验证阶段仍会随机失活。
