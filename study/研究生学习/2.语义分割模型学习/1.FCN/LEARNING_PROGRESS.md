# FCN 语义分割复现学习进度

## 已完成

- 已完成参考项目的 Python 文件、依赖关系和数据流只读核对。
- 当前尚未开始复现代码。

## 已通过测试

- 已确认最小验证环境可导入 Python 3.12.13、PyTorch 2.11.0、torchvision 0.26.0 和 Pillow 12.2.0：`D:\Environment\miniconda3\envs\dl\python.exe`。

## 暂未理解的知识点

- 暂无。

## 已发现的源码问题

- `requirements.txt` 未声明 `torch` 和 `torchvision`。
- 训练默认使用 20 个前景类（模型总计 21 类），而 LabelMe 脚本定义了 3 个前景类（模型总计应为 4 类）。
- 两个数据划分脚本都没有固定随机种子。
- `255` 被损失函数忽略，但部分评估指标仍将其计入比较或并集。
- `train.py` 在 CPU 情况下仍会无条件调用 `nvidia-smi`。
- `val.py` 重复实现了评估流程。

## 下一步

- `utils/transforms.py`：实现并测试 `pad_if_smaller`。
