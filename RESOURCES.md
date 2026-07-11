# 交叉熵损失 Resources

## Knowledge

- [Stanford CS231n: Linear Classification](https://cs231n.github.io/linear-classify/)
  经典视觉识别课程的损失函数讲义。用于理解 softmax 分类和交叉熵的概率解释。
- [PyTorch: CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
  PyTorch 的官方 API 文档。用于确认 logits、类别索引、输出形状和实现细节。
- [PyTorch: BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
  PyTorch 的官方二分类损失文档。用于二分类像素分割的 logits、掩码形状和数值稳定实现。
- [Deep Learning Book, Chapter 3](https://www.deeplearningbook.org/contents/prob.html)
  Goodfellow、Bengio、Courville 的概率与信息论章节。用于需要进一步理解负对数似然时查阅。

## Wisdom

- [PyTorch Forums](https://discuss.pytorch.org/)
  PyTorch 官方论坛。用于排查 `CrossEntropyLoss` 的维度、标签类型和数值问题。
