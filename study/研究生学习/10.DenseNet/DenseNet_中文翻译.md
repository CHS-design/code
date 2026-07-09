# Densely Connected Convolutional Networks（密集连接卷积网络）中文翻译

原文：Gao Huang、Zhuang Liu、Laurens van der Maaten、Kilian Q. Weinberger，*Densely Connected Convolutional Networks（密集连接卷积网络）*。

说明：本文按原论文结构翻译。公式、数据集名、网络名和部分常用术语保留英文或英文括注，便于后续对照论文和代码学习。

## 摘要

近期研究表明，如果卷积网络在靠近输入的层与靠近输出的层之间包含更短的连接，那么网络可以变得更深、更准确，并且训练效率更高。本文基于这一观察，提出了密集卷积网络 Dense Convolutional Network（密集卷积网络），即 DenseNet（密集卷积网络）。DenseNet（密集卷积网络）以 feed-forward（前馈）的方式将每一层与其他层连接起来。

传统的 L 层卷积网络通常只有 L 条连接，也就是每一层只连接到它的下一层；而 DenseNet（密集卷积网络）具有 \(L(L+1)/2\) 条直接连接。对每一层来说，前面所有层产生的 feature maps（特征图）都会作为它的输入；同时，它自己产生的 feature maps（特征图）也会作为后面所有层的输入。

DenseNet（密集卷积网络）有几个明显优势：缓解梯度消失问题，加强特征传播，鼓励特征复用，并显著减少参数数量。我们在四个具有竞争性的目标识别基准任务上评估了该结构，包括 CIFAR-10（CIFAR-10 图像分类数据集）、CIFAR-100（CIFAR-100 图像分类数据集）、SVHN（街景门牌号数据集）和 ImageNet（大型图像识别数据集）。实验表明，DenseNet（密集卷积网络）在大多数任务上都明显优于当时的 state-of-the-art（当前最优）方法，并且在达到较高性能时所需的计算量更少。代码和预训练模型可在 https://github.com/liuzhuang13/DenseNet 获取。

## 1. 引言

卷积神经网络 CNN（卷积神经网络）已经成为视觉目标识别中的主流机器学习方法。虽然 CNN（卷积神经网络）最早在二十多年前就已经被提出，但真正很深的 CNN（卷积神经网络）直到近年来才因为计算硬件和网络结构的发展而能够被训练出来。最初的 LeNet5（LeNet-5 网络）只有 5 层，VGG（VGG 网络）有 19 层，而就在前一年，Highway Networks（高速公路网络）和 Residual Networks（残差网络），也就是 ResNets（残差网络），已经突破了 100 层的限制。

随着 CNN（卷积神经网络）变得越来越深，一个新的研究问题出现了：输入信息或梯度在经过许多层传播时，可能会在到达网络末端或返回网络前端之前逐渐消失或被冲淡。很多近期工作都在处理这个问题或相关问题。ResNets（残差网络）和 Highway Networks（高速公路网络）通过 identity connections（恒等连接）让信号绕过某些层，从一层直接传到下一层。Stochastic depth（随机深度）在训练时随机丢弃 ResNet（残差网络）中的层，从而缩短网络路径，使信息和梯度更容易流动。FractalNets（分形网络）反复组合多个不同长度的并行卷积块序列，在保持大量短路径的同时获得很大的名义深度。

虽然这些方法在网络拓扑和训练过程上不同，但它们共享一个关键特征：都在从早期层到后期层之间创建短路径。

本文提出的结构把这一洞察浓缩成一种简单的连接模式：为了保证网络各层之间最大程度的信息流动，我们把所有 feature-map（特征图）尺寸匹配的层直接连接起来。为了保持 feed-forward（前馈）的性质，每一层都会从前面所有层获得额外输入，并把自己的 feature maps（特征图）传给后面所有层。图 1 给出了这种布局的示意。

关键的是，DenseNet（密集卷积网络）与 ResNet（残差网络）不同：我们不会在特征进入某一层之前通过求和把它们合并，而是通过拼接 concatenate 的方式合并。因此，第 \(\ell\) 层有 \(\ell\) 个输入，这些输入由前面所有卷积块的 feature maps（特征图）组成。它自己的 feature maps（特征图）会被传给后面的 \(L-\ell\) 层。这样，一个 L 层网络会有 \(L(L+1)/2\) 条连接，而不是传统结构中的 L 条连接。由于这种密集连接模式，我们把该方法称为 Dense Convolutional Network（密集卷积网络），即 DenseNet（密集卷积网络）。

这种密集连接模式有一个看起来可能有些反直觉的效果：它需要的参数比传统卷积网络更少，因为网络不需要反复重新学习冗余的 feature maps（特征图）。传统 feed-forward（前馈）结构可以被看作一种带有状态的算法，状态从一层传到下一层。每一层读取前一层的状态，并把状态写给下一层。它会改变这个状态，但也需要把必须保留的信息继续传递下去。ResNets（残差网络）通过 additive identity transformations（加性恒等变换）显式实现了这种信息保留。

ResNet（残差网络）的一些变体表明，很多层的贡献其实很小，甚至可以在训练时随机丢弃。这让 ResNet（残差网络）的状态有点类似展开的循环神经网络，但 ResNet（残差网络）的参数数量要大得多，因为每一层都有自己的权重。本文提出的 DenseNet（密集卷积网络）结构明确地区分了“添加到网络中的信息”和“需要被保留的信息”。DenseNet（密集卷积网络）的每一层都很窄，例如每层只有 12 个 filters（滤波器），它只向网络的“集体知识”中添加一小组 feature maps（特征图），并保持其余 feature maps（特征图）不变。最终分类器会基于网络中所有 feature maps（特征图）做出决策。

除了更好的参数效率之外，DenseNet（密集卷积网络）的另一个重要优势是改善了整个网络中的信息流和梯度流，因此更容易训练。每一层都可以直接接触来自 loss function（损失函数）的梯度和原始输入信号，从而形成一种隐式的 deep supervision（深度监督）。这有助于训练更深的网络结构。此外，我们还观察到 dense connections 具有正则化效果，可以在训练集较小的任务上减少过拟合。

我们在四个竞争性基准数据集上评估 DenseNet（密集卷积网络），包括 CIFAR-10（CIFAR-10 图像分类数据集）、CIFAR-100（CIFAR-100 图像分类数据集）、SVHN（街景门牌号数据集）和 ImageNet（大型图像识别数据集）。我们的模型通常只需要比现有同等精度算法少得多的参数。此外，在大多数基准任务上，我们显著超过了当时的 state-of-the-art（当前最优）结果。

## 2. 相关工作

自神经网络被提出以来，网络结构的探索就是神经网络研究的一部分。近年来神经网络重新受到关注，也推动了这一研究方向的复兴。现代网络层数不断增加，放大了不同结构之间的差异，也促使研究者探索不同的连接模式，并重新审视一些早期思想。

类似于本文密集网络布局的 cascade structure 早在 20 世纪 80 年代的神经网络文献中就已经被研究过。那项早期工作主要关注以逐层方式训练的全连接多层感知机。之后也有人提出了可用 batch gradient descent 训练的全连接 cascade networks。虽然这类方法在小数据集上有效，但它只能扩展到几百个参数的网络。在若干视觉任务中，研究者发现通过 skip connections（跳跃连接）在 CNN（卷积神经网络）中利用多层级特征是有效的。与本文工作同时期，也有研究提出了一个关于跨层连接网络的纯理论框架。

Highway Networks（高速公路网络）是较早能够有效训练超过 100 层端到端网络的结构之一。它使用 bypassing paths（旁路路径）和 gating units（门控单元），使几百层网络也可以较容易地优化。通常认为这些 bypassing paths（旁路路径）是让非常深的网络更容易训练的关键因素。ResNets（残差网络）进一步支持了这一观点，因为 ResNets（残差网络）使用纯 identity mappings（恒等映射）作为 bypassing paths（旁路路径）。ResNets（残差网络）在许多困难的图像识别、定位和检测任务上取得了突出的结果，例如 ImageNet（大型图像识别数据集）和 COCO object detection（目标检测）。

近期，stochastic depth（随机深度）被提出用于成功训练 1202 层 ResNet（残差网络）。Stochastic depth（随机深度）在训练时随机丢弃层，从而改善深层残差网络的训练。这说明并非所有层都是必要的，同时也表明深层残差网络中存在大量冗余。本文在一定程度上受到了这一观察的启发。使用 pre-activation（预激活）的 ResNets（残差网络）也使超过 1000 层的 state-of-the-art（当前最优）网络更容易训练。

另一条让网络变深之外的路线，是增加网络宽度，例如借助 skip connections（跳跃连接）。GoogLeNet（谷歌网络）使用 Inception module（Inception 模块），把不同尺寸 filters（滤波器）产生的 feature maps（特征图）拼接起来。有研究提出了带有宽泛化 residual blocks（残差块）的 ResNet（残差网络）变体。事实上，只要深度足够，简单增加 ResNet（残差网络）每一层 filters（滤波器）的数量也可以提高性能。FractalNets（分形网络）也通过较宽的网络结构在多个数据集上取得了有竞争力的结果。

DenseNet（密集卷积网络）并不是依靠极深或极宽的结构来获得表达能力，而是通过 feature reuse（特征复用）挖掘网络潜力，从而得到容易训练且参数效率很高的紧凑模型。把不同层学到的 feature maps（特征图）进行拼接，可以增加后续层输入的多样性，并提高效率。这是 DenseNets（DenseNet 网络）与 ResNets（残差网络）的一个主要区别。与同样会拼接不同层特征的 Inception networks（Inception 网络）相比，DenseNets（DenseNet 网络）更简单也更高效。

还有一些值得注意的网络结构创新也取得了有竞争力的结果。Network in Network（网络中的网络）在卷积层 filter 中加入微型多层感知机，用于提取更复杂的特征。Deeply Supervised Network（深度监督网络）通过辅助分类器直接监督内部层，从而加强早期层接收到的梯度。Ladder Networks（梯形网络）在 autoencoders（自编码器）中引入 lateral connections，在半监督学习任务上取得了很高精度。Deeply-Fused Nets（深度融合网络）通过组合不同基础网络的中间层来改善信息流。还有研究表明，在网络中加入用于最小化重建损失的路径，也能改善图像分类模型。

## 3. DenseNets（DenseNet 网络）

考虑一张输入图像 \(x_0\)，它被送入一个卷积网络。该网络包含 L 层，每一层都实现一个非线性变换 \(H_\ell(\cdot)\)，其中 \(\ell\) 表示层索引。\(H_\ell(\cdot)\) 可以是多个操作的组合函数，例如 Batch Normalization（批量归一化），ReLU（修正线性单元），Pooling（池化）或 Convolution（卷积）。我们把第 \(\ell\) 层的输出记为 \(x_\ell\)。

### ResNets（残差网络）

传统卷积 feed-forward（前馈）网络把第 \(\ell\) 层的输出连接为第 \(\ell+1\) 层的输入，因此层之间的转换可以写为：

\[
x_\ell = H_\ell(x_{\ell-1})
\]

ResNets（残差网络）增加了一条 skip-connection（跳跃连接），用 identity function（恒等函数）绕过非线性变换：

\[
x_\ell = H_\ell(x_{\ell-1}) + x_{\ell-1}
\]

ResNets（残差网络）的一个优势是，梯度可以通过 identity function（恒等函数）从后面的层直接流向前面的层。然而，identity function（恒等函数）和 \(H_\ell\) 的输出是通过求和合并的，这可能会阻碍网络中的信息流动。

### 密集连接

为了进一步改善层与层之间的信息流动，我们提出一种不同的连接模式：从任意一层到它之后的所有层都建立直接连接。图 1 给出了 DenseNet（密集卷积网络）的示意结构。因此，第 \(\ell\) 层会接收所有前面层的 feature maps（特征图），即 \(x_0, ..., x_{\ell-1}\)，作为输入：

\[
x_\ell = H_\ell([x_0, x_1, ..., x_{\ell-1}])
\]

其中 \([x_0, x_1, ..., x_{\ell-1}]\) 表示第 0 层到第 \(\ell-1\) 层产生的 feature maps（特征图）的拼接。由于这种密集连接，我们把该网络结构称为 Dense Convolutional Network（密集卷积网络），即 DenseNet（密集卷积网络）。为了便于实现，我们把 \(H_\ell(\cdot)\) 的多个输入在公式中拼接成一个 tensor（张量）。

### 组合函数

受相关工作的启发，我们把 \(H_\ell(\cdot)\) 定义为三个连续操作组成的组合函数：batch normalization（批量归一化），即 BN（批量归一化）；随后是 rectified linear unit，即 ReLU（修正线性单元）；再随后是一个 \(3 \times 3\) convolution（卷积），即 Conv（卷积）。

### Pooling（池化）层

当 feature maps（特征图）的尺寸发生变化时，公式中的拼接操作无法直接使用。然而，下采样层是卷积网络的重要组成部分，它会改变 feature maps（特征图）的尺寸。为了在本文结构中实现下采样，我们把网络划分成多个 densely connected dense blocks（密集块），如图 2 所示。块与块之间的层称为 transition layers（过渡层），它们执行 convolution（卷积）和 pooling（池化）。本文实验中的 transition layers（过渡层）由 batch normalization（批量归一化）层、\(1 \times 1\) convolution（卷积）层以及随后一个 \(2 \times 2\) average pooling（平均池化）层组成。

### Growth rate（增长率）

如果每个函数 \(H_\ell\) 产生 k 个 feature maps（特征图），那么第 \(\ell\) 层会有 \(k_0 + k \times (\ell - 1)\) 个输入 feature maps（特征图），其中 \(k_0\) 是输入层的通道数。DenseNet（密集卷积网络）与已有网络结构的一个重要区别是，DenseNet（密集卷积网络）的层可以非常窄，例如 \(k = 12\)。我们把超参数 k 称为网络的 growth rate（增长率）。

第 4 节会显示，相对较小的 growth rate（增长率）就足以在我们测试的数据集上取得 state-of-the-art（当前最优）结果。一个解释是，每一层都可以访问它所在 block 中所有先前的 feature maps（特征图），因此也就能访问网络的“集体知识”。可以把 feature maps（特征图）看作网络的全局状态。每一层向这个状态中添加 k 个自己的 feature maps（特征图）。Growth rate（增长率）控制每一层向全局状态贡献多少新信息。全局状态一旦写入，就可以在网络内部任意位置被访问；与传统网络结构不同，它不需要从一层复制到下一层。

### Bottleneck layers（瓶颈层）

虽然每一层只产生 k 个输出 feature maps（特征图），但它通常有更多输入。有研究指出，可以在每个 \(3 \times 3\) convolution（卷积）之前引入一个 \(1 \times 1\) convolution（卷积）作为 bottleneck layer（瓶颈层），用来减少输入 feature maps（特征图）的数量，从而提升计算效率。我们发现这种设计对 DenseNet（密集卷积网络）尤其有效。我们把带有这种 bottleneck layer（瓶颈层）的网络称为 DenseNet-B（带瓶颈层的 DenseNet），也就是 \(H_\ell\) 采用 BN-ReLU-Conv(\(1 \times 1\))-BN-ReLU-Conv(\(3 \times 3\)) 的版本。在实验中，我们让每个 \(1 \times 1\) convolution（卷积）产生 \(4k\) 个 feature maps（特征图）。

### Compression（压缩）

为了进一步提升模型紧凑性，我们可以在 transition layers（过渡层）处减少 feature maps（特征图）的数量。如果一个 dense block（密集块）包含 m 个 feature maps（特征图），我们让后续 transition layer（过渡层）产生 \(\lfloor \theta m \rfloor\) 个输出 feature maps（特征图），其中 \(0 < \theta \le 1\) 称为 compression factor（压缩因子）。当 \(\theta = 1\) 时，transition layers（过渡层）前后的 feature maps（特征图）数量保持不变。我们把 \(\theta < 1\) 的 DenseNet（密集卷积网络）称为 DenseNet-C（带压缩的 DenseNet），并在实验中设置 \(\theta = 0.5\)。当 bottleneck（瓶颈）和带有 \(\theta < 1\) 的 transition layers（过渡层）同时使用时，我们称该模型为 DenseNet-BC（带瓶颈层和压缩的 DenseNet）。

### 实现细节

除 ImageNet（大型图像识别数据集）外，本文在所有数据集上使用的 DenseNet（密集卷积网络）都包含三个 dense blocks（密集块），并且每个 dense block（密集块）的层数相同。在进入第一个 dense block（密集块）之前，我们先对输入图像执行一次 convolution（卷积），输出通道数为 16；对于 DenseNet-BC（带瓶颈层和压缩的 DenseNet），则输出通道数为 growth rate（增长率）的两倍。对 kernel size（卷积核大小）为 \(3 \times 3\) 的卷积层，我们在输入的每一边都 zero-pad（零填充）一个像素，以保持 feature-map（特征图）尺寸不变。

两个相邻 dense blocks（密集块）之间使用 \(1 \times 1\) convolution（卷积）加 \(2 \times 2\) average pooling（平均池化）作为 transition layers（过渡层）。在最后一个 dense block（密集块）结束后，执行 global average pooling（全局平均池化），然后接一个 softmax classifier（softmax 分类器）。三个 dense blocks（密集块）中 feature-map（特征图）的尺寸分别为 \(32 \times 32\)、\(16 \times 16\) 和 \(8 \times 8\)。

我们测试了基础 DenseNet（密集卷积网络）结构的若干配置：\(\{L = 40, k = 12\}\)、\(\{L = 100, k = 12\}\) 和 \(\{L = 100, k = 24\}\)。对于 DenseNet-BC（带瓶颈层和压缩的 DenseNet），我们评估了 \(\{L = 100, k = 12\}\)、\(\{L = 250, k = 24\}\) 和 \(\{L = 190, k = 40\}\)。

在 ImageNet（大型图像识别数据集）实验中，我们使用 DenseNet-BC（带瓶颈层和压缩的 DenseNet）结构，输入图像大小为 \(224 \times 224\)，网络包含 4 个 dense blocks（密集块）。初始 convolution（卷积）层由 \(2k\) 个大小为 \(7 \times 7\)、stride（步幅）为 2 的 convolutions（卷积）组成；其他所有层的 feature maps（特征图）数量也由 k 决定。ImageNet（大型图像识别数据集）上的具体网络配置见表 1。

### 表 1：ImageNet（大型图像识别数据集）上的 DenseNet（密集卷积网络）结构

所有网络的 growth rate（增长率）均为 \(k = 32\)。表中每个 “conv” 层都对应 BN-ReLU-Conv（批量归一化-ReLU-卷积）这一序列。

| 部分 | 输出尺寸 | DenseNet-121 | DenseNet-169 | DenseNet-201 | DenseNet-264 |
|---|---:|---:|---:|---:|---:|
| Convolution（卷积） | \(112 \times 112\) | \(7 \times 7\) conv, stride（步幅） 2 | 同左 | 同左 | 同左 |
| Pooling（池化） | \(56 \times 56\) | \(3 \times 3\) max pool（最大池化）, stride（步幅） 2 | 同左 | 同左 | 同左 |
| Dense Block (1) | \(56 \times 56\) | \([1 \times 1 \text{ conv}; 3 \times 3 \text{ conv}] \times 6\) | \(\times 6\) | \(\times 6\) | \(\times 6\) |
| Transition Layer (1) | \(56 \times 56 \to 28 \times 28\) | \(1 \times 1\) conv；\(2 \times 2\) average pool（平均池化）, stride（步幅） 2 | 同左 | 同左 | 同左 |
| Dense Block (2) | \(28 \times 28\) | \(\times 12\) | \(\times 12\) | \(\times 12\) | \(\times 12\) |
| Transition Layer (2) | \(28 \times 28 \to 14 \times 14\) | \(1 \times 1\) conv；\(2 \times 2\) average pool（平均池化）, stride（步幅） 2 | 同左 | 同左 | 同左 |
| Dense Block (3) | \(14 \times 14\) | \(\times 24\) | \(\times 32\) | \(\times 48\) | \(\times 64\) |
| Transition Layer (3) | \(14 \times 14 \to 7 \times 7\) | \(1 \times 1\) conv；\(2 \times 2\) average pool（平均池化）, stride（步幅） 2 | 同左 | 同左 | 同左 |
| Dense Block (4) | \(7 \times 7\) | \(\times 16\) | \(\times 32\) | \(\times 32\) | \(\times 48\) |
| Classification Layer（分类层） | \(1 \times 1\) | \(7 \times 7\) global average pool（全局平均池化）；1000D fully-connected, softmax（归一化指数函数） | 同左 | 同左 | 同左 |

## 4. 实验

我们在多个基准数据集上通过实验展示 DenseNet（密集卷积网络）的有效性，并与 state-of-the-art（当前最优）结构进行比较，尤其是与 ResNet（残差网络）及其变体进行比较。

### 4.1 数据集

**CIFAR（CIFAR 图像数据集）。** 两个 CIFAR（CIFAR 图像数据集）数据集由 \(32 \times 32\) 像素的彩色自然图像组成。CIFAR-10（CIFAR-10 图像分类数据集），即 C10，包含 10 个类别；CIFAR-100（CIFAR-100 图像分类数据集），即 C100，包含 100 个类别。训练集和测试集分别包含 50,000 和 10,000 张图像，我们从训练集中留出 5,000 张作为 validation set（验证集）。我们采用这两个数据集上广泛使用的标准数据增强方案，即 mirroring（镜像）/shifting（平移）。文中用数据集名称后面的 “+” 表示使用该数据增强方案，例如 C10+。预处理时，我们使用各通道均值和标准差对数据进行归一化。最终运行时，我们使用全部 50,000 张训练图像，并报告训练结束时的最终 test error（测试错误率）。

**SVHN（街景门牌号数据集）。** Street View House Numbers（街景门牌号数据集），即 SVHN（街景门牌号数据集）数据集，包含 \(32 \times 32\) 的彩色数字图像。训练集有 73,257 张图像，测试集有 26,032 张图像，另外还有 531,131 张额外训练图像。按照常见做法，我们使用全部训练数据，不做数据增强，并从训练集中划分 6,000 张图像作为 validation set（验证集）。训练过程中，我们选择 validation error（验证错误率）最低的模型，并报告 test error（测试错误率）。我们按照相关工作把像素值除以 255，使其位于 \([0, 1]\) 范围内。

**ImageNet（大型图像识别数据集）。** ILSVRC 2012（ImageNet 大规模视觉识别挑战赛 2012）分类数据集包含来自 1000 个类别的 120 万张训练图像和 50,000 张验证图像。训练图像采用与相关 ResNet（残差网络）工作相同的数据增强方案；测试时使用尺寸为 \(224 \times 224\) 的 single-crop（单裁剪）或 10-crop（十裁剪）。按照已有工作，我们报告 validation set（验证集）上的分类错误率。

### 4.2 训练

所有网络都使用 stochastic gradient descent（随机梯度下降），即 SGD（随机梯度下降）训练。在 CIFAR（CIFAR 图像数据集）和 SVHN（街景门牌号数据集）上，我们分别使用 batch size（批大小） 64 训练 300 个 epoch（训练轮次）和 40 个 epoch（训练轮次）。初始 learning rate（学习率）设置为 0.1，并在总训练 epoch（训练轮次）的 50% 和 75% 处分别除以 10。在 ImageNet（大型图像识别数据集）上，我们使用 batch size（批大小） 256 训练 90 个 epoch（训练轮次）。初始 learning rate（学习率）为 0.1，并在第 30 和第 60 个 epoch（训练轮次）时分别降低 10 倍。

需要注意的是，DenseNet（密集卷积网络）的朴素实现可能存在 memory inefficiencies。为了降低 GPU 上的显存消耗，可以参考作者关于 DenseNet（密集卷积网络） memory-efficient implementation 的技术报告。

按照相关设置，我们使用 \(10^{-4}\) 的 weight decay（权重衰减）和 0.9 的 Nesterov momentum（Nesterov 动量），且不使用 dampening（阻尼）。权重初始化采用已有方法。对于三个没有数据增强的数据集，即 C10、C100 和 SVHN（街景门牌号数据集），我们在每个 convolutional layer（卷积层）后加入 dropout layer（随机失活层），但第一层除外，并把 dropout rate（随机失活率）设置为 0.2。每个任务和模型设置下，test error（测试错误率）只评估一次。

### 4.3 CIFAR（CIFAR 图像数据集）和 SVHN（街景门牌号数据集）上的分类结果

我们训练了不同深度 L 和不同 growth rate（增长率） k 的 DenseNets（DenseNet 网络）。CIFAR（CIFAR 图像数据集）和 SVHN（街景门牌号数据集）上的主要结果见表 2。为了突出总体趋势，论文中把超过已有 state-of-the-art（当前最优）的结果用粗体表示，把整体最佳结果用蓝色表示。

#### 表 2：CIFAR（CIFAR 图像数据集）和 SVHN（街景门牌号数据集）数据集上的错误率

k 表示网络的 growth rate（增长率）。“+” 表示使用标准数据增强，即平移和/或镜像。带星号的结果为作者自己运行得到。没有数据增强的 DenseNet（密集卷积网络）结果，即 C10、C100 和 SVHN（街景门牌号数据集），均使用 Dropout（随机失活）。DenseNet（密集卷积网络）在使用更少参数的同时取得了比 ResNet（残差网络）更低的错误率；在没有数据增强时，DenseNet（密集卷积网络）的优势更明显。

| Method（方法） | Depth（深度） | Params（参数量） | C10 | C10+ | C100 | C100+ | SVHN（街景门牌号数据集） |
|---|---:|---:|---:|---:|---:|---:|---:|
| Network in Network（网络中的网络） | - | - | 10.41 | 8.81 | 35.68 | - | 2.35 |
| All-CNN | - | - | 9.08 | 7.25 | - | 33.71 | - |
| Deeply Supervised Net（深度监督网络） | - | - | 9.69 | 7.97 | - | 34.57 | 1.92 |
| Highway Network | - | - | - | 7.72 | - | 32.39 | - |
| FractalNet（分形网络） | 21 | 38.6M | 10.18 | 5.22 | 35.34 | 23.30 | 2.01 |
| FractalNet（分形网络） with Dropout（随机失活）/Drop-path | 21 | 38.6M | 7.33 | 4.60 | 28.20 | 23.73 | 1.87 |
| ResNet（残差网络） | 110 | 1.7M | - | 6.61 | - | - | - |
| ResNet（残差网络） reported by stochastic depth（随机深度） work | 110 | 1.7M | 13.63 | 6.41 | 44.74 | 27.22 | 2.01 |
| ResNet（残差网络） with Stochastic Depth（深度） | 110 | 1.7M | 11.66 | 5.23 | 37.80 | 24.58 | 1.75 |
| ResNet（残差网络） with Stochastic Depth（深度） | 1202 | 10.2M | - | 4.91 | - | - | - |
| Wide ResNet（宽残差网络） | 16 | 11.0M | - | 4.81 | - | 22.07 | - |
| Wide ResNet（宽残差网络） | 28 | 36.5M | - | 4.17 | - | 20.50 | - |
| Wide ResNet（宽残差网络） with Dropout（随机失活） | 16 | 2.7M | - | - | - | - | 1.64 |
| ResNet（残差网络） pre-activation（预激活） | 164 | 1.7M | 11.26 | 5.46 | 35.58 | 24.33 | - |
| ResNet（残差网络） pre-activation（预激活） | 1001 | 10.2M | 10.56 | 4.62 | 33.47 | 22.71 | - |
| DenseNet（密集卷积网络） \(k = 12\) | 40 | 1.0M | 7.00 | 5.24 | 27.55 | 24.42 | 1.79 |
| DenseNet（密集卷积网络） \(k = 12\) | 100 | 7.0M | 5.77 | 4.10 | 23.79 | 20.20 | 1.67 |
| DenseNet（密集卷积网络） \(k = 24\) | 100 | 27.2M | 5.83 | 3.74 | 23.42 | 19.25 | 1.59 |
| DenseNet-BC（带瓶颈层和压缩的 DenseNet） \(k = 12\) | 100 | 0.8M | 5.92 | 4.51 | 24.15 | 22.27 | 1.76 |
| DenseNet-BC（带瓶颈层和压缩的 DenseNet） \(k = 24\) | 250 | 15.3M | 5.19 | 3.62 | 19.64 | 17.60 | 1.74 |
| DenseNet-BC（带瓶颈层和压缩的 DenseNet） \(k = 40\) | 190 | 25.6M | - | 3.46 | - | 17.18 | - |

**准确率。** 表 2 最底部一行可能展示了最明显的趋势：\(L = 190, k = 40\) 的 DenseNet-BC（带瓶颈层和压缩的 DenseNet）在所有 CIFAR（CIFAR 图像数据集）数据集上都稳定超过了已有 state-of-the-art（当前最优）。它在 C10+ 上的错误率为 3.46%，在 C100+ 上为 17.18%，明显低于 Wide ResNet（宽残差网络）结构的错误率。我们在没有数据增强的 C10 和 C100 上取得的最佳结果更令人鼓舞：二者相较带有 drop-path regularization（随机路径正则化）的 FractalNet（分形网络）都接近 30% 的相对错误率下降。在 SVHN（街景门牌号数据集）上，使用 dropout（随机失活）的 \(L = 100, k = 24\) DenseNet（密集卷积网络）也超过了 Wide ResNet（宽残差网络）当时取得的最佳结果。不过，250 层 DenseNet-BC（带瓶颈层和压缩的 DenseNet）并没有比更短的模型进一步提升性能。这可能是因为 SVHN（街景门牌号数据集）是相对容易的任务，极深模型可能会对训练集过拟合。

**容量。** 在不使用 compression（压缩）或 bottleneck layers（瓶颈层）的情况下，DenseNet（密集卷积网络）的整体趋势是 L 和 k 越大，性能越好。我们主要把这归因于相应的模型容量增长。C10+ 和 C100+ 两列最能说明这一点。在 C10+ 上，当参数数量从 1.0M 增加到 7.0M，再增加到 27.2M 时，错误率从 5.24% 降到 4.10%，最后降到 3.74%。C100+ 上也能观察到类似趋势。这说明 DenseNet（密集卷积网络）能够利用更大、更深模型带来的表达能力提升，也说明它没有表现出 residual networks（残差网络）中可能出现的过拟合或优化困难。

**参数效率。** 表 2 的结果表明，相比其他结构，特别是 ResNets（残差网络），DenseNets（DenseNet 网络）对参数的利用更高效。带有 bottleneck（瓶颈）结构并在 transition layers（过渡层）进行维度缩减的 DenseNet-BC（带瓶颈层和压缩的 DenseNet）尤其参数高效。例如，250 层模型只有 15.3M 参数，但它稳定超过了 FractalNet（分形网络）和 Wide ResNets（残差网络）等超过 30M 参数的模型。还值得注意的是，\(L = 100, k = 12\) 的 DenseNet-BC（带瓶颈层和压缩的 DenseNet）使用少 90% 的参数，就能达到与 1001 层 pre-activation ResNet（预激活残差网络）相近的性能，例如 C10+ 上 4.51% 对 4.62%，C100+ 上 22.27% 对 22.71%。图 4 右侧展示了这两个网络在 C10+ 上的 training loss（训练损失）和 test error（测试错误率）。1001 层深 ResNet（残差网络）收敛到更低的 training loss（训练损失），但 test error（测试错误率）相近。论文之后会进一步分析这一现象。

**过拟合。** 更高效使用参数的一个积极副作用是，DenseNet（密集卷积网络）往往更不容易过拟合。我们观察到，在没有数据增强的数据集上，DenseNet（密集卷积网络）相比已有工作的提升尤其明显。在 C10 上，错误率从 7.33% 降到 5.19%，相对下降 29%。在 C100 上，错误率从 28.20% 降到 19.64%，下降约 30%。在实验中，我们只在一种设置下观察到了潜在过拟合：在 C10 上，把 k 从 12 增加到 24 导致参数数量增长 4 倍，但错误率从 5.77% 小幅上升到 5.83%。DenseNet-BC（带瓶颈层和压缩的 DenseNet）中的 bottleneck（瓶颈）和 compression layers（压缩层）看起来能有效缓解这一趋势。

### 4.4 ImageNet（大型图像识别数据集）上的分类结果

我们在 ImageNet（大型图像识别数据集）分类任务上评估了不同深度和 growth rate（增长率）的 DenseNet-BC（带瓶颈层和压缩的 DenseNet），并与 state-of-the-art（当前最优） ResNet（残差网络）结构比较。为了保证两种结构之间的公平比较，我们采用公开的 Torch ResNet（残差网络）实现，从而消除数据预处理和优化设置差异等其他因素。我们只是把 ResNet（残差网络）模型替换成 DenseNet-BC（带瓶颈层和压缩的 DenseNet）网络，并保持所有实验设置与 ResNet（残差网络）完全相同。

表 3 报告了 DenseNet（密集卷积网络）在 ImageNet（大型图像识别数据集）上的 single-crop（单裁剪）和 10-crop（十裁剪） validation errors（验证错误率）。图 3 展示了 DenseNet（密集卷积网络）和 ResNet（残差网络）的 single-crop（单裁剪） top-1（第一分类错误率） validation errors（验证错误率）随参数数量和 FLOPs（浮点运算量）变化的情况。图中结果表明，DenseNet（密集卷积网络）的性能可以与 state-of-the-art（当前最优） ResNets（残差网络）相当，但达到相近性能时需要的参数和计算量明显更少。

例如，具有 20M 参数的 DenseNet-201 可以得到与超过 40M 参数的 101 层 ResNet（残差网络）相近的 validation error（验证错误率）。从右图也可以看到类似趋势：如果一个 DenseNet（密集卷积网络）的计算量与 ResNet-50 相当，那么它的表现可以接近计算量约为其两倍的 ResNet-101。

值得注意的是，本文的实验设置意味着我们使用的是针对 ResNets（残差网络）优化过的超参数，而不是针对 DenseNets（DenseNet 网络）专门优化的超参数。因此，可以合理推测，更充分的 hyper-parameter search（超参数搜索）可能会进一步提升 DenseNet（密集卷积网络）在 ImageNet（大型图像识别数据集）上的性能。

#### 表 3：ImageNet（大型图像识别数据集） validation set（验证集）上的 top-1（第一分类错误率）和 top-5（前五分类错误率）错误率

结果格式为 single-crop（单裁剪） / 10-crop（十裁剪） testing（测试）。

| Model | top-1（第一分类错误率） | top-5（前五分类错误率） |
|---|---:|---:|
| DenseNet-121 | 25.02 / 23.61 | 7.71 / 6.66 |
| DenseNet-169 | 23.80 / 22.08 | 6.85 / 5.92 |
| DenseNet-201 | 22.58 / 21.46 | 6.34 / 5.54 |
| DenseNet-264 | 22.15 / 20.80 | 6.12 / 5.29 |

## 5. 讨论

从表面上看，DenseNets（DenseNet 网络）与 ResNets（残差网络）很相似：公式 2 与公式 1 的差别似乎只是 \(H_\ell(\cdot)\) 的输入由求和改成了拼接。然而，这个看似很小的改动会导致两种网络结构表现出实质上不同的行为。

### 模型紧凑性

由于输入采用拼接，DenseNet（密集卷积网络）任意一层学到的 feature maps（特征图）都可以被后续所有层访问。这鼓励了整个网络中的 feature reuse（特征复用），并产生更紧凑的模型。

图 4 左侧两个图展示了一个实验结果，该实验比较了所有 DenseNet（密集卷积网络）变体的参数效率，以及一个可比较的 ResNet（残差网络）结构的参数效率。我们在 C10+ 上训练了多个不同深度的小网络，并把它们的 test accuracy（测试准确率）作为网络参数数量的函数画出来。与 AlexNet 或 VGG-net（VGG 网络）等流行网络相比，使用 pre-activation（预激活）的 ResNets（残差网络）参数更少且通常结果更好。因此，我们把 \(k = 12\) 的 DenseNet（密集卷积网络）与这种结构比较。DenseNet（密集卷积网络）的训练设置与上一节保持一致。

图中显示，DenseNet-BC（带瓶颈层和压缩的 DenseNet）始终是 DenseNet（密集卷积网络）各变体中参数效率最高的版本。此外，要达到相同精度水平，DenseNet-BC（带瓶颈层和压缩的 DenseNet）大约只需要 ResNet（残差网络）三分之一的参数。这个结果与图 3 中 ImageNet（大型图像识别数据集）上的结果一致。图 4 右侧还显示，只有 0.8M 可训练参数的 DenseNet-BC（带瓶颈层和压缩的 DenseNet）能够达到与拥有 10.2M 参数的 1001 层 pre-activation ResNet（预激活残差网络）相近的精度。

### 隐式深度监督

Dense convolutional networks 精度提升的一个解释是，每个单独的层都可以通过更短的连接从 loss function（损失函数）获得额外监督。可以把 DenseNet（密集卷积网络）理解为执行了一种 deep supervision（深度监督）。Deep supervision 的好处此前已经在 Deeply-Supervised Nets（深度监督网络）中展示过，这类网络给每个 hidden layer 都接上分类器，迫使中间层学习具有判别性的特征。

DenseNet（密集卷积网络）以隐式方式执行类似的 deep supervision（深度监督）：网络顶部只有一个分类器，但它可以通过最多两三个 transition layers（过渡层）向所有层提供直接监督。不过，DenseNet（密集卷积网络）的 loss function（损失函数）和梯度要简单得多，因为所有层共享同一个 loss function（损失函数）。

### 随机连接与确定性连接

Dense convolutional networks 与 residual networks（残差网络）的 stochastic depth（随机深度） regularization（正则化）之间存在一个有趣联系。在 stochastic depth（随机深度）中，residual networks（残差网络）的层会被随机丢弃，从而在被丢弃层两侧的层之间创建直接连接。由于 pooling（池化） layers 永远不会被丢弃，网络最终会产生一种与 DenseNet（密集卷积网络）类似的连接模式：在同一组 pooling（池化） layers 之间，任意两层都有一个小概率被直接连接，前提是它们中间的所有层都被随机丢弃。虽然这两种方法最终有很大差异，但从 DenseNet（密集卷积网络）角度理解 stochastic depth（随机深度），可能有助于解释这种正则化方法为什么有效。

### 特征复用

从设计上看，DenseNet（密集卷积网络）允许每一层访问它之前所有层的 feature maps（特征图），虽然有时需要经过 transition layers（过渡层）。我们进行了一项实验，研究训练好的网络是否真的利用了这一机会。我们首先在 C10+ 上训练一个 \(L = 40, k = 12\) 的 DenseNet（密集卷积网络）。然后，对一个 block 内的每个 convolutional layer（卷积层） \(\ell\)，我们计算它分配给不同层连接的平均绝对权重。

图 5 展示了三个 dense blocks（密集块）的 heat-map（热力图）。平均绝对权重可以作为某个 convolutional layer（卷积层）对其前面各层依赖程度的替代指标。位置 \((\ell, s)\) 上的红点表示第 \(\ell\) 层平均而言强烈使用了前面第 s 层产生的 feature maps（特征图）。图中可以得到几个观察：

1. 所有层都会把权重分散到同一个 block 内的许多输入上。这说明很早的层提取的特征确实会被同一个 dense block（密集块）中更深的层直接使用。
2. Transition layers 的权重也分布在前一个 dense block（密集块）内的所有层上，说明信息可以通过很少的中间环节从 DenseNet（密集卷积网络）的前几层流向最后几层。
3. 第二和第三个 dense block（密集块）内的层始终给 transition layer（过渡层）的输出分配最小权重，也就是三角形最上面一行。这说明 transition layer（过渡层）输出了很多冗余特征，平均权重较低。这与 DenseNet-BC（带瓶颈层和压缩的 DenseNet）的强结果一致，因为 DenseNet-BC（带瓶颈层和压缩的 DenseNet）正是压缩这些输出。
4. 最终 classification layer（分类层），也就是图中最右侧部分，也会使用整个 dense block（密集块）中的权重，但权重似乎更集中在最后的 feature maps（特征图）上。这表明网络后期可能确实产生了一些更高级的特征。

## 6. 结论

本文提出了一种新的卷积网络结构，称为 Dense Convolutional Network（密集卷积网络），即 DenseNet（密集卷积网络）。它在任意两个具有相同 feature-map（特征图）尺寸的层之间引入直接连接。我们展示了 DenseNet（密集卷积网络）可以自然扩展到数百层，而且没有出现优化困难。在实验中，随着参数数量增加，DenseNet（密集卷积网络）通常能稳定提高精度，没有表现出性能退化或过拟合迹象。在多种设置下，它在几个竞争性很强的数据集上取得了 state-of-the-art（当前最优）结果。

此外，DenseNet（密集卷积网络）达到 state-of-the-art（当前最优）性能所需的参数和计算量都显著更少。由于本文采用的是针对 residual networks（残差网络）优化的超参数设置，我们相信，通过更细致地调整超参数和 learning rate（学习率） schedules（学习率调度），DenseNet（密集卷积网络）的精度还有可能进一步提升。

DenseNet（密集卷积网络）遵循一条简单的连接规则，但自然融合了 identity mappings（恒等映射）、deep supervision（深度监督）和 diversified depth 等特性。它允许整个网络进行 feature reuse（特征复用），因此可以学习更紧凑、并且根据本文实验也更准确的模型。由于其内部表示紧凑且特征冗余减少，DenseNet（密集卷积网络）可能适合作为各种基于卷积特征的计算机视觉任务的 feature extractor（特征提取器）。作者计划在未来工作中研究 DenseNet（密集卷积网络）的这种 feature transfer（特征迁移）。

## 致谢

作者部分受到 NSF III-1618134、III-1526012、IIS-1149882，美国海军研究办公室 Grant N00014-17-1-2175，以及 Bill and Melinda Gates foundation 的支持。GH 受到中国博士后国际交流计划 Fellowship Program 的支持，编号 No.20150015。ZL 受到中国国家重点基础研究发展计划 Grants 2011CBA00300、2011CBA00301，以及 NSFC 61361136003 的支持。作者还感谢 Daniel Sedra、Geoff Pleiss 和 Yu Sun 的许多有启发性的讨论。

## 图表说明翻译

**图 1。** 一个 5 层 dense block（密集块），growth rate（增长率）为 \(k = 4\)。每一层都把前面所有 feature maps（特征图）作为输入。

**图 2。** 一个具有三个 dense blocks（密集块）的深层 DenseNet（密集卷积网络）。两个相邻 blocks 之间的层称为 transition layers（过渡层），它们通过 convolution（卷积）和 pooling（池化）改变 feature-map（特征图）尺寸。

**图 3。** 在 ImageNet（大型图像识别数据集） validation dataset（验证数据集）上，比较 DenseNets（DenseNet 网络）和 ResNets（残差网络）的 top-1（第一分类错误率） error rates（错误率），即 single-crop（单裁剪） testing（测试）。左图以 learned parameters（学习到的参数）数量为横轴，右图以 test-time（测试阶段） FLOPs（浮点运算量）为横轴。

**图 4。** 左图：在 C10+ 上比较 DenseNet（密集卷积网络）各变体的参数效率。中图：比较 DenseNet-BC（带瓶颈层和压缩的 DenseNet）和 pre-activation（预激活） ResNets（残差网络）的参数效率；DenseNet-BC（带瓶颈层和压缩的 DenseNet）达到相近精度大约只需要 ResNet（残差网络）三分之一的参数。右图：比较拥有超过 10M 参数的 1001 层 pre-activation ResNet（预激活残差网络）与只有 0.8M 参数的 100 层 DenseNet（密集卷积网络）的训练曲线和测试曲线。

**图 5。** 一个训练好的 DenseNet（密集卷积网络）中，卷积层 filter weights（滤波器权重）的平均绝对值。像素 \((s, \ell)\) 的颜色表示在一个 dense block（密集块）内，从 convolutional layer（卷积层） s 连接到 layer \(\ell\) 的权重的平均 L1 norm，并按输入 feature maps（特征图）数量归一化。黑色矩形标出的三列分别对应两个 transition layers（过渡层）和 classification layer（分类层）。第一行表示与 dense block（密集块）输入层相连的权重。

## 参考文献

参考文献部分保留原文编号与英文题名，便于后续按论文编号查找。

[1] C. Cortes, X. Gonzalvo, V. Kuznetsov, M. Mohri, and S. Yang. Adanet: Adaptive structural learning of artificial neural networks. arXiv preprint arXiv:1607.01097, 2016.

[2] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.

[3] S. E. Fahlman and C. Lebiere. The cascade-correlation learning architecture. In NIPS, 1989.

[4] J. R. Gardner, M. J. Kusner, Y. Li, P. Upchurch, K. Q. Weinberger, and J. E. Hopcroft. Deep manifold traversal: Changing labels with convolutional features. arXiv preprint arXiv:1511.06421, 2015.

[5] L. Gatys, A. Ecker, and M. Bethge. A neural algorithm of artistic style. Nature Communications, 2015.

[6] X. Glorot, A. Bordes, and Y. Bengio. Deep sparse rectifier neural networks. In AISTATS, 2011.

[7] I. Goodfellow, D. Warde-Farley, M. Mirza, A. Courville, and Y. Bengio. Maxout networks. In ICML, 2013.

[8] S. Gross and M. Wilber. Training and investigating residual nets, 2016.

[9] B. Hariharan, P. Arbelaez, R. Girshick, and J. Malik. Hypercolumns for object segmentation and fine-grained localization. In CVPR, 2015.

[10] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In ICCV, 2015.

[11] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[12] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.

[13] G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger. Deep networks with stochastic depth. In ECCV, 2016.

[14] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, 2015.

[15] A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. Tech Report, 2009.

[16] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.

[17] G. Larsson, M. Maire, and G. Shakhnarovich. Fractalnet: Ultra-deep neural networks without residuals. arXiv preprint arXiv:1605.07648, 2016.

[18] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural computation, 1(4):541-551, 1989.

[19] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.

[20] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply-supervised nets. In AISTATS, 2015.

[21] Q. Liao and T. Poggio. Bridging the gaps between residual learning, recurrent neural networks and visual cortex. arXiv preprint arXiv:1604.03640, 2016.

[22] M. Lin, Q. Chen, and S. Yan. Network in network. In ICLR, 2014.

[23] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.

[24] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng. Reading digits in natural images with unsupervised feature learning. In NIPS Workshop, 2011.

[25] M. Pezeshki, L. Fan, P. Brakel, A. Courville, and Y. Bengio. Deconstructing the ladder network architecture. In ICML, 2016.

[26] G. Pleiss, D. Chen, G. Huang, T. Li, L. van der Maaten, and K. Q. Weinberger. Memory-efficient implementation of densenets. arXiv preprint arXiv:1707.06990, 2017.

[27] A. Rasmus, M. Berglund, M. Honkala, H. Valpola, and T. Raiko. Semi-supervised learning with ladder networks. In NIPS, 2015.

[28] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio. Fitnets: Hints for thin deep nets. In ICLR, 2015.

[29] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. IJCV.

[30] P. Sermanet, S. Chintala, and Y. LeCun. Convolutional neural networks applied to house numbers digit classification. In ICPR, pages 3288-3291. IEEE, 2012.

[31] P. Sermanet, K. Kavukcuoglu, S. Chintala, and Y. LeCun. Pedestrian detection with unsupervised multi-stage feature learning. In CVPR, 2013.

[32] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806, 2014.

[33] N. Srivastava, G. E. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. JMLR, 2014.

[34] R. K. Srivastava, K. Greff, and J. Schmidhuber. Training very deep networks. In NIPS, 2015.

[35] I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the importance of initialization and momentum in deep learning. In ICML, 2013.

[36] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

[37] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.

[38] S. Targ, D. Almeida, and K. Lyman. Resnet in resnet: Generalizing residual architectures. arXiv preprint arXiv:1603.08029, 2016.

[39] J. Wang, Z. Wei, T. Zhang, and W. Zeng. Deeply-fused nets. arXiv preprint arXiv:1605.07716, 2016.

[40] B. M. Wilamowski and H. Yu. Neural network learning without backpropagation. IEEE Transactions on Neural Networks, 21(11):1793-1803, 2010.

[41] S. Yang and D. Ramanan. Multi-scale recognition with dag-cnns. In ICCV, 2015.

[42] S. Zagoruyko and N. Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.

[43] Y. Zhang, K. Lee, and H. Lee. Augmenting supervised neural networks with unsupervised objectives for large-scale image classification. In ICML, 2016.
