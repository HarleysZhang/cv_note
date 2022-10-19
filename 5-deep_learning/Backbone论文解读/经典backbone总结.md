## 目录
  - [VGG](#vgg)
  - [ResNet](#resnet)
  - [Inceptionv3](#inceptionv3)
  - [Resnetv2](#resnetv2)
  - [ResNeXt](#resnext)
  - [Darknet53](#darknet53)
  - [DenseNet](#densenet)
  - [CSPNet](#cspnet)
  - [VoVNet](#vovnet)
  - [一些结论](#一些结论)
  - [参考资料](#参考资料)

## VGG

`VGG`网络结构参数表如下图所示。

![VGG](../../data/images/backbone/VGG.png)

## ResNet

`ResNet` 模型比 `VGG` 网络具有更少的滤波器数量和更低的复杂性。 比如 `Resnet34` 的 `FLOPs` 为 `3.6G`，仅为 `VGG-19` `19.6G` 的 `18%`。
> 注意，论文中算的 `FLOPs`，把乘加当作 `1` 次计算。

`ResNet` 和 `VGG` 的网络结构连接对比图，如下图所示。

![resnet](../../data/images/backbone/resnet.png)

不同层数的 `Resnet` 网络参数表如下图所示。

![resnet网络参数表](../../data/images/backbone/resnet网络参数表.png)

> 看了后续的 `ResNeXt`、`ResNetv2`、`Densenet`、`CSPNet`、`VOVNet` 等论文，越发觉得 `ResNet` 真的算是 `Backone` 领域划时代的工作了，因为它让**深层**神经网络可以训练，基本解决了深层神经网络训练过程中的梯度消失问题，并给出了系统性的解决方案（两种残差结构），即系统性的让网络变得更“深”了。而让网络变得更“宽”的工作，至今也没有一个公认的最佳方案（`Inception`、`ResNeXt` 等后续没有广泛应用），难道是因为网络变得“宽”不如“深”更重要，亦或是我们还没有找到一个更有效的方案。

## Inceptionv3

常见的一种 `Inception Modules` 结构如下：

![Inception模块](../../data/images/backbone/Inception模块.jpg)

## Resnetv2

作者总结出**恒等映射形式的快捷连接和预激活对于信号在网络中的顺畅传播至关重要**的结论。

## ResNeXt

ResNeXt 的卷积 block 和 Resnet 对比图如下所示。

![resnext的卷积block和resnet的对比图](../../data/images/backbone/resnext的卷积block和resnet的对比图.png)

ResNeXt和Resnet的模型结构参数对比图如下图所示。

![resnext的结构参数和resnet的对比图](../../data/images/backbone/resnext的结构参数和resnet的对比图.png)

## Darknet53

`Darknet53` 模型结构连接图，如下图所示。

![darknet53](../../data/images/backbone/darknet53.png)

## DenseNet

> 作者 `Gao Huang` 于 `2018` 年发表的论文 `Densely Connected Convolutional Networks`。

**在密集块（`DenseBlock`）结构中，每一层都会将前面所有层 `concate` 后作为输入**。`DenseBlock`（类似于残差块的密集块结构）结构的 `3` 画法图如下所示：

![3种DenseNet结构画法](../../data/images/backbone/3种DenseNet结构画法.png)

可以看出 `DenseNet` 论文更侧重的是 `DenseBlock` 内各个卷积层之间的密集连接（`dense connection`）关系，另外两个则是强调每层的输入是前面所有层 feature map 的叠加，反映了 feature map 数量的变化。

## CSPNet

**`CSPDenseNet` 的一个阶段是由局部密集块和局部过渡层组成（`a partial dense block and a partial transition layer`）**。

![Figure3几种不同形式的CSP](../../data/images/backbone/Figure3几种不同形式的CSP.png)

`CSP` 方法可以减少模型计算量和提高运行速度的同时，还不降低模型的精度，是一种更高效的网络设计方法，同时还能和 `Resnet`、`Densenet`、`Darknet` 等 `backbone` 结合在一起。

## VoVNet

**One-Shot Aggregation（只聚集一次）是指 OSA 模块的 concat 操作只进行一次，即只有最后一层($1\times 1$ 卷积)的输入是前面所有层 feature map 的 concat（叠加）**。`OSA` 模块的结构图如图 1(b) 所示。

![VoVNet](../../data/images/backbone/VoVNet.png)

在 `OSA module` 中，每一层产生两种连接，一种是通过 `conv` 和下一层连接，产生 `receptive field` 更大的 `feature map`，另一种是和最后的输出层相连，以聚合足够好的特征。通过使用 `OSA module`，`5` 层 `43` `channels` 的 `DenseNet-40` 的 `MAC` 可以被减少 `30%`（`3.7M -> 2.5M`）。

基于 OSA 模块构建的各种 `VoVNet` 结构参数表如下。

![各种VoVNet结构](../../data/images/backbone/各种VoVNet结构.png)

作者认为 `DenseNet` 用更少的参数与 `Flops` 而性能却比 `ResNet` 更好，主要是因为`concat` 比 `add` 能保留更多的信息。但是，实际上 `DenseNet` 却比 `ResNet`要慢且消耗更多资源。

`GPU` 的计算效率：

- `GPU` 特性是擅长 `parallel computation`，`tensor` 越大，`GPU` 使用效率越高。
- 把大的卷积操作拆分成碎片的小操作将不利于 `GPU` 计算。
- 设计 `layer` 数量少的网络是更好的选择。
- 1x1 卷积可以减少计算量，但不利于 GPU 计算。

在 CenterMask 论文提出了 VoVNetv2，其卷积模块结构图如下：

![VoVNetv2](../../data/images/backbone/VoVNetv2.png)

## 一些结论

- 当卷积层的输入输出通道数相等时，内存访问代价（`MAC`）最小。
- 影响 CNN 功耗的主要因素在于内存访问代价 MAC，而不是计算量 FLOPs。
- GPU 擅长并行计算，Tensor 越大，GPU 使用效率越高，把大的卷积操作拆分成碎片的小操作不利于 GPU 计算。
- 1x1 卷积可以减少计算量，但不利于 GPU 计算。

## 参考资料

+ `VGG/ResNet/Inception/ResNeXt/CSPNet` 论文
+ [深度学习论文: An Energy and GPU-Computation Efficient Backbone Network for Object Detection及其PyTorch](https://blog.csdn.net/shanglianlm/article/details/106482678)