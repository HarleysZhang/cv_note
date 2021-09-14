- [摘要](#摘要)
- [1，介绍](#1介绍)
  - [1.1，Faster RCNN 回顾](#11faster-rcnn-回顾)
  - [1.2，mismatch 问题](#12mismatch-问题)
- [2，网络结构](#2网络结构)
- [参考资料](#参考资料)

## 摘要

虽然低 `IoU` 阈值，如 `0.5`，会产生噪声检测（`noisy detections`），但是，随着 `IoU` 阈值的增加，检测性能往往会下降。造成这种情况的主要因素有两个：1）由于正样本呈指数级消失，在训练期间过度拟合;2)**dismatch**：检测器最佳的 `IoU` 与输入假设的 `IoU` 之间的推理时间不匹配。由此，我们提出了多阶段的目标检测器结构：`Cascade R-CNN` 来解决 `IoU` 选择的问题。它**由一系列用不断增加 `IoU` 阈值训练的检测器组成，依次对接近的误报更具选择性**。检测器是逐步训练的，一个检测器输出一个良好的数据分布并作为输入，用于训练下一个更高质量的检测器。逐步改进假设的重采样保证了所有检测器都有一组相同大小的正样本，从而减少了过拟合问题。在 `inference` 阶段使用同样的网络结构合理的提高了 `IOU` 的阈值而不会出现 `mismatch` 问题。

## 1，介绍

> `Cascade RCNN` 是作者 `Zhaowei Cai` 于 `2018` 年发表的论文 `Cascade R-CNN: Delving into High Quality Object Detection`.

![提升IOU阈值对检测器性能的影响](../../data/images/cascade%20rcnn/提升IOU阈值对检测器性能的影响.png)

### 1.1，Faster RCNN 回顾

先回顾下 `Faster RCNN` 的结构，下图是 `Faster RCNN` 的结构图。

![Faster-rcnn网络结构图](../../data/images/faster-rcnn/Faster-rcnn网络结构图.png)

`training` 阶段和 `inference` 阶段的不同在于，`inference` 阶段不能对 `proposala` 进行采样（因为不知道 `gt`，自然无法计算 `IoU`），所以 `RPN` 网络输出的 `300` `RoIs`(`Proposals`)会直接输入到 `RoI pooling` 中，之后通过两个全连接层分别进行类别分类和 `bbox` 回归。

值得注意的是，`Faster RCNN` 网络在 `RPN` 和 `Fast RCNN` 阶段都需要计算 `IoU`，用于判定 `positive` 和 `negative`。前者是生成 `256` 个 `Proposal` 用于 `RPN` 网络训练，后者是生成 `128` 个 `RoIs`(可以理解为 `RPN` 网络优化后的 `Proposals`)用于 `Fast RCNN` 训练。

### 1.2，mismatch 问题

`training` 阶段和 `inference` 阶段，`bbox` 回归器的输入 `proposals` 分布是不一样的，`training` 阶段的输入`proposals` 质量更高(被采样过，IoU > threshold)，`inference` 阶段的输入 `proposals` 质量相对较差（没有被采样过，可能包括很多 IoU < threshold 的），这就是论文中提到 `mismatch` 问题，这个问题是固有存在的，但通常 `threshold` 取 `0.5` 时，`mismatch` 问题还不会很严重。

## 2，网络结构

网络结构如下图(d)

![cascade_rcnn和其他框架的网络结构简略图](../../data/images/cascade%20rcnn/cascade_rcnn和其他框架的网络结构简略图.png)

上图中 (d) 和 (c) 很像，`iterative bbox at inference` 是在推断时候对回归框进行后处理，即模型输出预测结果后再多次处理，而 `Cascade R-CNN` 在训练的时候就进行重新采样，不同的 `stage` 的输入数据分布已经是不同的了。

作者在 COCO 数据集上做了对比实验，达到了 `state-of-the-art` 精度。其中 `backbone` 为`RsNet-101` 的 `Cascade RCNN` 的 `AP` 达到了 `42.8`。

![对比实验结果](../../data/images/cascade%20rcnn/对比实验结果.png)

## 参考资料

- [Cascade R-CNN 详细解读](https://zhuanlan.zhihu.com/p/42553957)