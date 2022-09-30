## 一，概述

`TensorRT` 是 NVIDIA 官方推出的基于 `CUDA` 和 `cudnn` 的高性能深度学习推理加速引擎，能够使深度学习模型在 `GPU` 上进行低延迟、高吞吐量的部署。采用 `C++` 开发，并提供了 `C++` 和 `Python` 的 API 接口，支持 TensorFlow、Pytorch、Caffe、Mxnet 等深度学习框架，其中 `Mxnet`、`Pytorch` 的支持需要先转换为中间模型 `ONNX` 格式。截止到 2021.4.21 日， `TensorRT` 最新版本为 `v7.2.3.4`。

延迟和吞吐量的一般解释：

+ 延迟 (`Latency`): 指执行一个操作所花的时间。
+ 吞吐量 (`Throughput`): 在单位时间内，可执行的运算次数。
## 二，TensorRT 工作流程

在描述 `TensorRT` 的优化原理之前，需要先了解 `TensorRT` 的工作流程。首先输入一个训练好的 `FP32` 模型文件，并通过 `parser` 等方式输入到 `TensorRT` 中做解析，解析完成后 `engin` 会进行计算图优化（优化原理在下一章）。得到优化好的 `engine` 可以序列化到内存（`buffer`）或文件（`file`），读的时候需要反序列化，将其变成 `engine`以供使用。然后在执行的时候创建 `context`，主要是分配预先的资源，`engine` 加 `context` 就可以做推理（`Inference`）。

![TensorRT工作流程.jpg](../../data/images/TensorRT/TensorRT工作流程.jpg)

## 三，TensorRT 的优化原理

`TensorRT` 的优化主要有以下几点：

1. **算子融合（网络层合并）**：我们知道 `GPU` 上跑的函数叫 `Kernel`，`TensorRT` 是存在 `Kernel` 调用的，频繁的 `Kernel` 调用会带来性能开销，主要体现在：数据流图的调度开销，GPU内核函数的启动开销，以及内核函数之间的数据传输开销。大多数网络中存在连续的卷积 `conv` 层、偏置 `bias` 层和 激活 `relu` 层，这三层需要调用三次 cuDNN 对应的 API，但实际上这三个算子是可以进行融合（合并）的，合并成一个 `CBR` 结构。同时目前的网络一方面越来越深，另一方面越来越宽，可能并行做若干个相同大小的卷积，这些卷积计算其实也是可以合并到一起来做的（横向融合）。比如 `GoogLeNet` 网络，把结构相同，但是权值不同的层合并成一个更宽的层。
2. `concat` 层的消除。对于 `channel` 维度的 `concat` 层，`TensorRT` 通过非拷贝方式将层输出定向到正确的内存地址来消除 `concat` 层，从而减少内存访存次数。
3. `Kernel` 可以根据不同 `batch size` 大小和问题的复杂度，去自动选择最合适的算法，`TensorRT` 预先写了很多 `GPU` 实现，有一个自动选择的过程（没找到资料理解）。其问题包括：怎么调用 `CUDA` 核心、怎么分配、每个 `block` 里面分配多少个线程、每个 `grid` 里面有多少个 `block`。

4. `FP32->FP16、INT8、INT4`：低精度量化，模型体积更小、内存占用和延迟更低等。
5. 不同的硬件如 `P4` 卡还是 `V100` 卡甚至是嵌入式设备的卡，`TensorRT` 都会做对应的优化，得到优化后的 `engine`。

## 四，参考资料

1. [内核融合：GPU深度学习的“加速神器”](https://www.msra.cn/zh-cn/news/features/kernel-fusion-20170925)
2. [高性能深度学习支持引擎实战——TensorRT](https://zhuanlan.zhihu.com/p/35657027)
3. 《NVIDIA TensorRT 以及实战记录》PPT
