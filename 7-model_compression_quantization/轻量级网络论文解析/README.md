## 轻量级网络模型总结

最近复习了下以前看的 `mobilenet` 系列、`MobileDets`、`shufflenet` 系列、`cspnet`、`vovnet`、`repvgg` 等模型，做了以下总结：
1. 低算力设备-手机移动端 cpu 硬件，考虑 mobilenetv1(深度可分离卷机架构-低 FLOPs)、shuffletnetv2（低 FLOPs 和 MAC）
2. 专用 asic 硬件设备-npu 芯片（地平线x3/x4等、海思3519、安霸cv22等），目标检测问题考虑cspnet网络(减少重复梯度信息)、repvgg（直连架构-部署简单，量化后有掉点风险）
3. 英伟达gpu硬件-t4 芯片，考虑 repvgg 网络（类 vgg 卷积架构-高并行度带来高速度、单路架构省显存/内存）
4. 在大多数的硬体上，`channel` 数为 `8` 的倍数比较有利高效计算。

以上，均是看了轻量级网络论文总结出来的一些**高效模型设计思路**，实际结果还需要自己手动运行测试。