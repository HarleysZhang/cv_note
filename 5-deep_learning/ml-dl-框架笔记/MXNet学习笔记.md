- [制作MXNet数据集](#制作mxnet数据集)
- [数据操作-NDArray](#数据操作-ndarray)
- [创建NDArray](#创建ndarray)
- [NDArray实例运算](#ndarray实例运算)
- [NDArray其他操作](#ndarray其他操作)
- [autograd模块](#autograd模块)
- [简单使用](#简单使用)
- [训练模式和预测模式](#训练模式和预测模式)
- [MXNet用于构建CNN的模块-Symbol、gluon](#mxnet用于构建cnn的模块-symbolgluon)
- [Symbol模块学习](#symbol模块学习)
- [基本神经网络构建](#基本神经网络构建)
- [深度网络模块化构建](#深度网络模块化构建)
- [mxnet.sym.Group](#mxnetsymgroup)
- [mxnet.io.DataIter](#mxnetiodataiter)
- [mxnet.io.MXDataIter](#mxnetiomxdataiter)
- [mxnet.image.ImageIter](#mxnetimageimageiter)
- [MXNet读取图像数据总结](#mxnet读取图像数据总结)

### 制作MXNet数据集

与tf不同，MXNet也有自己的专属图像数据格式，MXNet读取图像有两种方式：

+ 读.rec格式文件，包含文件路径、标签和图像信息
+ 读.lst和图像结合方式，.lst文件其实就是图像路径和标签的对应列表，有点类似.csv文件

### 数据操作-NDArray

NDArray 功能类似 numpy 库的多维数组操作，NDArray 提供了 GPU 计算和自动求梯度等更多应用于深度学习的功能，用法如下：

```python
from mcnet import nd # 导入NDArray(ndarray, nd)模块
x = arange(10) # arange函数创建一个行向量
```
### 创建NDArray

这里的操作和numpy类似，创建零元素张量、1元素张量、改变张量形状等

```python
x = nd.arrange(12)
x.size # 12
x.reshape((3,4))
x.zeros((3, 4, 5)) # 创建元素为0，形状为(3, 4, 5)的张量
x.ones((3, 4, 5))
```
### NDArray实例运算

和numpy类似，矩阵乘法`np.dot(X, Y.T)`，***矩阵连结操作（concatenate）***。
下面分别在行上（***维度0，即形状中的最左边元素***）和列上（维度1，即形状中左起第二个元素）连结两个矩阵。可以看到，输出的第一个NDArray在维度0的长度（ 6 ）为两个输入矩阵在维度0的长度之和（ 3+3 ），而输出的第二个NDArray在维度1的长度（ 8 ）为两个输入矩阵在维度1的长度之和（ 4+4 ）

```python
X = nd.arrange(12).reshape(3,4)
Y = nd.arange(12).reshape(3,4)
Z1 = nd.concat(X, Y, dim=0)   # Z1.shape (6,4)
Z2 = nd.concat(X, Y, dim=1)   # Z1.shape (3,8)
```

### NDArray其他操作

常用的还有广播机制、索引、运算的内存开销和NDArray和Numpy相互互换。NDArray实例和NumPy实例互换如下：

```python
import numpy as np
from mxnet import nd
# 将NumPy实例变换成NDArray实例
P = np.ones((3,4))
D = nd.array(P)
# 将NDArray实例变换成NumPy实例
D.asnumpy
```

### autograd模块

autograd模块实现对函数求梯度(gradient)。

`from mxnet import autograd, nd`

### 简单使用

主要是两个函数：
+ attach_grad()：申请存储梯度所需内存
+ record()：求MXNet记录与求梯度有关的计算
+ backward()：自动求梯度

### 训练模式和预测模式

调用record函数后，MXNet会记录并计算梯度。此外，默认情况下autograd还会将运行模式从预测模式转为训练模式。这可以通过调用is_training函数来查看。

### MXNet用于构建CNN的模块-Symbol、gluon 

涉及使用MXNet框架完成模型构造、参数的访问和初始化、自定义层构建、模型读取、加载和使用GPU等。

### Symbol模块学习
> MXNet 提供了符号接口，用于符号编程的交互。它和一步步的解释的命令式编程不同，我们首先要定义一个计算图。这个图包括了输入的占位符和设计的输出。之后编译这个图，产生一个可以绑定到NDArray s并运行的函数。Symbol API类似于caffe中的网络配置或者Theano中的符号编程。

symbol 构建神经网络有点类似于 tensorflow 的静态图 api。

### 基本神经网络构建

除了基本操作符，symbol 模块提供了丰富的神经网络api，以下示例构建了一个两层的全连接层，通过给定输入数据大小实例化该结构：
```python
import mxnet as mx
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, num_hidden=128, name='fc1')
net = mx.sym.relu(net, name='relu1', act_type='relu')
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=2)
net = mx.sym.SoftmaxOutput(data=net, name='out')
mx.viz.plot_network(net, shape={'data': (100, 200)})
print(net.list_arguments())  # 遍历参数
```

输出如下：
> ['data', 'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias', 'out_label']
注意：FullyConnected 层有三个输入：数据，权值，偏置。任何，任何没有指定输入的将自动生成一个变量。***这里一般权值w，偏置b参数不用指定***。

### 深度网络模块化构建

对于一些深度的CNN网络，可用模块化的方式将其构建，下面一个示例将卷积层、批标准化层和relu激活层捆绑在一起形成一个新的函数。

```python
def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act
```

注意这里的mx.sym.Convolution函数需要自己指定`data`输入张量、`num_filter`滤波器（卷积核）数量、`kernel`卷积核尺寸、`stride`移动步长、`pad`填充shape。
对于2-D convolution, 各参数对应shape是：
- **data**: *(batch_size, channel, height, width)*
- **weight**: *(num_filter, channel, kernel[0], kernel[1])*
- **bias**: *(num_filter,)*
- **out**: *(batch_size, num_filter, out_height, out_width)*.
### mxnet.sym.Group

组合两个输出层：

```python
net = mx.sym.Variable('data')
fc1 = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128)
net = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")
out1 = mx.sym.SoftmaxOutput(data=net, name='softmax')
out2 = mx.sym.LinearRegressionOutput(data=net, name='regression')
group = mx.sym.Group([out1, out2])
print(group.list_outputs())
```

输出如下：
> ['softmax_output', 'regression_output']

MXNet数据读取类：基础类mxnet.io.DataIter、高级类mxnet.io.MXDataIter、高级类mxnet.image.ImageIter

### mxnet.io.DataIter

**mxnet.io.DataIter是MXNet 框架中构造数据迭代器的基础类**，在MXNet框架下只要和数据读取相关的接口基本上都继承该类，比如我们常用的图像算法相关的 mxnet.io.ImageRecordIter 类或 mxnet.image.ImageIter 类都直接或间接继承 mxnet.io.DataIter 类进行封装。
### mxnet.io.MXDataIter

初始化一个mxnet.io.ImageRecordIter类时会得到一个MXDataIter实例，然后当你调用该实例的时候就会调用MXDataIter类的底层C++数据迭代器读取数据（后面会介绍是通过next方法实现的）。
### mxnet.image.ImageIter
Imageter类是纯python实现，继承自DataIter类，与mxnet.io.imageRecordIter类不同，该接口是Python代码实现的图像数据迭代器，既可以读取.rec文件，也可以以图像+.lst方式来读取数据。
> 由于mxnet.image.ImageIter接口在以原图像+.lst文件形式读取数据时是基于python代码实现的，因此在速度上会比基于C++代码实现的mxnet.io.ImageRecordIter接口效率低，尤其是当数据是存储在机械硬盘上时。

### MXNet读取图像数据总结

MXNet的图像数据导入模块主要有mxnet.io.ImageRecordIter和mxnet.image.ImageIter两个类，前者主要用来读取.rec格式的数据，**后者既可以读.rec格式文件，也可以读原图像数据**。

注意：***在MXNet框架中，数据存储为NDArray格式，图像数据也是如此，因此mxnet.image中的很多函数的输入输出都是NDArray格式***
当我们使用mxnet.io.ImageRecordIter这个类读取图像时，必须先用im2rec.py生成lst和rec文件，然后采用mxnet.io.ImageRecordIter类读取rec文件。
