- [numpy 堆叠数组](#numpy-堆叠数组)
- [ravel() 函数](#ravel-函数)
- [stack() 函数](#stack-函数)
- [vstack()函数](#vstack函数)
- [hstack()函数](#hstack函数)
- [concatenate() 函数](#concatenate-函数)
- [参考资料](#参考资料)

## numpy 堆叠数组

在做图像和 nlp 的数组数据处理的时候，经常需要实现两个数组堆叠或者连接的功能，这就需用到 `numpy` 库的一些函数，numpy 库中的常用**堆叠数组**函数如下：

* `stack` : Join a sequence of arrays along a new axis.
* `hstack`: Stack arrays in sequence horizontally (column wise).
* `vstack` : Stack arrays in sequence vertically (row wise).
* `dstack` : Stack arrays in sequence depth wise (along third axis).
* `concatenate` : Join a sequence of arrays along an existing axis.

## ravel() 函数

ravel() 方法可让将多维数组展平成一维数组。如果不指定任何参数，ravel() 将沿着行（第 0 维/轴）展平/拉平输入数组。

示例代码如下:

```python
std_array = np.random.normal(3, 2.5, size=(2, 4))
array1d = std_array.ravel()
print(std_array)
print(array1d)
```

程序输出结果如下：

```shell
[[5.68301857 2.09696067 2.20833423 2.83964393]
 [2.38957339 9.66254303 1.58419716 2.82531094]]
 
[5.68301857 2.09696067 2.20833423 2.83964393 2.38957339 9.66254303 1.58419716 2.82531094]
```
## stack() 函数

stack() 函数原型是 stack(*arrays*, _axis_=0, _out_=*None*)，功能是沿着**给定轴**连接数组序列，轴默认为第0维，即默认沿着第 0 维 stacks 数组。

1，**参数解析：**
- arrays: 类似数组（数组、列表）的序列，这里的**每个数组必须有相同的shape。**
- axis: 默认为整形数据，axis决定了沿着哪个维度stack输入数组。

2，**返回：**
- stacked : `ndarray` 类型。The stacked array has one more dimension than the input arrays.

实例如下：

```python
import numpy as np
# 一维数组进行stack
a1 = np.array([1, 3, 4])    # shape (3,)
b1 = np.array([4, 6, 7])    # shape (3,)
c1 = np.stack((a,b))
print(c1)
print(c1.shape)    # (2,3)
# 二维数组进行堆叠
a2 = np.array([[1, 3, 5], [5, 6, 9]])    # shape (2,3)
b2 = np.array([[1, 3, 5], [5, 6, 9]])    # shape (2,3)
c2 = np.stack((a2, b2), axis=0)
print(c2)
print(c2.shape)
```
输出为：

> [[1 3 4]
[4 6 7]]

> (2, 3)

> [[[1 3 5]
[5 6 9]]
[[1 3 5]
[5 6 9]]]
(2, 2, 3)

可以看到，进行 stack 的两个数组必须**有相同的形状**，同时，输出的结果的维度是比输入的数组都要多一维的。我们拿第一个例子来举例，两个含 3 个数的一维数组在第 0 维进行堆叠，其过程等价于先给两个数组增加一个第0维，变为1*3的数组，再在第 0 维进行 `concatenate()` 操作：

```python
a = np.array([1, 3, 4])
b = np.array([4, 6, 7])
a = a[np.newaxis,:]
b = b[np.newaxis,:]
np.concatenate([a,b],axis=0)
```
输出为：

> array([[1, 2, 3],
      [2, 3, 4]])

## vstack()函数
vstack函数原型是vstack(tup)，功能是垂直的（按照行顺序）堆叠序列中的数组。tup是数组序列(元组、列表、数组)，数组必须在所有轴上具有相同的shape，除了第一个轴。1-D arrays must have the same length.

```python
# 一维数组
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
np.vstack((a,b))
```
> array([[1, 2, 3],
[2, 3, 4]])

```Plain Text
# 二维数组
a = np.array([[1], [2], [3]])
b = np.array([[2], [3], [4]])
np.vstack((a,b))
```
> array([[1],
[2],
[3],
[2],
[3],
[4]])

## hstack()函数
hstack()的函数原型：hstack(tup) ，参数tup可以是元组，列表，或者numpy数组，返回结果为numpy的数组。它其实就是**水平(按列顺序)**把数组给堆叠起来，与vstack()函数正好相反。举几个简单的例子：

```python
# 一维数组
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
np.hstack((a,b))
```
> array([1, 2, 3, 2, 3, 4])

```python
# 二维数组
a = np.array([[1], [2], [3]])
b = np.array([[2], [3], [4]])
np.hstack((a,b))
```
> array([[1, 2],
[2, 3],
[3, 4]])

**vstack()和hstack函数对比：**

这里的**v**是vertically的缩写，代表垂直（沿着行）堆叠数组，这里的**h**是horizontally的缩写，代表水平（沿着列）堆叠数组。
tup是数组序列(元组、列表、数组)，数组必须在所有轴上具有相同的shape，除了第一个轴。  

## concatenate() 函数

concatenate()函数功能齐全，理论上可以实现上面三个函数的功能，concatenate()函数**根据指定的维度，对一个元组、列表中的list或者ndarray进行连接**，函数原型：

```python
numpy.concatenate((a1, a2, ...), axis=0)
```

```python
a = np.array([[1, 2], [3，4]])　　　　　　　　　　　　　　　
b = np.array([[5, 6], [7, 8]])
# a、b的shape为（2,2），连接第一维就变成（4,2），连接第二维就变成（2,4）
np.concatenate((a, b), axis=0)
```
> array([[1, 2],
[3, 4],
[5, 6],
[7, 8]])

**注意：axis指定的维度（即拼接的维度）可以是不同的，但是axis之外的维度（其他维度）的长度必须是相同的**。注意 concatenate 函数使用最广，必须在项目中熟练掌握。

## 参考资料
- [numpy中的hstack()、vstack()、stack()、concatenate()函数详解](https://mp.weixin.qq.com/s?__biz=MzI1MzY0MzE4Mg==&mid=2247484138&idx=2&sn=f1dca4b3790284371fe103b2108d92a6&chksm=e9d0122bdea79b3d832612ae41a68e120a74764d0b557ffc286da1c4163f397c4e780a77a5b9&mpshare=1&scene=1&srcid=0501xRYNtB00dNBcW8UlLzml#rd)



