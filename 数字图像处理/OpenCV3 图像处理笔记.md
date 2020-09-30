> 此笔记针对 Python 版本的 opencv3，c++ 版本的函数和 python 版本的函数参数几乎一样，只是矩阵格式从 ndarray 类型变成适合 c++ 的 mat 模板类型。注意，因为 python 版本的opncv只提供接口没有实现，故函数原型还是来自 c++版本的opencv，但是参数解释中的数据类型还是和 python 保持一致。
### 图像的载入：imread() 函数
**函数原型：**
```cpp
Mat imread(const sting& filename, int flags=None)
```
**参数解释：**
+ `filename`：图像的文件路径，`sting` 字符串类型
+ `flags`：载入标识，以何种方式读取图片，`int` 类型的 `flags`。常用取值解释如下：
    + `flags = 0`：始终将图像转成灰度图再返回
    + `flags = 1`：始终将图像转换成彩色图再返回，如果读取的是灰度图，则其返回的矩阵 `shape` 将变为 `(height, width, 3)`
    + `flags = 2`：如果载入的图像深度为 `16` 位或者 `32` 位，就返回对应深度的图像，否则，就转换为 `8` 位图像再返回。

**总结：**读取文件中的图片到 `OpenCV` 中，返回 `Mat` 或者 `ndarray` 类型的矩阵，以彩色模式载入图像时，解码后的图像会默认以 `BGR` 的通道顺序进行存储。

**cv2.imread()函数：**
`python-opencv` 库的 `imread` 函数的 `flags` 参数取值方式与 `C++` 版有所区别。使用函数 `cv2.imread()` 读入图像，图像要么在此程序的工作路径，要么函数参数指定了完整路径，第二个参数是要告诉函数应该如何读取这幅图片，取值如下：
+ `cv2.IMREAD_COLOR` : 取值 `1`，读入一副彩色图像。图像的透明度会被忽略，这是默认参数。
+ `cv2.IMREAD_GRAYSCALE` : 取值 `0`，以灰度模式读入图像。
+ `cv2.IMREAD_UNCHANGED` : 取值 `-1`，读入一幅图像，并且包括图像的 alpha 通道。

> Instead of these three flags, you can simply pass integers 1, 0 or -1 respectively.

```python
import numpy as np
import cv2
# Load an color image in grayscale
img = cv2.imread('messi5.jpg',0)
```
`opencv-python` 库的读取图像函数 `cv2.imread()` 官方定义如下图。

![opencv-python库的读取图像函数官方定义](../images/cv2.imread函数.png)

### 图像的显示：imshow()函数
**函数原型：**
```cpp
void imshow(const string &winname, InputArray mat)
```
**参数解释：**
+ `winname`：需要显示的窗口标识名称，`string` 字符串类型
+ `mat`：需要显示的图像矩阵，`ndarray` numpy 矩阵类型

**总结：**`imshow` 函数用于在指定的窗口显示图像，窗口会自动调整为图像大小。
### minMaxLoc 函数
函数 `cv :: minMaxLoc` 查找最小和最大元素值及其位置，返回的位置坐标是**先列号，后行号**（列号，行号） 。在整个数组中搜索极值，或者如果mask不是空数组，则在指定的数组区域中搜索极值。（**只适合单通道矩阵**）。函数原型：
```CPP
CV_EXPORTS_W void minMaxLoc(InputArray src, CV_OUT double* minVal,
                            CV_OUT double* maxVal = 0, CV_OUT Point* minLoc = 0,
                            CV_OUT Point* maxLoc = 0, InputArray mask = noArray());
```
函数参数解释：
+ `src`：input single-channel array.
+ `minVal`：pointer to the returned minimum value; NULL is used if not required.
+ `maxVal`：pointer to the returned maximum value; NULL is used if not required.
+ `minLoc`：pointer to the returned minimum location (in 2D case); NULL is used if not required.
+ `maxLoc`：pointer to the returned maximum location (in 2D case); NULL is used if not required.
### 位深度的概念
+ 灰度图的位深度是 `16`，则其矩阵的元素类型为 `uint16` ，彩色图其位深度一般是 `24` ，红色占 `8` 个位、蓝色占 `8` 个位、绿色占 `8` 个位，其矩阵的元素类型为 `uint8`。
+ 位分辨率（ `Bit Resolution` ）又称色彩深度、色深或位深度，在位图图像或视频视频缓冲区，指一个像素中，每个颜色分量（`Red、Green、Blue、Alpha` 通道）的比特数。
+ `matplotlib.image.imsave` 将灰度图的矩阵保存为图像格式时，其默认保存的图像通道数为 `4`：`RGBA`，其中 `RGB` 三个通道对应的二维矩阵数值完全一样。
