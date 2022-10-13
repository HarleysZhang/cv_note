## 一，背景知识

**实践作业1**：如何通过工具得到自己当前使用 `PC` 的 `Cache` 信息（15分）。包括 `L1/L2/L3 Cache`（数据和指令）的大小，`Cache Line` 的大小。

### 1.1，Linux 查看 CPU 和 Cache 信息
1，`Linux` 查看 `cpu` 信息命令：`cat /proc/cpuinfo`。

```bash
(base) harley@harley-pc:/sys/devices/system/cpu/cpu0/cache/index3$ cat /proc/cpuinfo
processor    : 0
vendor_id    : GenuineIntel
cpu family    : 6
model        : 158
model name    : Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
stepping    : 10
microcode    : 0xb4
cpu MHz        : 3192.005
cache size    : 12288 KB
physical id    : 0
siblings    : 1
core id        : 0
cpu cores    : 1
apicid        : 0
initial apicid    : 0
fpu        : yes
fpu_exception    : yes
cpuid level    : 22
wp        : yes
flags        : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon nopl xtopology tsc_reliable nonstop_tsc cpuid pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault invpcid_single pti ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 avx2 smep bmi2 invpcid rdseed adx smap clflushopt xsaveopt xsavec xsaves arat md_clear flush_l1d arch_capabilities
bugs        : cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf mds swapgs itlb_multihit srbds
bogomips    : 6384.01
clflush size    : 64
cache_alignment    : 64
address sizes    : 43 bits physical, 48 bits virtual
power management:
...
processor    : 3
...
```
2， `Linux` 查询 `L1/L2/L3 cache`大小：`cat /sys/devices/system/cpu/cpu0/cache/index*/size`(`*`为 `0/1/2/3`)

```bash
(base) harley@harley-pc:~$ cat /sys/devices/system/cpu/cpu0/cache/index0/size
32K
(base) harley@harley-pc:~$ cat /sys/devices/system/cpu/cpu0/cache/index0/type
Data
(base) harley@harley-pc:~$ cat /sys/devices/system/cpu/cpu0/cache/index1/type
Instruction
(base) harley@harley-pc:~$ cat /sys/devices/system/cpu/cpu0/cache/index1/size
32K
(base) harley@harley-pc:~$ cat /sys/devices/system/cpu/cpu0/cache/index2/size
256K
(base) harley@harley-pc:~$ cat /sys/devices/system/cpu/cpu0/cache/index3/size
12288K
```
现代 `CPU`的 `L1 cache` 是逻辑核私有的，L1 cache 分指令 L1 cache 和数据 L1 cache，大小相等都为 `32 KB`；目前，L2 cache 也是片内私有，所以每个核只有`256 KB`；而对于 L3 cache，一个物理核CPU 的所有逻辑核共享，所以在每个逻辑核来看，L3 cache 都为`12288 KB`。本机的虚拟机总共有 `1` 个物理核，而本机共有 `6` 核 `12` 线程，所以可推算得到\*\*本机的 `Cache` 信息：

* L1 cache：
   * L1 Data: `192 KB = 32 x 6 KB`
   * L1 Instruction: `192 KB = 32 x 6 KB`
* L2 cache：`1536 KB = 256 X 6 KB`
* L3 cache：`12288 KB`

### 1.2，Windows查看cpu和cache信息
1，任务管理器->性能，即可查看 `cpu` 和 `L1/L2/L3 Cache` 大小，如下图所示。

![image](images/5bf8b57e-8a78-4de8-8499-7fc9960af1f0.png)

2，或者下载安装 `cpuz` 软件，打开即可查看，如下图所示。

![image](images/50ede7f0-9d94-4d16-8c61-0438f39b3900.png)

### 1.3， 尝试分析 `init()` 函数

**实践作业2**：尝试分析 `init()`函数使用 `O1` 和 `O3` 优化的`profile`结果差异。

* O3 相对 O1，执行时间减少，尝试分析反汇编代码（O3.s 和 O1.s），给出解释（15 分）
* O3 相对 O1，D1 Cache 的访问次数为什么从 8409K 下降到 2125K（15 分）
* O3 相对 O1，D1 Cache 的 `miss rate` 为什么从 6.2% 上升到 24.7%（15 分）

**问题分析和答案**:

1. 使用 `-O3` 参数优化，编译器会采取很多向量化算法，提高代码的并行执行程度，利用现代 `CPU` 中的流水线，`Cache` 等。`O3` 优化会提高执行代码的大小，也会降低目标代码的执行时间。
2. 访问次数下降是因为使用了 `O3` 优化，使得程序会自动访问连续的内存。
3. 高速缓存缺失（`cache miss`）是因为访存的内存都是不连续的。

## 二，优化 $A^{T}*A$ 矩阵乘法

**实践作业3**：优化 $A^T*A$ 的矩阵乘法，目标是尽量减少计算时间。

* 其中 `A` 的大小为 `1024x8192`，元素为 `int`类型。
* 需要从算法层面，指令层面和访存优化的角度联合优化。
* 通过文档说明自己的优化思路（20 分）。
* 可以选择自己熟悉的处理器平台进行代码编写，如 `Intel` 平台或者`ARM`平台（20 分）

**问题分析**：矩阵乘的算法优化可分为两类：

- 基于**算法分析**的方法：根据矩阵乘计算特性，从数学角度优化，典型的算法包括 Strassen 算法和 Coppersmith–Winograd 算法。
- 基于**软件优化**的方法：根据计算机存储系统的层次结构特性，选择性地调整计算顺序，主要有循环拆分向量化、内存重排等。
### 2.1，算法层面优化
从算法层面优化，首先需要分析朴素矩阵乘法的算法复杂度，分析可知，朴素的矩阵乘算法的时间复杂度为 $O(n^3)$ 。根据矩阵乘计算特性，从数学角度（算法层面）优化，典型的算法包括 `Strassen` 算法和 `Coppersmith–Winograd` 算法。

#### 2.1.1，`Strassen` 算法
`Strassen` 算法是 `1969` 年提出的复杂度为 $O(n^{log_2{7}})$ 的矩阵乘法，这是历史上第一次将矩阵乘的计算复杂度价格低到 $O(n^3)$ 以下。

基于**分治（Divide and Conquer）的思想**，将矩阵 $A, B, C∈R^{n^2×n^2}$ 分别拆分为更小的矩阵，根据矩阵基本的运算法则，拆分后朴素算法的计算共需要**八次小矩阵乘法和四次小矩阵加法**计算。`Strassen` 算法的核心思想是通过引入辅助计算的中间矩阵，再将中间矩阵进行组合得到最后的矩阵，这个过程使用了**七次乘法和十八次加法**，将矩阵乘的算法复杂度降低到了 $O(n^{log_27})$ （递归地运行该算法）。算法的详细推导过程如下：

1，基于分治（Divide and Conquer）的思想，Starssen 算法将矩阵 $A,\ B,\ C \in R^{n^2 \times n^2}$ 分别拆分为更小的矩阵：

$$
\mathbf{A} =
\begin{bmatrix}
\mathbf{A}_{1,1} & \mathbf{A}_{1,2} \\
\mathbf{A}_{2,1} & \mathbf{A}_{2,2}
\end{bmatrix},
\mathbf{B} =
\begin{bmatrix}
\mathbf{B}_{1,1} & \mathbf{B}_{1,2} \\
\mathbf{B}_{2,1} & \mathbf{B}_{2,2}
\end{bmatrix},
\mathbf{C} =
\begin{bmatrix}
\mathbf{C}_{1,1} & \mathbf{C}_{1,2} \\
\mathbf{C}_{2,1} & \mathbf{C}_{2,2}
\end{bmatrix}
$$

其中，$A_{i,j},\ B_{i,j},\ C_{i,j} \in R^{2^{n-1} \times 2^{n-1}}$。拆分后朴素算法的计算如下所示，共需要八次小矩阵乘法和四次小矩阵加法计算。

![image](images/7ef2efa9-1119-4c4d-9bc8-bb006ef6eaf6.png)

2，引入七个如下所示的用于辅助计算的中间矩阵。

![image](images/fa36902b-ad65-44ea-9fd3-9464fd0fd83c.png)

3，将中间矩阵进行组合得到最后的结果矩阵。

![image](images/ac614b82-a91a-4803-901e-30dc58399cf8.png)

#### 2.1.2，Coppersmith–Winograd 算法
`Strassen` 算法尽管学术意义重大，但实际应用有限，`Coppersmith–Winograd` 算法(`1990`年)的提出将矩阵乘法的算法复杂度降低到了$O(n^2.376)$。其算法的详细推导过程可参考 [Matrix multiplication via arithmetic progressions （原始论文）](https://www.sciencedirect.com/science/article/pii/S0747717108800132)。

### 2.2，指令层面优化
改进访存局部性和利用向量指令等方法都是属于软件优化方法。软件优化方法基于对计算机体系机构和软件系统的特征分析，结合具体计算的特性，设计出针对性的优化方法。

现在的 `CPU` 处理器，基本上想获得高的性能，必须要用**向量化**指令，不管是老的 `SSE2`，`AVX` 或者 `AVX 2.0` 等，对于`CPU` 的优化，如果想达到高性能，必须要用到单指令多数据（`SIMD`）的向量化指令。

### 2.3，访存优化
**程序运行环境**(`g++` 编译器基础上 `Linux`系统比 `Windows` 运行程序时间更少一些)：

* 操作系统：`Ubuntu`
* 编译器：`g++`，`g++ --std=c++17 -O3 matrix_multiplication.cpp`
* 编程语言：`C++`
* `CPU`平台: `Intel` 的 `I7-8700` `CPU`

朴素的矩阵乘算法的时间复杂度为 $O(n^3)$，以 $A^T*A$ 为例，矩阵相乘核心代码如下：

```cpp
vector<vector<int>> matrix_mul(vector<vector<int>> A, vector<vector<int>> B){
    /*二维矩阵相乘函数
    */
    // vector<vector<int>> A_T = matrix_transpose(A);
    assert((*A.begin()).size()==B.size()); //断言，第一个矩阵的列必须等于第二个矩阵的行
    int new_rows = A.size();
    int new_cols = (*B.begin()).size();
    int L = B.size();
    vector<vector<int>> C(new_rows, vector<int>(new_cols,0));

    for(int i=0; i<new_rows; i++){
        for(int j=0; j<new_cols;j++){
           for(int k=0;k<L;k++){
              C[i][j] += A[i][k]*B[k][j]
           }
            // C[i][j] = vector_mul(A[i], get_col(B, j));
        }
    }
    return C;
}
```
从以上代码可以看出，`B[k][j]` 读取内存中的数据，是不连续的。在最底层的循环中，随着 `k` 不断加 `1`，`B[k][j]` 不断的在内存中跳跃。这会引起缓存命中率低，循环程序不断的把内存转移至缓存，引起效率降低。在我的台式机的虚拟机上，当`A` 的大小为 `1024x8192`时，需要用时 `85.3 s`（我用的编译器是`g++ -O3`）。下面的代码是我从访存优化的角度使用的两种优化方法。

#### 2.3.1，优化方法 1 (改进访存局部性)
> 内存使用上，程序访问的内存地址之间连续性越好，程序的访问效率就越高。

充分利用计算机系统的特性可以大幅度提高程序性能，参考卡内基梅隆大学的镇校神课《深入理解计算机系统》里面，给出一种方法，仅仅改变循环的次序，就可以大幅度提高性能，修改后代码如下：

```cpp
vector<vector<int>> matrix_mul_optim(vector<vector<int>> A, vector<vector<int>> B){
    /*二维矩阵相乘函数
    */
    // vector<vector<int>> A_T = matrix_transpose(A);
    assert((*A.begin()).size()==B.size()); //断言，第一个矩阵的列必须等于第二个矩阵的行
    int new_rows = A.size();
    int new_cols = (*B.begin()).size();
    int L = B.size();
    vector<vector<int>> C(new_rows, vector<int>(new_cols,0));
    for(int k=0; k<L; k++){
        for(int i=0; i<new_rows; i++){
            int r = A[i][k];
            for(int j=0; j<new_cols;j++){
                C[i][j] += A[i][k]*B[k][j];
            }
            // C[i][j] = vector_mul(A[i], get_col(B, j));
        }
    }
    return C;
}
```
首先，最内层的循环，随着 `j` 加 `1`，`C[i][j]` 和 `B[k][j]` 都是每次只加 1，这符合空间局部性的原理，也就是说，内存每次读取都是一个接着一个的来，没有大幅度跳跃。其次，`A[i][k]` 在中间层循环是跳跃的，但是中间层执行的没有底层那么多，而且我们把 `A[i][k]` 赋给了局部变量 `r`，在编译器生成汇编代码的过程中，局部变量 `r` 应该由 `CPU` 寄存器存储，最底层循环程序读取寄存器的时间几乎可以忽略不计的。修改后的代码运行耗时 `25.2 s`。

#### 2.3.2，优化方法2(分块矩阵+改进访存局部性)
**将矩阵分块（计算拆分），每次计算一部分内容**。分块的目的就是优化访存，通过分块之后让访存都集中在一定区域，能够提高了数据局部性，从而提高 Cache 利用率，性能就会更好。结合分块矩阵和改进访存局部性两种方法的代买运行耗时 `18.8 s`。修改后的代码如下：

```cpp
vector<vector<int>> matrix_mul_optim3(vector<vector<int>> A, vector<vector<int>> B){
    /*二维矩阵相乘函数，优化方法-分块矩阵
    */
    // vector<vector<int>> A_T = matrix_transpose(A);
    assert((*A.begin()).size()==B.size()); //断言，第一个矩阵的列必须等于第二个矩阵的行
    int M = A.size();
    int N = (*B.begin()).size();
    int K = B.size();  // 第二个矩阵的行

    int NUM = 8;  // 分块数
    int MT = A.size()/NUM;  // 分块矩阵的行
    int NT = (*B.begin()).size()/NUM;  // 分块矩阵的列
    int KT = B.size()/NUM;  // 
    vector<vector<int>> C(M, vector<int>(N,0));
    for(int kt = 0; kt < NUM; ++kt){
        for(int it = 0; it < NUM; ++it){
            for(int jt = 0; jt < NUM; ++jt){
                int ktt = kt * KT;
                int itt = it * MT;
                int jtt = jt * NT;
                for(int k = ktt; k < ktt + KT; ++k){
                    int num_k = k * NUM;
                    for(int i = itt; i < itt + MT; ++i){
                        // int num_i = i * NUM;
                        int r = A[i][k];
                        for(int j = jtt; j < jtt + NT; ++j){
                            C[i][j] += r * B[k][j];
                        }
                    }
                }
            }
        }
    }
    return C;
}
```
程序输出结果如下：

> Done! Timing : 18.889000 s
The size of result matrix is (8192, 8192)

## 三，优化方法集合的完整代码
完整可直接在`windows/Linux` 上可运行的代码如下：

```cpp
/*
 * 矩阵乘法(A^T*A)实现
 */

#include<iostream>
#include<stdlib.h>
#include<vector>
#include<cassert>
#include"iomanip"
#include <chrono>
#include <iomanip>
using namespace std::chrono;
using namespace std;


typedef std::vector<int> Row;
typedef std::vector<Row> Matrix;

using namespace std;
int m = 1024;
int n = 8192;

vector<vector<int>> init_matrix(int mm, int nn){
    /*初始化指定行和列的二维矩阵
    */
    int random_integer;

    vector<vector<int>> matrix(mm, vector<int>(nn,0)); // 初始化二维数组matrix为1024*8192，所有元素为0
    // cout << "The size of init matrix is " << "(" << matrix.size() << ", " << matrix[0].size() << ")" << std::endl;
    for (int i=0;i < matrix.size();i++)
    {
        for(int j=0;j < matrix[i].size();j++)
        {
            random_integer = rand() % 128;
            matrix[i][j] = random_integer;  // 利用下标给二维数组赋值
            // matrix[i].push_back(random_integer);  // 利用push_back给vactor添加元素
            // cout << random_integer << endl;
        }
    }
    return matrix;

}

void print_matrix(vector<vector<int>> matrix){
    /*打印二维向量（矩阵）的元素
    */
    cout << "The size of matrix is" << "(" << matrix.size() << ", " << matrix[0].size() << ")" << std::endl;
    //迭代器遍历
    // vector<vector<int >>::iterator iter;
    for(auto iter=matrix.cbegin();iter != matrix.cend(); ++iter)
    {
        for(int i = 0;i<(*iter).size();i++){
            cout << (*iter)[i] << " ";
        }
        cout << std::endl;
    }

    // cout << "print success" << endl;
}

vector<vector<int>> matrix_transpose(vector<vector<int>> A){
    /*获取矩阵的转置
    */
    int rows = A.size();
    int cols = (*A.begin()).size();
    vector<vector<int>> A_T(cols, vector<int>(rows,0));
    for (int j=0;j < cols; j++){
        for(int i=0;i< rows; i++){
            A_T[j][i] = A[i][j];
        }
    }
    return A_T;
}

int vector_mul(vector<int> A1, vector<int> B1){
    /*向量相乘函数
    */
    assert(A1.size()==B1.size()); //断言，两个向量的长度必须相等
    vector<int>::iterator begin;  // 定义迭代器
    int result;
    // 迭代器循环遍历元素
    for(int i=0; i<A1.size(); i++){
        result += A1[i]*B1[i];
    }
    return result;

}

vector<int> get_col(vector<vector<int>> matrix, int n){
    /*获取矩阵指定列的向量
    */
//    vector<int> col(matrix.size());
    vector<int> col;
    col.reserve(matrix.size());
    for(auto row: matrix){
        col.push_back(row[n]);
    }
    // cout << "The size of vector is " << col.size() << endl;
    return col;
}

vector<vector<int>> matrix_mul(vector<vector<int>> A, vector<vector<int>> B){
    /*二维矩阵相乘函数
    */
    // vector<vector<int>> A_T = matrix_transpose(A);
    assert((*A.begin()).size()==B.size()); //断言，第一个矩阵的列必须等于第二个矩阵的行
    int new_rows = A.size();
    int new_cols = (*B.begin()).size();
    int L = B.size();
    vector<vector<int>> C(new_rows, vector<int>(new_cols,0));

    for(int i=0; i<new_rows; i++){
        for(int j=0; j<new_cols;j++){
            for(int k=0; k<L; k++){
                C[i][j] += A[i][k]*B[k][j];
            }
            // C[i][j] = vector_mul(A[i], get_col(B, j));
        }
    }
    return C;
}

vector<vector<int>> matrix_mul_optim1(vector<vector<int>> A, vector<vector<int>> B){
    /*二维矩阵相乘优化函数1-改进访存局部性
    */
    // vector<vector<int>> A_T = matrix_transpose(A);
    assert((*A.begin()).size()==B.size()); //断言，第一个矩阵的列必须等于第二个矩阵的行
    int new_rows = A.size();
    int new_cols = (*B.begin()).size();
    int L = B.size();
    vector<vector<int>> C(new_rows, vector<int>(new_cols,0));
    for(int k=0; k<L; k++){
        for(int i=0; i<new_rows; i++){
            int r = A[i][k];
            for(int j=0; j<new_cols;j++){
                C[i][j] += r * B[k][j];
            }
            // C[i][j] = vector_mul(A[i], get_col(B, j));
        }
    }
    return C;
}

vector<vector<int>> matrix_mul_optim2(vector<vector<int>> A, vector<vector<int>> B){
    /*二维矩阵相乘优化函数2-计算拆分
    */
    // vector<vector<int>> A_T = matrix_transpose(A);
    assert((*A.begin()).size()==B.size()); //断言，第一个矩阵的列必须等于第二个矩阵的行
    int M = A.size();
    int N = (*B.begin()).size();
    int K = B.size();  // 第二个矩阵的行
    vector<vector<int>> C(M, vector<int>(N,0));
    for(int m = 0; m < M; m += 4){
        for(int n = 0; n < N; n += 4){
            C[m + 0][n + 0] = 0;
            C[m + 0][n + 1] = 0;
            C[m + 0][n + 2] = 0;
            C[m + 0][n + 3] = 0;

            C[m + 1][n + 0] = 0;
            C[m + 1][n + 1] = 0;
            C[m + 1][n + 2] = 0;
            C[m + 1][n + 3] = 0;

            C[m + 2][n + 0] = 0;
            C[m + 2][n + 1] = 0;
            C[m + 2][n + 2] = 0;
            C[m + 2][n + 3] = 0;

            C[m + 3][n + 0] = 0;
            C[m + 3][n + 1] = 0;
            C[m + 3][n + 2] = 0;
            C[m + 3][n + 3] = 0;
            for(int k = 0;k < K; k +=4){
                //**********************************************//
                C[m + 0][n + 0] += A[m + 0][k + 0] * B[k][n + 0];
                C[m + 0][n + 0] += A[m + 0][k + 1] * B[k][n + 0];
                C[m + 0][n + 0] += A[m + 0][k + 2] * B[k][n + 0];
                C[m + 0][n + 0] += A[m + 0][k + 3] * B[k][n + 0];

                C[m + 0][n + 1] += A[m + 0][k + 0] * B[k][n + 1];
                C[m + 0][n + 1] += A[m + 0][k + 1] * B[k][n + 1];
                C[m + 0][n + 1] += A[m + 0][k + 2] * B[k][n + 1];
                C[m + 0][n + 1] += A[m + 0][k + 3] * B[k][n + 1];

                C[m + 0][n + 2] += A[m + 0][k + 0] * B[k][n + 2];
                C[m + 0][n + 2] += A[m + 0][k + 1] * B[k][n + 2];
                C[m + 0][n + 2] += A[m + 0][k + 2] * B[k][n + 2];
                C[m + 0][n + 2] += A[m + 0][k + 3] * B[k][n + 2];

                C[m + 0][n + 3] += A[m + 0][k + 0] * B[k][n + 3];
                C[m + 0][n + 3] += A[m + 0][k + 1] * B[k][n + 3];
                C[m + 0][n + 3] += A[m + 0][k + 2] * B[k][n + 3];
                C[m + 0][n + 3] += A[m + 0][k + 3] * B[k][n + 3];
                //**********************************************//
                C[m + 1][n + 0] += A[m + 1][k + 0] * B[k][n + 0];
                C[m + 1][n + 0] += A[m + 1][k + 1] * B[k][n + 0];
                C[m + 1][n + 0] += A[m + 1][k + 2] * B[k][n + 0];
                C[m + 1][n + 0] += A[m + 1][k + 3] * B[k][n + 0];

                C[m + 1][n + 1] += A[m + 1][k + 0] * B[k][n + 1];
                C[m + 1][n + 1] += A[m + 1][k + 1] * B[k][n + 1];
                C[m + 1][n + 1] += A[m + 1][k + 2] * B[k][n + 1];
                C[m + 1][n + 1] += A[m + 1][k + 3] * B[k][n + 1];

                C[m + 1][n + 2] += A[m + 1][k + 0] * B[k][n + 2];
                C[m + 1][n + 2] += A[m + 1][k + 1] * B[k][n + 2];
                C[m + 1][n + 2] += A[m + 1][k + 2] * B[k][n + 2];
                C[m + 1][n + 2] += A[m + 1][k + 3] * B[k][n + 2];

                C[m + 1][n + 3] += A[m + 1][k + 0] * B[k][n + 3];
                C[m + 1][n + 3] += A[m + 1][k + 1] * B[k][n + 3];
                C[m + 1][n + 3] += A[m + 1][k + 2] * B[k][n + 3];
                C[m + 1][n + 3] += A[m + 1][k + 3] * B[k][n + 3];
                //**********************************************//
                C[m + 2][n + 0] += A[m + 2][k + 0] * B[k][n + 0];
                C[m + 2][n + 0] += A[m + 2][k + 1] * B[k][n + 0];
                C[m + 2][n + 0] += A[m + 2][k + 2] * B[k][n + 0];
                C[m + 2][n + 0] += A[m + 2][k + 3] * B[k][n + 0];

                C[m + 2][n + 1] += A[m + 2][k + 0] * B[k][n + 1];
                C[m + 2][n + 1] += A[m + 2][k + 1] * B[k][n + 1];
                C[m + 2][n + 1] += A[m + 2][k + 2] * B[k][n + 1];
                C[m + 2][n + 1] += A[m + 2][k + 3] * B[k][n + 1];

                C[m + 2][n + 2] += A[m + 2][k + 0] * B[k][n + 2];
                C[m + 2][n + 2] += A[m + 2][k + 1] * B[k][n + 2];
                C[m + 2][n + 2] += A[m + 2][k + 2] * B[k][n + 2];
                C[m + 2][n + 2] += A[m + 2][k + 3] * B[k][n + 2];

                C[m + 2][n + 3] += A[m + 2][k + 0] * B[k][n + 3];
                C[m + 2][n + 3] += A[m + 2][k + 1] * B[k][n + 3];
                C[m + 2][n + 3] += A[m + 2][k + 2] * B[k][n + 3];
                C[m + 2][n + 3] += A[m + 2][k + 3] * B[k][n + 3];
                //**********************************************//
                C[m + 3][n + 0] += A[m + 3][k + 0] * B[k][n + 0];
                C[m + 3][n + 0] += A[m + 3][k + 1] * B[k][n + 0];
                C[m + 3][n + 0] += A[m + 3][k + 2] * B[k][n + 0];
                C[m + 3][n + 0] += A[m + 3][k + 3] * B[k][n + 0];

                C[m + 3][n + 1] += A[m + 3][k + 0] * B[k][n + 1];
                C[m + 3][n + 1] += A[m + 3][k + 1] * B[k][n + 1];
                C[m + 3][n + 1] += A[m + 3][k + 2] * B[k][n + 1];
                C[m + 3][n + 1] += A[m + 3][k + 3] * B[k][n + 1];

                C[m + 3][n + 2] += A[m + 3][k + 0] * B[k][n + 2];
                C[m + 3][n + 2] += A[m + 3][k + 1] * B[k][n + 2];
                C[m + 3][n + 2] += A[m + 3][k + 2] * B[k][n + 2];
                C[m + 3][n + 2] += A[m + 3][k + 3] * B[k][n + 2];

                C[m + 3][n + 3] += A[m + 3][k + 0] * B[k][n + 3];
                C[m + 3][n + 3] += A[m + 3][k + 1] * B[k][n + 3];
                C[m + 3][n + 3] += A[m + 3][k + 2] * B[k][n + 3];
                C[m + 3][n + 3] += A[m + 3][k + 3] * B[k][n + 3];
            }
        }
    }
    return C;
}
vector<vector<int>> matrix_mul_optim3(vector<vector<int>> A, vector<vector<int>> B){
    /*二维矩阵相乘优化函数3-（分块矩阵+改进访存局部性）
    */
    // vector<vector<int>> A_T = matrix_transpose(A);
    assert((*A.begin()).size()==B.size()); //断言，第一个矩阵的列必须等于第二个矩阵的行
    int M = A.size();
    int N = (*B.begin()).size();
    int K = B.size();  // 第二个矩阵的行

    int NUM = 8;  // 分块数
    int MT = A.size()/NUM;  // 分块矩阵的行
    int NT = (*B.begin()).size()/NUM;  // 分块矩阵的列
    int KT = B.size()/NUM;  //
    vector<vector<int>> C(M, vector<int>(N,0));
    for(int kt = 0; kt < NUM; ++kt){
        for(int it = 0; it < NUM; ++it){
            for(int jt = 0; jt < NUM; ++jt){
                int ktt = kt * KT;
                int itt = it * MT;
                int jtt = jt * NT;
                for(int k = ktt; k < ktt + KT; ++k){
                    int num_k = k * NUM;
                    for(int i = itt; i < itt + MT; ++i){
                        // int num_i = i * NUM;
                        int r = A[i][k];
                        for(int j = jtt; j < jtt + NT; ++j){
                            C[i][j] += r * B[k][j];
                        }
                    }
                }
            }
        }
    }
    return C;
}

int main(){
    vector<vector<int>> A;
    A = init_matrix(1024,8192);

    auto A_T = matrix_transpose(A);
    // print_matrix(A);
    // print_matrix(A_T);
    auto start = std::chrono::steady_clock::now();  // 开始时间
    auto B = matrix_mul_optim3(A_T, A);
    // print_matrix(B);
    auto end = std::chrono::steady_clock::now();  // 匹配结束后时间
    auto tt = duration_cast < std::chrono::milliseconds > (end - start);
    printf("Done! Timing : %lf s\n", tt.count() / 1000.0);
    cout << "The size of result matrix is " << "(" << B.size() << ", " << B[0].size() << ")" << std::endl;

    return 0;
}
```
**程序输出结果如下：**

> The size of init A matrix is (1024, 8192)
Done! Timing : 18.787000 s
The size of result matrix is (8192, 8192)

## 参考资料
* [通用矩阵乘（GEMM）优化算法](https://jackwish.net/2019/gemm-optimization.html)
* [OpenBLAS项目与矩阵乘法优化 | AI 研习社](https://www.leiphone.com/news/201704/Puevv3ZWxn0heoEv.html)
* [矩阵乘法的优化](http://blog.sciencenet.cn/blog-3316223-1085257.html)
* [how-to-optimize-gemm](https://github.com/flame/how-to-optimize-gemm)

