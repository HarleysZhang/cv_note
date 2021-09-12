
> `FPN` 是作者 `T.-Y. Lin` 于 `2017` 发表的论文 `Feature pyramid networks for object detection.`


## 3，Focal Loss

`Focal Loss` 是在二分类问题的交叉熵（`CE`）损失函数的基础上引入的，所以需要先学习下交叉熵损失的定义。

### 3.1，Cross Entropy

在深度学习中我们常使用交叉熵来作为分类任务中训练数据分布和模型预测结果分布间的代价函数。对于同一个离散型随机变量 $\textrm{x}$ 有两个单独的概率分布 $P(x)$ 和 $Q(x)$，其交叉熵定义为：
> P 表示真实分布， Q 表示预测分布。

$$H(P,Q) = \mathbb{E}_{\textrm{x}\sim P} log Q(x)= -\sum_{i}P(x_i)logQ(x_i) \tag{1} $$

但在实际计算中，我们通常不这样写，因为不直观。在深度学习中，以二分类问题为例，其交叉熵损失（`CE`）函数如下：

$$Loss = L(y, p) = -ylog(p)-(1-y)log(1-p) \tag{2}$$

其中 $p$ 表示当预测样本等于 $1$ 的概率，则 $1-p$ 表示样本等于 $0$ 的预测概率。因为是二分类，所以样本标签 $y$ 取值为 $\{1,0\}$，上式可缩写至如下：

$$CE = \left\{\begin{matrix}
-log(p), & if \quad y=1 \\ 
-log(1-p), &  if\quad y=0  \tag{3} 
\end{matrix}\right.$$

为了方便，用 $p_t$ 代表 $p$，$p_t$ 定义如下：

$$p_t = \left\{\begin{matrix}
p, & if \quad y=1\\ 
1-p, &  if\quad y=0
\end{matrix}\right.$$

则$(3)$式可写成：

$$CE(p, y) = CE(p_t) = -log(p_t) \tag{4}$$

前面的交叉熵损失计算都是针对单个样本的，对于**所有样本**，二分类的交叉熵损失计算如下：

$$L = \frac{1}{N}(\sum_{y_i = 1}^{m}-log(p)-\sum_{y_i = 0}^{n}log(1-p))$$

其中 $m$ 为正样本个数，$n$ 为负样本个数，$N$ 为样本总数，$m+n=N$。当样本类别不平衡时，损失函数 $L$ 的分布也会发生倾斜，如 $m \ll n$ 时，负样本的损失会在总损失占主导地位。又因为损失函数的倾斜，模型训练过程中也会倾向于样本多的类别，造成模型对少样本类别的性能较差。

再衍生以下，对于**所有样本**，多分类的交叉熵损失计算如下：

$$L = \frac{1}{N} \sum_i^N L_i = -\frac{1}{N}(\sum_i \sum_{c=1}^M y_{ic}log(p_{ic})$$

其中，$M$ 表示类别数量，$y_{ic}$ 是符号函数，如果样本 $i$ 的真实类别等于 $c$ 取值 1，否则取值 0; $p_{ic}$ 表示样本 $i$ 预测为类别 $c$ 的概率。

对于多分类问题，交叉熵损失一般会结合 `softmax` 激活一起实现，`PyTorch` 代码如下，代码出自[这里](https://mp.weixin.qq.com/s/FGyV763yIKsXNM40lMO61g)。

```python

import numpy as np

# 交叉熵损失
class CrossEntropyLoss():
    """
    对最后一层的神经元输出计算交叉熵损失
    """
    def __init__(self):
        self.X = None
        self.labels = None
    
    def __call__(self, X, labels):
        """
        参数：
            X: 模型最后fc层输出
            labels: one hot标注，shape=(batch_size, num_class)
        """
        self.X = X
        self.labels = labels

        return self.forward(self.X)
    
    def forward(self, X):
        """
        计算交叉熵损失
        参数：
            X：最后一层神经元输出，shape=(batch_size, C)
            label：数据onr-hot标注，shape=(batch_size, C)
        return：
            交叉熵loss
        """
        self.softmax_x = self.softmax(X)
        log_softmax = self.log_softmax(self.softmax_x)
        cross_entropy_loss = np.sum(-(self.labels * log_softmax), axis=1).mean()
        return cross_entropy_loss
    
    def backward(self):
        grad_x =  (self.softmax_x - self.labels)  # 返回的梯度需要除以batch_size
        return grad_x / self.X.shape[0]
        
    def log_softmax(self, softmax_x):
        """
        参数:
            softmax_x, 在经过softmax处理过的X
        return: 
            log_softmax处理后的结果shape = (m, C)
        """
        return np.log(softmax_x + 1e-5)
    
    def softmax(self, X):
        """
        根据输入，返回softmax
        代码利用softmax函数的性质: softmax(x) = softmax(x + c)
        """
        batch_size = X.shape[0]
        # axis=1 表示在二维数组中沿着横轴进行取最大值的操作
        max_value = X.max(axis=1)
        #每一行减去自己本行最大的数字,防止取指数后出现inf，性质：softmax(x) = softmax(x + c)
        # 一定要新定义变量，不要用-=，否则会改变输入X。因为在调用计算损失时，多次用到了softmax，input不能改变
        tmp = X - max_value.reshape(batch_size, 1)
        # 对每个数取指数
        exp_input = np.exp(tmp)  # shape=(m, n)
        # 求出每一行的和
        exp_sum = exp_input.sum(axis=1, keepdims=True)  # shape=(m, 1)
        return exp_input / exp_sum
```

### 3.2，Balanced Cross Entropy

对于正负样本不平衡的问题，较为普遍的做法是引入 $\alpha \in(0,1)$ 参数来解决，上面公式重写如下：

$$CE(p_t) = -\alpha log(p_t) = \left\{\begin{matrix}
-\alpha log(p), & if \quad y=1\\ 
-(1-\alpha)log(1-p), &  if\quad y=0
\end{matrix}\right.$$

对于所有样本，二分类的平衡交叉熵损失函数如下：

$$L = \frac{1}{N}(\sum_{y_i = 1}^{m}-\alpha log(p)-\sum_{y_i = 0}^{n}(1 - \alpha) log(1-p))$$

其中 $\frac{\alpha}{1-\alpha} = \frac{n}{m}$，即 $\alpha$ 参数的值是根据正负样本分布比例来决定的，

### 3.3，Focal Loss Definition

虽然 $\alpha$ 参数平衡了正负样本（`positive/negative examples`），但是它并不能区分难易样本（`easy/hard examples`），而实际上，目标检测中大量的候选目标都是易分样本。这些样本的损失很低，但是由于难易样本数量极不平衡，易分样本的数量相对来讲太多，最终主导了总的损失。而本文的作者认为，易分样本（即，置信度高的样本）对模型的提升效果非常小，模型应该主要关注与那些难分样本（这个假设是有问题的，是 `GHM` 的主要改进对象）

`Focal Loss` 作者建议在交叉熵损失函数上加上一个调整因子（`modulating factor`）$(1-p_t)^\gamma$，把高置信度 $p$（易分样本）样本的损失降低一些。`Focal Loss` 定义如下：

$$FL(p_t) = -(1-p_t)^\gamma log(p_t) = \left\{\begin{matrix}
-(1-p)^\gamma log(p), & if \quad y=1\\ 
-p^\gamma log(1-p), &  if\quad y=0
\end{matrix}\right.$$

`Focal Loss` 有两个性质：

+ 当样本被错误分类且 $p_t$ 值较小时，调制因子接近于 `1`，`loss` 几乎不受影响；当 $p_t$ 接近于 `1`，调质因子（`factor`）也接近于 `0`，**容易分类样本的损失被减少了权重**，整体而言，相当于增加了分类不准确样本在损失函数中的权重。
+ $\gamma$ 参数平滑地调整容易样本的权重下降率，当 $\gamma = 0$ 时，`Focal Loss` 等同于 `CE Loss`。 $\gamma$ 在增加，调制因子的作用也就增加，实验证明  $\gamma = 2$ 时，模型效果最好。

直观地说，**调制因子减少了简单样本的损失贡献，并扩大了样本获得低损失的范围**。例如，当$\gamma = 2$ 时，与 $CE$ 相比，分类为 $p_t = 0.9$ 的样本的损耗将降低 `100` 倍，而当 $p_t = 0.968$ 时，其损耗将降低 `1000` 倍。这反过来又增加了错误分类样本的重要性（对于 $pt≤0.5$ 和 $\gamma = 2$，其损失最多减少 `4` 倍）。在训练过程关注对象的排序为正难 > 负难 > 正易 > 负易。

![难易正负样本](../../data/images/难易正负样本.jpg)

在实践中，我们常采用带 $\alpha$ 的 `Focal Loss`：

$$FL(p_t) = -\alpha (1-p_t)^\gamma log(p_t)$$

作者在实验中采用这种形式，发现它比非 $\alpha$ 平衡形式（non-$\alpha$-balanced）的精确度稍有提高。实验表明 $\gamma$ 取 2，$\alpha$ 取 0.25 的时候效果最佳。

网上有各种版本的 `Focal Loss` 实现代码，大多都是基于某个深度学习框架实现的，如 `Pytorch`和 `TensorFlow`，我选取了一个较为清晰的代码作为参考，代码来自 [这里](https://github.com/yatengLG/Retinanet-Pytorch/blob/master/Model/struct/Focal_Loss.py)。
> 后续有必要自己实现以下，有时间还要去看看 `Caffe` 的实现。

```Python
# -*- coding: utf-8 -*-
# @Author  : LG
from torch import nn
import torch
from torch.nn import functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]，为 one-hot 编码格式
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
```

`mmdetection` 框架给出的 `focal loss` 代码如下（有所删减）：

```python
# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weigh
    return loss
```
