目录
---
[toc]

## Python global 语句的作用

在编写程序的时候，如果想要**改变(重新赋值)**函数外部的变量，并且这个变量会作用于许多函数中，就需要告诉 Python 程序这个变量的作用域是全局变量，`global` 语句可以实现定义全局变量的作用。

## lambda 匿名函数好处

精简代码，`lambda`省去了定义函数，`map` 省去了写 `for` 循环过程:

```python
str_1 = ["中国", "美国", "法国", "", "", "英国"]
res = list(map(lambda x: "填充值" if x=="" else x, str_1))
print(res)  # ['中国', '美国', '法国', '填充值', '填充值', '英国']
```

## Python 错误处理

和其他高级语言一样，`Python` 也内置了一套`try...except...finally...` 的错误处理机制。

当我们认为某些代码可能会出错时，就可以用 `try` 来运行这段代码，如果执行出错，则后续代码不会继续执行，而是直接跳转至跳转至错误处理代码，即 `except` 语句块，执行完 `except` 后，如果有 `finally` 语句块，则执行。至此，执行完毕。跳转至错误处理代码，

## Python 内置错误类型

+ `IOError`：输入输出异常
+ `AttributeError`：试图访问一个对象没有的属性
+ `ImportError`：无法引入模块或包，基本是路径问题
+ `IndentationError`：语法错误，代码没有正确的对齐
+ `IndexError`：下标索引超出序列边界
+ `KeyError`: 试图访问你字典里不存在的键
+ `SyntaxError`: Python 代码逻辑语法出错，不能执行
+ `NameError`: 使用一个还未赋予对象的变量

## 简述 any() 和 all() 方法

+ `any()`: 只要迭代器中有一个元素为真就为真;
+ `all()`: 迭代器中所有的判断项返回都是真，结果才为真.

## Python 中什么元素为假？

答案：（0，空字符串，空列表、空字典、空元组、None, False）

## 提高 Python 运行效率的方法

1. 使用生成器，因为可以节约大量内存;
2. 循环代码优化，避免过多重复代码的执行;
3. 核心模块用 `Cython PyPy` 等，提高效率;
4. 多进程、多线程、协程;
5. 多个 `if elif` 条件判断，可以把最有可能先发生的条件放到前面写，这样可以减少程序判断的次数，提高效率。

## Python 单例模式

## 为什么 Python 不提供函数重载

> 参考知乎[为什么 Python 不支持函数重载？其他函数大部分都支持的？](https://www.zhihu.com/question/20053359)

我们知道 `函数重载` 主要是为了解决两个问题。

1. 可变参数类型。
2. 可变参数个数。

另外，一个函数重载基本的设计原则是，仅仅当两个函数除了参数类型和参数个数不同以外，其功能是完全相同的，此时才使用函数重载，如果两个函数的功能其实不同，那么不应当使用重载，而应当使用一个名字不同的函数。

1. 对于情况 1 ，函数功能相同，但是参数类型不同，Python 如何处理？答案是根本不需要处理，因为 `Python` 可以接受任何类型的参数，如果函数的功能相同，那么不同的参数类型在 Python 中很可能是相同的代码，没有必要做成两个不同函数。
2. 对于情况 2 ，函数功能相同，但参数个数不同，Python 如何处理？大家知道，答案就是**缺省参数(默认参数)**。对那些缺少的参数设定为缺省参数(默认参数)即可解决问题。因为你假设函数功能相同，那么那些缺少的参数终归是需要用的。所以，鉴于情况 1 跟 情况 2 都有了解决方案，Python 自然就不需要函数重载了。

## 实例方法/静态方法/类方法

`Python` 类语法中有三种方法，**实例方法，静态方法，类方法**，它们的区别如下：

+ 实例方法只能被实例对象调用，静态方法(由 `@staticmethod` 装饰器来声明)、类方法(由 `@classmethod` 装饰器来声明)，可以被类或类的实例对象调用;
+ `实例方法`，第一个参数必须要默认传实例对象，一般习惯用self。`静态方法`，参数没有要求。`类方法`，第一个参数必须要默认传类，一般习惯用 `cls` .

实例代码如下：

```Python
class Foo(object):
    """类三种方法语法形式
    """
    def instance_method(self):
        print("是类{}的实例方法，只能被实例对象调用".format(Foo))

    @staticmethod
    def static_method():
        print("是静态方法")

    @classmethod
    def class_method(cls):
        print("是类方法")


foo = Foo()
foo.instance_method()
foo.static_method()
foo.class_method()
print('##############')
Foo.static_method()
Foo.class_method()
```

程序执行后输出如下：
> 是类 <class '__main__.Foo'> 的实例方法，只能被实例对象调用
是静态方法
是类方法
##############
是静态方法
是类方法

## \_\_new\_\_和 \_\_init \_\_方法的区别

+ `__init__` 方法并不是真正意义上的构造函数, `__new__` 方法才是(类的构造函数是类的一种特殊的成员函数，它会**在每次创建类的新对象时执行**);
+ `__new__` 方法用于创建对象并返回对象，当返回对象时会自动调用 `__init__` 方法进行初始化, `__new__` 方法比 `__init__` 方法更早执行;
+ `__new__` 方法是**静态方法**，而 `__init__` 是**实例方法**。

## Python 的函数参数传递

> 参考这两个链接，stackoverflow的最高赞那个讲得很详细
[How do I pass a variable by reference?](https://stackoverflow.com/questions/986006/how-do-i-pass-a-variable-by-reference)
[Python 面试题](https://github.com/taizilongxu/interview_Python#Python%E8%AF%AD%E8%A8%80%E7%89%B9%E6%80%A7)

个人总结（有点不好）：

+ 将可变对象：列表list、字典dict、NumPy数组ndarray和用户定义的类型（类），作为参数传递给函数，函数内部将其改变后，函数外部这个变量也会改变（对变量进行重新赋值除外 `rebind the reference in the method`）
+ 将不可变对象：字符串string、元组tuple、数值numbers，作为参数传递给函数，函数内部将其改变后，函数外部这个变量不会改变

## Python 实现对函参做类型检查

`Python` 自带的函数一般都会有对函数参数类型做检查，自定义的函数参数类型检查可以用函数 `isinstance()` 实现，例如：

```Python
def my_abs(x):
    """
    自定义的绝对值函数
    :param x: int or float
    :return: positive number, int or float
    """
    if not isinstance(x, (int, float)):
        raise TypeError('bad operand type')
    if x > 0:
        return x
    else:
        return -x
```

添加了参数检查后，如果传入错误的参数类型，函数就可以抛出一个 `TypeError` 错误。

## 为什么说 Python 是动态语言

在 `Python` 中，等号 `=` 是赋值语句，可以把`任意数据类型`赋值给变量，同样一个变量可以反复赋值，而且可以是不同类型的变量，例如：

```Python
a = 100 # a是int型变量
print(a)
a = 'ABC'  # a 是str型变量
print(a)
```

Pyhon 这种变量本身类型不固定，可以反复赋值不同类型的变量称为动态语言，与之对应的是静态语言。静态语言在定义变量时必须指定变量类型，如果赋值的时候类型不匹配，就会报错，Java/C++ 都是静态语言（`int a; a = 100`）

## Python 装饰器理解

装饰器本质上是一个 Python 函数或类，**它可以让其他函数或类在不需要做任何代码修改的前提下增加额外功能**，装饰器的返回值也是一个函数/类对象。它经常用于有切面需求的场景，比如：插入日志、性能测试、事务处理、缓存、权限校验等场景，装饰器是解决这类问题的绝佳设计。有了装饰器，我们就可以**抽离出大量与函数功能本身无关的雷同代码到装饰器中并继续重用**。概括的讲，**装饰器的作用就是为已经存在的对象添加额外的功能**。

## map 与 reduce 函数用法解释

1、`map()` 函数接收两个参数，一个是函数，一个是 Iterable，map 将传入的函数依次作用到序列的**每个元素**，并将结果作为新的 Iterator 返回，简单示例代码如下：

```Python
# 示例１
def square(x):
    return x ** 2
r = map(square, [1, 2, 3, 4, 5, 6, 7])
squareed_list = list(r)
print(squareed_list)  # [1, 4, 9, 16, 25, 36, 49]
# 使用lambda匿名函数简化为一行代码
list(map(lambda x: x * x, [1, 2, 3, 4, 5, 6, 7, 8, 9]))
# 示例２
list(map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9]))　＃　['1', '2', '3', '4', '5', '6', '7', '8', '9']
```

**注意map函数返回的是一个Iterator（惰性序列），要通过list函数转化为常用列表结构**。map()作为高阶函数，事实上它是把运算规则抽象了。

2、`reduce()` 函数也接受两个参数，一个是函数（**两个参数**），一个是序列，与 `map` 不同的是**reduce 把结果继续和序列的下一个元素做累积计算**,效果如下：
`reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)`

示例代码如下：

```Python
from functools import reduce
CHAR_TO_INT = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9
}
def str2int(str):
    ints = map(lambda x:CHAR_TO_INT[x], str)  # str对象是Iterable对象
    return reduce(lambda x,y:10*x + y, ints)
print(str2int('0'))
print(str2int('12300'))
print(str2int('0012345'))  # 0012345
```

## Python 深拷贝、浅拷贝区别

> Python 中的大多数对象，比如列表 `list`、字典 `dict`、集合 `set`、`numpy` 数组，和用户定义的类型（类），都是可变的。意味着这些对象或包含的值可以被修改。但也有些对象是不可变的，例如数值型 `int`、字符串型 `str` 和元组 `tuple`。

**1、复制不可变数据类型：**

复制不可变数据类型，不管 `copy` 还是 `deepcopy`, 都是同一个地址。当浅复制的值是不可变对象（数值，字符串，元组）时和=“赋值”的情况一样，对象的 `id` 值与浅复制原来的值相同。

**2、复制可变数据类型：**

1. 直接赋值：其实就是对象的引用（别名）。
2. 浅拷贝(`copy`)：拷贝父对象，不会拷贝对象内部的子对象（拷贝可以理解为创建内存）。产生浅拷贝的操作有以下几种：
    + 使用切片 `[:]` 操作
    + 使用工厂函数（如 `list/dir/set` ）, 工厂函数看上去像函数，实质上是类，调用时实际上是生成了该类型的一个实例，就像工厂生产货物一样.
    + 使用`copy` 模块中的 `copy()` 函数，`b = a.copy()`, `a` 和 `b` 是一个独立的对象，**但他们的子对象还是指向统一对象（是引用）**。
3. 深拷贝(`deepcopy`)： copy 模块的 `deepcopy()` 方法，**完全**拷贝了父对象及其子对象，两者是完全独立的。**深拷贝，包含对象里面的子对象的拷贝，所以原始对象的改变不会造成深拷贝里任何子元素的改变**。

**注意：浅拷贝和深拷贝的不同仅仅是对组合对象来说，所谓的组合对象（容器）就是包含了其它对象的对象，如列表，类实例。而对于数字、字符串以及其它“原子”类型（没有子对象），没有拷贝一说，产生的都是原对象的引用。**更清晰易懂的理解，可以参考这篇[文章](https://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html)。

看一个示例程序，就能明白浅拷贝与深拷贝的区别了：

```python
#!/usr/bin/Python3
# -*-coding:utf-8 -*-

import copy
a = [1, 2, 3, ['a', 'b', 'c']]

b = a  # 赋值，传对象的引用
c = copy.copy(a)  # 浅拷贝
d = copy.deepcopy(a)  # 深拷贝

a.append(4)
a[3].append('d')

print(id(a), id(b), id(c), id(d))  # a 与 b 的内存地址相同
print('a = ', a)
print('b = ', b)
print('c = ', c)
print('d = ', d)  # [1, 2, 3, ['a', 'b', 'c']]
```

程序输出如下：
> 2061915781832 2061915781832 2061932431304 2061932811400
>
> a =  [1, 2, 3, ['a', 'b', 'c', 'd'], 4]
> b =  [1, 2, 3, ['a', 'b', 'c', 'd'], 4]
> c =  [1, 2, 3, ['a', 'b', 'c', 'd']]
> d =  [1, 2, 3, ['a', 'b', 'c']]

## Python 继承多态理解

多态是指对不同类型的变量进行相同的操作，它会根据对象（或类）类型的不同而表现出不同的行为。

### 总结

+ 继承可以拿到父类的所有数据和方法，子类可以重写父类的方法，也可以新增自己特有的方法。
+ 有了继承，才有了多态，不同类的对象对同一消息会作出不同的相应。

## Python 面向对象的原则

+ [Python 工匠：写好面向对象代码的原则（上）](https://www.zlovezl.cn/articles/write-solid-python-codes-part-1/)

+ [Python 工匠：写好面向对象代码的原则（中）](https://www.zlovezl.cn/articles/write-solid-python-codes-part-2/)
+ [Python 工匠：写好面向对象代码的原则（下）](https://www.zlovezl.cn/articles/write-solid-python-codes-part-3/)

## 参考资料

1. 参考[这里](https://zhuanlan.zhihu.com/p/23526961)
2. [110道Python面试题（真题）](https://zhuanlan.zhihu.com/p/54430650)
3. [关于Python的面试题](https://github.com/taizilongxu/interview_Python#15-__new__%E5%92%8C__init__%E7%9A%84%E5%8C%BA%E5%88%AB)
4. [继承和多态](http://funhacks.net/explore-python/Class/inheritance_and_polymorphism.html)
5. [Python 直接赋值、浅拷贝和深度拷贝解析](https://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html)
