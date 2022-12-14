- [概述](#概述)
- [使用 whatis](#使用-whatis)
- [使用 man](#使用-man)
- [查看命令程序路径 which](#查看命令程序路径-which)
- [总结](#总结)
- [参考资料](#参考资料)

### 概述

`Linux` 命令及其参数繁多，大多数人都是无法记住全部功能和具体参数意思的。在 `linux` 终端，面对命令不知道怎么用，或不记得命令的拼写及参数时，我们需要求助于系统的帮助文档； `linux` 系统内置的帮助文档很详细，通常能解决我们的问题，我们需要掌握如何正确的去使用它们。

* 需要知道某个命令的**简要说明**，可以使用 `whatis`；而更详细的介绍，则可用 `info` 命令；
* 在只记得**部分命令关键字**的场合，我们可通过 `man -k` 来搜索；
* 查看命令在哪个**位置**，我们需要使用 `which`；
* 而对于命令的**具体参数**及使用方法，我们需要用到强大的 `man` ；

### 使用 whatis

使用方法如下：

```bash
$ whatis ls # 查看 ls 命令的简要说明
ls (1)               - list directory contents
$ info ls  # 查看 ls 命令的详细说明，会进入一个窗口内，按 q 退出
File: coreutils.info,  Node: ls invocation,  Next: dir invocation,  Up: Directory listing
10.1 'ls': List directory contents
The 'ls' program lists information about files (of any type, including
directories).  Options and file arguments can be intermixed arbitrarily,
as usual.
... 省略
```

### 使用 man

查看命令 `cp` 的说明文档。

```bash
$ man cp  # 查看 cp 命令的说明文档，主要是命令的使用方法及具体参数意思
CP(1)      User Commands      CP(1)

NAME
       cp - copy files and directories
... 省略
```

在 `man` 的帮助手册中，将帮助文档分为了 `9` 个类别，对于有的关键字可能存在多个类别中， 我们就需要指定特定的类别来查看；（一般我们查询的 bash 命令，归类在1类中）；如我们常用的 `printf` 命令在分类 `1` 和分类 `3` 中都有(CentOS 系统例外)；分类 `1` 中的页面是命令操作及可执行文件的帮助；而3是常用函数库说明；如果我们想看的是 `C` 语言中 `printf` 的用法，可以指定查看分类 `3` 的帮助：

```bash
$man 3 printf
```

`man` 页面所属的分类标识(常用的是分类 `1` 和分类 `3` )

```bash
(1)、用户可以操作的命令或者是可执行文件
(2)、系统核心可调用的函数与工具等
(3)、一些常用的函数与数据库
(4)、设备文件的说明
(5)、设置文件或者某些文件的格式
(6)、游戏
(7)、惯例与协议等。例如Linux标准文件系统、网络协议、ASCⅡ，码等说明内容
(8)、系统管理员可用的管理条令
(9)、与内核有关的文件
```

### 查看命令程序路径 which

查看程序的 `binary` 文件所在路径，可用 `which` 命令。

```bash
$ which ls  # 查看 ping 程序(命令)的 binary 文件所在路径
/bin/ls
$ cd /bin;ls 
```

![image](https://img2023.cnblogs.com/blog/2989634/202212/2989634-20221214152715787-869065501.png)

查看程序的搜索路径：

```bash
$ whereis ls
ls: /bin/ls /usr/share/man/man1/ls.1.gz
```
当系统中安装了同一软件的多个版本时，不确定使用的是哪个版本时，这个命令就能派上用场。

### 总结

本文总共讲解了 `whatis info man which whereis` 五个帮助命令的使用，`Linux` 命令的熟练使用需要我们在项目中多加实践、思考和总结。

### 参考资料

[《Linux基础》](https://linuxtools-rst.readthedocs.io/zh_CN/latest/base/index.html)