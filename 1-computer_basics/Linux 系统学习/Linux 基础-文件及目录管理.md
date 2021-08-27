- [前言](#前言)
- [概述](#概述)
  - [创建、删除、移动和复制](#创建删除移动和复制)
  - [目录切换](#目录切换)
  - [列出目录内容](#列出目录内容)
  - [查找目录及文件 find/locate](#查找目录及文件-findlocate)
  - [查看文件内容](#查看文件内容)
  - [查找文件内容](#查找文件内容)
  - [文件与目录权限修改](#文件与目录权限修改)
  - [总结](#总结)
  - [管道和重定向](#管道和重定向)
  - [设置环境变量](#设置环境变量)
  - [Bash快捷输入或删除](#bash快捷输入或删除)
  - [总结](#总结-1)
- [参考资料](#参考资料)

## 前言

本文大部分内容参看 《Linux基础》一书，根据自己的工程经验和理解加以修改、拓展和优化形成了本篇博客，不适合 Linux 纯小白，适合有一定基础的开发者阅读。

## 概述

`在 Linux 中一切皆文件`。文件管理主要是问价或目录的创建、删除、移动、复制和查询，有`mkdir/rm/mv/cp/find` 等命令。其中 `find` 文件查询命令较为复杂，参数丰富，功能十分强大；查看文件内容是一个比较大的话题，文本处理也有很多工具供我们使用，本文涉及到这两部分的内容只是点到为止，没有详细讲解。
给文件创建一个别名，我们需要用到 `ln`，使用这个别名和使用原文件是相同的效果。

### 创建、删除、移动和复制

创建和删除命令的常用用法如下：

+ 创建目录：`mkdir`
+ 删除文件：`rm file(删除目录 rm -r)`
+ 移动指定文件到目标目录中：`mv source_file(文件) dest_directory(目录)` 
+ 复制：`cp(复制目录 cp -r)`

这些命令的常用和复杂例子程序如下

```shell
$ find ./ | wc -l  # 查看当前目录下所有文件个数(包括子目录)
14995
$ cp –r test/ newtest   # 使用指令 cp 将当前目录 test/ 下的所有文件复制到新目录 newtest 下
$ mv test.txt demo.txt  # 将文件 test.txt 改名为 demo.txt
```

### 目录切换

+ 切换到上一个工作目录： `cd -`
+ 切换到 home 目录： `cd or cd ~`
+ 显示当前路径: `pwd`
+ 更改当前工作路径为 path: `$ cd path`

### 列出目录内容

+ **显示当前目录下的文件及文件属性**：`ls`
+ 按时间排序，以列表的方式显示目录项：`ls -lrt`

`ls` 命令部分参数解释如下：

+ `-a`：显示所有文件及目录 (. 开头的隐藏文件也会列出)
+ `-l`：除文件名称外，亦将文件型态、权限、拥有者、文件大小等资讯详细列出
+ `-r`：将文件以相反次序显示(原定依英文字母次序)
+ `-t`： 将文件依建立时间之先后次序列出

常用例子如下：

```shell
$ pwd
/
$ ls -al  # 列出根目录下所有的文件及文件类型、大小等资讯
total 104
drwxr-xr-x   1 root root 4096 Dec 24 01:24 .
drwxr-xr-x   1 root root 4096 Dec 24 01:24 ..
drwxrwxrwx  11 1019 1002 4096 Jan 13 09:34 data
drwxr-xr-x  15 root root 4600 Dec 24 01:24 dev
drwxr-xr-x   1 root root 4096 Jan  8 03:15 etc
drwxr-xr-x   1 root root 4096 Jan 11 05:49 home
drwxr-xr-x   1 root root 4096 Dec 23 01:15 lib
drwxr-xr-x   2 root root 4096 Dec 23 01:15 lib32
... 省略

```

### 查找目录及文件 find/locate

查找文件或目录

```shell
$ find ./ -name "cali_bin*" | xargs file  # 查找当前目录下文件名含有 cali_bin 字符串的文件
./classifynet_calib_set/cali_bin.txt: ASCII text
./calib_set/cali_bin.txt:             ASCII text
./cali_bin.txt:                       ASCII text
```

查找目标文件夹中是否含有 `obj` 文件:

```shell
$ find ./ -name '*.o'
```

`find` 是实时查找，如果需要更快的查询，可试试 `locate`；locate 会为文件系统建立索引数据库，如果有文件更新，需要定期执行更新命令来更新索引库。

```shell
$ locate string  # 寻找包含有 string 的路径
```

### 查看文件内容

查看文件内容命令：`cat` `vi` `head` `tail more`。

```shell
$ cat -n  # 显示时同时显示行号 
$ ls -al | more  # 按页显示列表内容
$ head -1 filename  # 显示文件内容第一行
$ diff file1 file1  # 比较两个文件间的差别
```

### 查找文件内容

使用 `egrep` 查询文件内容:

```shell
$ egrep "ls" log.txt  # 查找 log.txt 文件中包含 ls 字符串的行内容
-rw-r--r--   1 root root       2009 Jan 13 06:56 ls.txt
```

### 文件与目录权限修改

> chown(change owner); chgrp(change group)。Linux 文件的基本权限就有九个，分别是 owner/group/others 三种身份各有自己的 read/write/execute 权限。

`Linux/Unix` 是多人多工操作系统，所有的文件皆有拥有者。利用 `chown` 命令可以改变文件的拥有者(用户)和群组，用户可以是用户名或者`用户 ID`，组可以是组名或者`组 ID`。**注意，普通用户不能将自己的文件改变成其他的拥有者，其操作权限一般为管理员(`root` 用户)**；同时用户必须是已经存在系统中的账号，也就是在 `/etc/passwd` 这个文件中有纪录的用户名称才能改变。

+ 改变文件的拥有者： `chown`
+ 改变文件所属群组：`chgrp`
+ 改变文件读、写、执行权限(权限分数 `r:4 w:2 x:1`)属性： `chmod`
+ 递归子目录修改： `chown -R user_name folder/`
+ 增加脚本可执行权限： `chmod a+x myscript`

`cown` 范例：改变文件拥有者和群组。

```shell
root@17c30d837aba:/data/script_test# touch demo.txt
root@17c30d837aba:/data/script_test# ls -al
total 8
drwxr-xr-x  2 root root 4096 Jan 13 11:43 .
drwxrwxrwx 12 1019 1002 4096 Jan 13 11:42 ..
-rw-r--r--  1 root root    0 Jan 13 11:43 demo.txt
root@17c30d837aba:/data/script_test# chown mail:mail demo.txt
root@17c30d837aba:/data/script_test# ls -al
total 8
drwxr-xr-x  2 root root 4096 Jan 13 11:43 .
drwxrwxrwx 12 1019 1002 4096 Jan 13 11:42 ..
-rw-r--r--  1 mail mail    0 Jan 13 11:43 demo.txt
```

### 总结

利用 `ls -al` 命令查看文件属性及权限，已知了 `Linux` 系统内文件的三种身份(拥有者、群组与其他人)，每种身份都有三种权限(`rwx`)，可以使用 `chown`, `chgrp`, `chmod` 去修改这些权限与属性。文件是实际含有数据的地方，包括一般文本文件、数据库内容文件、二进制可执行文件(binary program)等等。

### 管道和重定向

+ 批处理命令连接执行，使用 `|`
+ 串联: 使用分号 `;`
+ 前面成功，则执行后面一条，否则，不执行：`&&`
+ 前面失败，则后一条执行： `||`

实例1：判断 /proc 目录是否存在，存在输出success，不存在输出 failed。

```shell
$ ls /proc > log.txt && echo  success! || echo failed.
success!
$ if ls /proc > log.txt;then echo success!;else echo failed.;fi  # 与前面脚本效果相同
success!
$ :> log.txt  # 清空文件
```

### 设置环境变量

开机启动帐号后自动执行的是 文件为 `~/profile`，然后通过这个文件可设置自己的环境变量。而在 /etc/profile文件中添加变量 对所有用户生效（永久的），用 `vi` 命令给文件 `/etc/profile` 增加变量，该变量将会对 `Linux`下所有用户有效，并且是“永久的”。

+ `export`：显示当前系统定义的所有环境变量
+ `echo $PATH`：输出当前的PATH环境变量的值，`PATH` 变量定义的是运行命令的查找路径，以冒号 `:`分割不同的路径

实例1：编辑 `/etc/profile` 文件，添加 `CLASSPATH` 变量

```shell
vim /etc/profile    
export CLASSPATH=./JAVA_HOME/lib;$JAVA_HOME/jre/lib
```

修改 `profile` 文件后需运行 `source /etc/profile` 命令才能生效，否则只能在下次重进此用户时生效。

### Bash快捷输入或删除

常用快捷键：

```shell
Ctl-U   删除光标到行首的所有字符,在某些设置下,删除全行
Ctl-W   删除当前光标到前边的最近一个空格之间的字符
Ctl-H   backspace,删除光标前边的字符
Ctl-R   匹配最相近的一个文件，然后输出
```

### 总结

+ 文件管理，目录的创建、删除、查询、管理: `mkdir` `rm` `mv` `cp`
+ 文件的查询和检索命令： `find` `locate`
+ 查看文件内容命令：`cat` `vi` `tail more`
+ 管道和重定向命令： `;` `|` `&&` `>`

## 参考资料

[《Linux基础》](https://linuxtools-rst.readthedocs.io/zh_CN/latest/base/index.html)
