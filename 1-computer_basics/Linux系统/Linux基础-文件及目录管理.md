- [一，概述](#一概述)
- [二，文件及目录常见操作](#二文件及目录常见操作)
  - [2.1，创建、删除、移动和复制](#21创建删除移动和复制)
  - [2.2，目录切换](#22目录切换)
  - [2.3，列出目录内容](#23列出目录内容)
  - [2.4，查找目录或者文件 find/locate](#24查找目录或者文件-findlocate)
  - [2.5，查看及搜索文件内容](#25查看及搜索文件内容)
- [三，总结](#三总结)
- [四，参考资料](#四参考资料)

> 本文大部分内容参看 《Linux基础》一书，根据自己的工程经验和理解加以修改、拓展和优化形成了本篇博客，不适合 Linux 纯小白，适合有一定基础的开发者阅读。

## 一，概述
**在 Linux 中一切皆文件**。文件管理主要是涉及文件/目录的创建、删除、移动、复制和查询，有`mkdir/rm/mv/cp/find` 等命令。其中 `find` 文件查询命令较为复杂，参数丰富，功能十分强大；查看文件内容是一个比较大的话题，文本处理也有很多工具供我们使用，本文涉及到这两部分的内容只是点到为止，没有详细讲解。另外给文件创建一个别名，我们需要用到 `ln`，使用这个别名和使用原文件是相同的效果。

## 二，文件及目录常见操作
### 2.1，创建、删除、移动和复制
创建和删除命令的常用用法如下：

* 创建目录：`mkdir`
* 删除文件：`rm file(删除目录 rm -r)`
* 移动指定文件到目标目录中：`mv source_file(文件) dest_directory(目录)` 
* 复制：`cp(复制目录 cp -r)`

这些命令的常用和复杂例子程序如下

```bash
$ find ./ | wc -l  # 查看当前目录下所有文件个数(包括子目录)
14995
$ cp –r test/ newtest   # 使用指令 cp 将当前目录 test/ 下的所有文件复制到新目录 newtest 下
$ mv test.txt demo.txt  # 将文件 test.txt 改名为 demo.txt
```
### 2.2，目录切换
* 切换到上一个工作目录： `cd -`
* 切换到 home 目录： `cd or cd ~`
* 显示当前路径: `pwd`
* 更改当前工作路径为 path: `$ cd path`

### 2.3，列出目录内容
* **显示当前目录下的文件及文件属性**：`ls`
* 按时间排序，以列表的方式显示目录项：`ls -lrt`

`ls` 命令部分参数解释如下：

* `-a`：显示所有文件及目录 (. 开头的隐藏文件也会列出)
* `-l`：除文件名称外，亦将文件型态、权限、拥有者、文件大小等资讯详细列出
* `-r`：将文件以相反次序显示(原定依英文字母次序)
* `-t`： 将文件依建立时间之先后次序列出

常用例子如下：

```bash
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
### 2.4，查找目录或者文件 find/locate
1，查找文件或目录

```bash
$ find ./ -name "cali_bin*" | xargs file  # 查找当前目录下文件名含有 cali_bin 字符串的文件
./classifynet_calib_set/cali_bin.txt: ASCII text
./calib_set/cali_bin.txt:             ASCII text
./cali_bin.txt:                       ASCII text
```
2，查找目标文件夹中是否含有 `obj` 文件:

```bash
$ find ./ -name '*.o'
```
`find` 是实时查找，如果需要更快的查询，可试试 `locate`；locate 会为文件系统建立索引数据库，如果有文件更新，需要定期执行更新命令来更新索引库。

```bash
$ locate string  # 寻找包含有 string 的路径
```
### 2.5，查看及搜索文件内容
1，查看文件内容命令：`cat` `vi` `head` `tail more`。

```bash
$ cat -n  # 显示时同时显示行号 
$ ls -al | more  # 按页显示列表内容
$ head -1 filename  # 显示文件内容第一行
$ diff file1 file1  # 比较两个文件间的差别
```
2，使用 `egrep` 查询文件内容:

```bash
$ egrep "ls" log.txt  # 查找 log.txt 文件中包含 ls 字符串的行内容
-rw-r--r--   1 root root       2009 Jan 13 06:56 ls.txt
```
## 三，总结
利用 `ls -al` 命令查看文件属性及权限，已知了 `Linux` 系统内文件的三种身份(文件拥有者、文件所属群组与其他用户)，每种身份都有四种权限(`rwxs`)。可以使用 `chown`, `chgrp`, `chmod` 去修改这些权限与属性。文件是实际含有数据的地方，包括一般文本文件、数据库内容文件、二进制可执行文件(binary program)等等。

* 文件管理，目录的创建、删除、查询、管理: `mkdir` `rm` `mv` `cp`
* 文件的查询和检索命令： `find` `locate`
* 查看文件内容命令：`cat` `vi` `tail more`
* 管道和重定向命令： `;` `|` `&&` `>`

## 四，参考资料
[《Linux基础》](https://linuxtools-rst.readthedocs.io/zh_CN/latest/base/index.html)