- [概述](#概述)
- [find 文件查找](#find-文件查找)
- [grep 文本搜索](#grep-文本搜索)
- [参考资料](#参考资料)

## 概述

`Linux` 下使用 Shell 处理文本时最常用的工具有： `find、grep、xargs、sort、uniq、tr、cut、paste、wc、sed、awk`。

## find 文件查找

`man` 文档给出的 `find` 命令的一般形式为：
```shell 
find [-H] [-L] [-P] [-D debugopts] [-Olevel] [starting-point...] [expression]
```
这对于大部分人来说都太复杂了，[-H] [-L] [-P] [-D debugopts] [-Olevel] 这几个选项并不常用，`find` 命令的常用形式可以简化为：
```shell
$ find [PATH] [option] [action]
```
**1，根据文件或者正则表达式进行匹配**
```shell
$ find .  # 查找当前目录及子目录下所有文件及文件夹
$ find /data -name "*.txt"  # 在 /data 目录及子目录下查找以 .txt 结尾的文件名
$ find . \( -name "*.txt" -o -name "*.pdf" \)  # 当前目录及子目录下查找所有以 .txt 和 .pdf 结尾的文件
$ find . -maxdepth 1 -type d  # 查找当前目录下所有的子目录
$ find . -maxdepth 1 -regex ".*\.txt$"  # 基于正则表达式匹配当前目录下的所有以 .txt 结尾的文件
./multi_classifynet_infer_ret.txt
./cali_left_img.txt
... 省略
```
**2，根据文件类型进行搜索**
```shell
find . -type 类型参数，f 普通文件，l 符号连接，d 目录，c 字符设备，b 块设备，s 套接字，p Fifo
$ find . -maxdepth 1 -type d  # 查找当前目录下的所有子目录
```
**3，基于目录深度搜索**
```shell
$ find . maxdepth 3 -type f  # 目录向下最大深度限制 3
```
**4，根据文件时间戳进行搜索**
`find . -type -f 时间戳参数`。与时间有关的选项：共有 `-atime`, `-ctime` 与 `-mtime`，以 `-mtime` 说明
+ -mtime n ： n 为数字，意义为在 n 天之前的『一天之内』被更改过内容的文件；
+ -mtime +n ：列出在 n 天之前(不含 n 天本身)被更改过内容的文件名；
+ -mtime -n ：列出在 n 天之内(含 n 天本身)被更改过内容的文件名。
+ -newer file ： file 为一个存在的文件，列出比 file 还要新的文件名

```shell
$ find /etc -newer /etc/passwd  # 寻找 /etc 底下的文件，如果文件日期比 /etc/passwd 新就列出
```
**5，与文件权限及名称有关的参数**：
+ `-name filename`：搜寻文件名为 `filename` 的文件。
+ `-size [+-]SIZE`：搜寻比 SIZE 还要大(+)或小(-)的文件。 这个 SIZE 的规格有：`c`: 代表 byte， `k`: 代表 1024 bytes。所以，要找比 50KB还要大的文件，就是 `-size +50k`。
+ `-type TYPE`：搜寻文件的类型为 TYPE 的， 类型主要有：一般正规文件 (f), 装置文件 (b, c), 目录 (d), 连结档 (l), socket (s), 及 FIFO (p) 等属性。
+ `-perm mode`：搜寻文件权限『刚好等于』 `mode` 的文件， 这个 mode 为类似 chmod 的属性值， 举例来说， `-rwxr-xr-x` 的属性为 `755`。
+ `-perm -mode`：搜寻文件权限『必须要全部囊括 mode 的权限』的文件， 举例来说，我们要搜寻 `-rwxr--r--`，亦即 `744` 的文件，使用 `-perm -744`，但是当一个文件的权限为 `-rwxr-xr-x` ，亦即 `755` 时，也会被列出来，因为 `-rwxr-xr-x` 的属性已经包括了` -rwxr--r--` 的属性了。
+ `-perm /mode`：搜寻文件权限『包含任一 `mode` 的权限』的文件， 举例来说，我们搜寻 -rwxr-xr-x ，亦即 -perm /755 时，但一个文件属性为 -rw-------也会被列出来，因为他有 `-rw....` 的属性存在。
```
范例：
```shell
root@17c30d837aba:/data# find . -maxdepth 1 -perm 777  # 查找当前目录下文件权限刚好等于777 的文件
.
./honggaozhang
./demo.sh
```
## grep 文本搜索
`grep` 支持使用正则表达式搜索文本，并把匹配的行打印出来。`grep` 命令常见用法，在文件中搜索一个单词，命令会返回一个包含 `“match_pattern”` 的文本行：
```shell
grep match_pattern file_name
grep "match_pattern" file_name
```
常用参数
+ `-o`：只输出匹配的文本行，`-v` 只输出没有匹配的文本行
+ `-c`：统计文件中包含文本的次数： `grep -c “text” filename
+ `-n`：打印匹配的行号
+ `-i`：搜索时忽略大小写
+ `-l`：只打印文件名

```shell
$ grep "class" . -R -n  # 在多级目录中对文本递归搜索(程序员搜代码的最爱)
$ grep -e "class" -e "vitural" file  #  匹配多个模式
```
## 参考资料
+ [【日常小记】linux中强大且常用命令：find、grep](cnblogs.com/skynet/archive/2010/12/25/1916873.html)
+ 鸟哥的Linux私房菜 基础篇 第四版
