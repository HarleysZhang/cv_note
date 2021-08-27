- [大并发服务器架构学习](#大并发服务器架构学习)
  - [第一章 高性能的服务器架构](#第一章-高性能的服务器架构)
  - [第二章 大型网站架构演变过程](#第二章-大型网站架构演变过程)
  - [第三章 poll](#第三章-poll)
- [Linux网络编程基础](#linux网络编程基础)
  - [网络基础-综述](#网络基础-综述)
    - [1. 协议的概念](#1-协议的概念)
    - [2. b/s（浏览器/服务端模式） c/s（客户端/服务端模式）](#2-bs浏览器服务端模式-cs客户端服务端模式)
    - [3. 分层模型：OSI七层模型-TCP/IP四层模型](#3-分层模型osi七层模型-tcpip四层模型)
    - [4. 协议格式](#4-协议格式)
    - [5. NAT映射 打洞机制](#5-nat映射-打洞机制)
    - [6. 套接字socket通信原理概念](#6-套接字socket通信原理概念)
      - [6.1 网络字节序列化](#61-网络字节序列化)
      - [6.2 ip地址转换函数](#62-ip地址转换函数)
      - [6.3 socketaddr数据结构](#63-socketaddr数据结构)
      - [6.4 网络套接字函数](#64-网络套接字函数)
    - [7. TCP C/S 模型：server.c client.c](#7-tcp-cs-模型serverc-clientc)

## 3.1 课程内容概述

1. Libevent源码的跨平台编译和测试
2. Libevent原理和网络模型设置
3. event事件处理原理和实战
4. bufferevent缓冲IO
5. bufferevent、zlib实现过滤器中压缩和解压缩
6. libevent的http接口实现服务端和客户端
7. 搭建基于libevent的C++跨平台线程池
8. 基于libevent和线程池完成FTP服务器开发

# 大并发服务器架构学习

## 第一章 高性能的服务器架构

+ 网络I/O + 服务器高性能编程技术 + 数据库
+ 超出数据库连接数
+ 超出时限（队列+连接池，缓冲更新，缓冲换页，数据库读写分离进行负载均衡-replication机制）

服务器性能四大杀手：

+ 数据拷贝-缓存技术解决
+ 环境切换-有理性创建线程，单线程还是多线程好，单核服务器（采用状态机编程，效率最佳，减少线程间的切换开销），多线程能够充分发挥多核服务器的性能。
+ 内存分配-内存池
+ 锁竞争

## 第二章 大型网站架构演变过程

1. web动静资源分离
2. 缓存处理
    + 减少对网站的访问-客户端（浏览器）缓存
    + 减少对Web应用服务器的请求-前端页面缓存（squid）
    + 减少对数据库的查询-页面片段缓存ESI（Edge Side Includes）
    + 减少对文件系统I/O操作-本地数据缓存

3. Web server集群+读写分离（负载均衡）
    + 前端负载均衡：DNS负载均衡、反向代理、基于NAT的负载均衡技术、LVS、F5硬件负载均衡
    + 应用服务器负载均衡
    + 数据库负载均衡
4. CDN、分布式缓存、分库分表
5. 多数据中心+分布式存储与计算（技术点DFS分布式文件系统、Key-Value DB、Map/Reduce算法）

## 第三章 poll

Linux下有三种I/O复用模型：select、poll、epoll。

# Linux网络编程基础

## 网络基础-综述

### 1. 协议的概念

TCP协议注重数据的传输，HTTP协议注重数据的解释。

### 2. b/s（浏览器/服务端模式） c/s（客户端/服务端模式）

+ `C/S模式优点`：协议选用灵活，提前对数据进行缓存
+ C/S模式缺点：对用户安全构成威胁，开发任务工作量大
+ `B/S模式优点`：安全性高，跨平台
+ B/S模式缺点：协议选用不灵活，数据加载不缓存
+ 两者使用场景不同。

### 3. 分层模型：OSI七层模型-TCP/IP四层模型

+ `OSI七层模型`: 物理层->数据链路层->网络层（IP协议）->传输层（TCP/UDP协议）->会话层->表示层->应用层
+ `TCP/IP四层模型`：网络接口层------>网络层（IP协议）->传输层（TCP/UDP协议）--------------->应用层（FTP协议）

### 4. 协议格式

+ `数据包基本格式`：操作系统封装数据和解析数据包；路由器寻路（寻找下一路由节点）的一般思想。
+ `以太网帧格式`：|目的地址(6字节)|源地址6字节|类型2字节|数据|CRC校应2字节|
+ `arp数据包格式`:|目的地址(6字节)|源地址6字节|帧类型0806|硬件类型-协议类型-硬件地址长度-协议地址长度-op-发送端以太网地址-发送端IP地址目的以太网地址-目的IP地址（28字节ARP请求/应答）|PAD填充18字节|。`arp数据报的目的：获取下一跳mac的地址`。
+ `IP段格式`：在网络层。|4位版本号|4位首部长度|8位服务类型（TOS）|16位总长度（字节数）|
+ `TCP/UDP数据报格式`

### 5. NAT映射 打洞机制

+ `NAT（Network Address Translation，网络地址转换）`，也叫做网络掩蔽或者IP掩蔽。NAT是一种网络地址翻译技术，主要是将内部的私有IP地址（private IP）转换成可以在公网使用的公网IP（public IP）。
+ NAT可以同时让多个计算机同时联网，并隐藏其内网IP，因此也增加了内网的网络安全性；此外，NAT对来自外部的数据查看其NAT映射记录，对没有相应记录的数据包进行拒绝，提高了网络安全性。
+ 打洞机制需要借助公网 `IP` 实现；
+ 公-公通信（局域网内IP通信）：直接访问；公-私：NAT映射；私-公：NAT映射；私-私：NAT映射、打洞机制。

### 6. 套接字socket通信原理概念

+ IP地址：在网络环境中唯一标识一台主机
+ 端口号：在主机中唯一标识一个进程
+ `IP地址 + 端口`：唯一标识网络通讯中的一个进程，对应一个socket。
+ `socket` 成对出现、必须绑定IP+端口、一个文件描述符指向两个缓冲区（一个读一个写）。

#### 6.1 网络字节序列化

+ 大端存储：低地址--高位(高地址存低位)
+ 小端存储：高地址--低位

TCP/IP协议规定，`网络数据流应采用大端字节序`。为了使网络程序具有可移植性，使同样的 c 代码在大端和小端计算机上编译后都能正常运行，可以调用以下库函数做**网络字节序和主机字节序的转换**。

```C++
#include<arpa/inet.h>

uint32_t htonl(uint32_t hostlong);
uint16_t htons(uint16_t hostshort);
uint32_t ntohl(uint32_t netlong);
uint16_t ntohs(uint16_t netshort);
```

`h` 标识`host`， `n` 表示 `network`， `I` 表示 32 位长整数， `s` 表示 `16`位短整数。

#### 6.2 ip地址转换函数

``` C++
#include<arpa/inet.h>
int inet_pton(int af, const char *src, void *dst) //字符串ip转换网络字节序;
const char *inet_ntop(int af,const void *src,char *dst, socklen_t size) //网络字节序转换字符串ip;
```

#### 6.3 socketaddr数据结构

```C++
// sockaddr_in 结构体定义
struct sockaddr_in {
    sa_family_t    sin_family; /* address family: AF_INET */
    in_port_t      sin_port;   /* port in network byte order */
    struct in_addr sin_addr;   /* internet address */
};

/* Internet address. */
struct in_addr {
    uint32_t       s_addr;     /* address in network byte order */
};
```

#### 6.4 网络套接字函数

1，创建套接字`socket`函数.

```cpp
int socket(int domain, int type, int protocol);
返回值：
    成功：返回指向创建的socket的文件描述符，失败：返回-1，设置errno。
```

2，绑定 `ip` 和端口号函数 `bind`.

```cpp
#include<sys/types.h>
#include<sys/socket.h>
int bind(int cockfd, const struct sockaddr *addr, socklen_t addrlen);
sockfd:
    socket 文件描述符
addr:
    构造出IP地址+端口号
addrlen：
    sizeof(addr)长度
返回值：
    成功返回0，失败返回-1，设置errno
```

### 7. TCP C/S 模型：server.c client.c
