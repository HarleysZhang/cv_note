### 模板代码

**字符串处理**这类题目可以分为两类，一类是有前置或者后置空格的，另一类是没有前置和后置空格的。
1、如果有前后置空格，那么必须判断临时字符串非空才能输出，否则会输出空串。模板如下：

```cpp
// 模板代码
s += " "; //这里在最后一个字符位置加上空格，这样最后一个字符串就不会遗漏
string temp = "";  //临时字符串
vector<string> res; //存放字符串的数组
for (char ch : s)  //遍历字符句子
{
    if (ch == ' ') //遇到空格
    {
        if (!temp.empty()) //临时字符串非空
        {
            res.push_back(temp);
            temp.clear();  //清空临时字符串
        }
    }
    else
        temp += ch; 
}
```

2、没有前后置的空格不需要判断空串。模板如下：

```cpp
s += " ";
string temp = "";
vector<string> res;
for (char ch : s)
{
    if (ch == ' ')
    {
        res.push_back(temp);
        temp.clear();
    }
    else
        temp += ch;
}
```

### 参考资料
[作者：eh-xing-qing](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/solution/yi-ge-mo-ban-shua-bian-suo-you-zi-fu-chu-x6vh/)
