// 剑指 offer 面试题41.2：字符流中第一个不重复的字符
// 请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。
// 当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

// 解题思路：对于“重复问题”，惯性思维应该想到哈希或者set。对于“字符串问题”，大多会用到哈希。
// 因此一结合，应该可以想到，判断一个字符是否重复，可以选择用哈希，在c++中，可以选择用 unordered_map<char, int>。

#include<vector>
#include<string>
#include<stdio.h>
#include<iostream>
#include<stack>
#include<algorithm>
#include<queue>
#include<map>
#include <unordered_map>
using namespace std;

class Solution
{
public:
  //Insert one char from stringstream
    queue<char> q;
    unordered_map<char, int> mp;
    void Insert(char ch)
    {
         // 如果是第一次出现，则添加到队列中
         if (mp.find(ch) == mp.end()) {
             q.push(ch);
         }
         // 不管是不是第一次出现，都进行计数
         ++mp[ch];
    }
    // return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        while (!q.empty()) {
            char ch = q.front();
            // 拿出头部，如果是第一次出现，则返回
            if (mp[ch] == 1) {
                return ch;
            }
            // 不是第一次出现，则弹出，然后继续判断下一个头部
            else {
                q.pop();
            }
        }
        return '#';
    }
};