// 剑指 offer 面试题5：替换空格, 将一个字符串中的空格替换成 "%20"。
// 题解：双指针法： p2 指针指向扩容之后的string 最后一位，p1指向原指针最后一位，遍历指针，
// 如果 p1 遇到空格，就将 p2 向前移动三次并赋值为'%20'，没有，则将p1字符赋值给p2字符。

#include<vector>
#include<string>
#include<stdio.h>
#include<iostream>

using namespace std;

class Solution {
public:
    string replaceSpace(string s) {
        int count = 0, len = s.size();
        for(char& c:s){
            if(c == ' ') count++;
        }
        s.resize(len + 2*count);
        cout << count;
        for(int i = len-1, j=s.size()-1; i<j; i--,j--){
            if(s[i] == ' '){
                cout << s[i];
                s[j] = '0';
                s[j-1] = '2';
                s[j-2] = '%';
                j -= 2;
            }
            else{
                s[j] = s[i];
            }
        }
        return s;
    }
};