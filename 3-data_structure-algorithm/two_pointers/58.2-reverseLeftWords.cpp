// 剑指 Offer 58 - II. 左旋转字符串

// 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。
// 比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

#include<vector>
#include<string>
#include<stdio.h>
#include<iostream>
#include<stack>
#include<algorithm>
#include<queue>

using namespace std;


class Solution {
public:
    // 字符串切片1，时间和空间复杂度都为O(n)
    string reverseLeftWords2(string s, int n) {
        if (s.empty()) return s;
        auto s1 = s.substr(n);
        for(int i=0; i<n; i++){
            s1.push_back(s[i]);  // s1 += s[i];
        }
        return s1;
    }
    // 字符串切片2，时间和空间复杂度都为O(n)
    string reverseLeftWords2(string s, int n) {
        if (s.empty()) return s;
        string ans;
        ans = s.substr(n) + s.substr(0, n);
        return ans;
    }
    // 原地三次翻转
    string reverseLeftWords(string s, int n) {
        if (s.empty()) return s;
        reverse(s.begin(), s.begin()+n);
        reverse(s.begin()+n, s.end());
        reverse(s.begin(), s.end());
        return s;
    }

};

int main(){
    string s = "rutervzs";
    int target = 9;
    Solution solution;
    auto ret = solution.reverseLeftWords(s, 3);
    cout << ret;
}