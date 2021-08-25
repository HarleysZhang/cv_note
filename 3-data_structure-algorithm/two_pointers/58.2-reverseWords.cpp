// 剑指 Offer 58 - I. 翻转单词顺序
// 输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。
// 例如输入字符串 "I am a student. "，则输出"student. a am I"。

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
    string reverseWords(string s) {
        vector<string> ret;
        string temp = ""; // 存放临时字符串
        s += ' ';
        for (char c : s) {
            if (c == ' ') {
                if (!temp.empty()) {
                    ret.push_back(temp);
                    temp.clear();
                }
            }
            else {
                temp += c;
            }
        }
        reverse(ret.begin(), ret.end());
        string words = "";
        for (auto word : ret) {
            words += word + " ";
        }
        words.pop_back();
        return words;
    }
};

int main() {
    string s = "the sky is blue";
    Solution solution;
    auto ret = solution.reverseWords(s);
    cout << ret << endl;
}