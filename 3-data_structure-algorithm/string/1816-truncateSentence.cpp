// leetcode 1816 截断句子
// 句子 是一个单词列表，列表中的单词之间用单个空格隔开，且不存在前导或尾随空格。每个单词仅由大小写英文字母组成（不含标点符号）。
// 给你一个句子 s​​​​​​ 和一个整数 k​​​​​​ ，请你将 s​​ 截断 ​，​​​使截断后的句子仅含 前 k​​​​​​ 个单词。返回 截断 s​​​​​​ 后得到的句子。

// leetcode 557. 反转字符串中的单词 III
// 给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。

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
    string truncateSentence(string s, int k) {
        s +=  ' ';
        vector<string> res; // 存放字符串的数组
        string temp; // 临时字符串
        for(char ch:s){
            if(ch == ' '){
                res.push_back(temp);
                temp.clear();
            }
            else{
                temp += ch;
            }
        }
        s.clear();
        for(int i=0; i< k;i++){
            s += res[i] + ' ';
        }
        s.pop_back();
        return s;
    }
};

int main() {
    string s = "the sky is blue";int k=3;
    Solution solution;
    auto ret = solution.truncateSentence(s, 3);
    cout << ret << endl;
}