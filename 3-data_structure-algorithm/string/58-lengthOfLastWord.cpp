// leetcode58. 最后一个单词的长度
// 给你一个字符串 s，由若干单词组成，单词前后用一些空格字符隔开。返回字符串中最后一个单词的长度。
// 单词：是指仅由字母组成、不包含任何空格字符的最大子字符串。

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
    int lengthOfLastWord(string s) {
        s +=  ' ';
        vector<string> res; // 存放字符串的数组
        string temp; // 临时字符串
        for(char ch:s){
            if(ch == ' '){
                if(!temp.empty()){
                    res.push_back(temp);
                    temp.clear();
                }
            }
            else{
                temp += ch;
            }
        }
        string last_word = res.back();  // 数组最后一个元素
        return last_word.size();
    }
};

int main() {
    string s = "the sky is blue";
    Solution solution;
    auto ret = solution.lengthOfLastWord(s);
    cout << ret << endl;
}