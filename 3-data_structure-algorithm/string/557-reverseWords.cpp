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
    string reverseWords(string s) {
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
        s.clear();
        for(string &str: res){
            reverse(str.begin(), str.end());
            s += str + ' ';
        }
        s.pop_back();
        return s;
    }
};

int main() {
    string s = "the sky is blue";
    Solution solution;
    auto ret = solution.reverseWords(s);
    cout << ret << endl;
}