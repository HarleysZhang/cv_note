// leetcode 1805. 字符串中不同整数的数目

// 给你一个字符串 word ，该字符串由数字和小写英文字母组成。

// 请你用空格替换每个不是数字的字符。例如，"a123bc34d8ef34" 将会变成 " 123  34 8  34" 。
// 注意，剩下的这些整数为（相邻彼此至少有一个空格隔开）："123"、"34"、"8" 和 "34" 。
// 返回对 word 完成替换后形成的不同整数的数目。
// 只有当两个整数的 不含前导零的十进制表示不同，才认为这两个整数也不同。


#include<vector>
#include<string>
#include<stdio.h>
#include<iostream>
#include<stack>
#include<algorithm>
#include<queue>
#include<set>

using namespace std;

class Solution {
public:
    int numDifferentIntegers(string word) {
        set<string> s;
        word += 'a';
        string temp; // 临时字符串
        for(char ch:word){
            // 如果遇到字母且临时字符串非空，就把它加入集合并重置临时字符串
            if(isalpha(ch)){   
                if(!temp.empty()){
                    s.insert(temp);
                    temp.clear();
                }
            }
            else{
                if(temp == "0") temp.clear();  // "001" 和 "1" 是等值的
                temp += ch;
            }
        }
        return s.size();
    }
};

int main() {
    string word = "the423df34fds23";
    Solution solution;
    auto ret = solution.numDifferentIntegers(word);
    cout << ret << endl;
}