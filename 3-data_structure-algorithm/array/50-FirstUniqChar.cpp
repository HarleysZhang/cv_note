// 剑指offer 题50. 第一个只出现一次的字符位置
// 在一个字符串中找到第一个只出现一次的字符，并返回它的位置。字符串只包含 ASCII 码字符。
// map：基于红黑树，元素有序存储; unordered_map：基于散列表，元素无序存储
// 解题思路：哈希表法

#include<vector>
#include<string>
#include<stdio.h>
#include<iostream>
#include<unordered_map>

using namespace std;

class Solution {
public:
    char firstUniqChar(string s) {
        unordered_map<char, bool> dic;
        for(char c:s){
            dic[c] = dic.find(c) == dic.end();
        }
        for(char c:s){
            if(dic[c] == true)
                return c;
        }
        return ' ';
    }
    
    char FirstNotRepeatingChar(string s) {
        unordered_map<char, bool> dic;
        for(char c:s){
            dic[c] = dic.find(c) == dic.end();
        }
        for(int i=0; i<s.size();i++){
            if(dic[s[i]] == true)
                return i;
        }
        return -1;
    }
};