// 剑指 Offer 48. 最长不含重复字符的子字符串 https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/

#include <string>

using namespace std;

class Solution {
public:
    // 动态规划+线性遍历
    int lengthOfLongestSubstring(string s) {
        int res=0, tmp = 0, i=0;
        for(int j=0; j < s.size(); j++){
            i = j-1;
            while(i>=0 && s[i] != s[j]) i-= 1;
            if(tmp < j-i) tmp += 1;
            else tmp = j - i;
            res = max(res, tmp);
        }
        return res;
    }
};

