// leetcode989. 数组形式的整数加法
// 对于非负整数 X 而言，X 的数组形式是每位数字按从左到右的顺序形成的数组。
// 例如，如果 X = 1231，那么其数组形式为 [1,2,3,1]。

// 给定非负整数 X 的数组形式 A，返回整数 X+K 的数组形式。

/*
解题思路：两数相加形式的题目，可用以下加法公式模板。
当前位 = (A 的当前位 + B 的当前位 + 进位carry) % 10
/*来源：https://leetcode-cn.com/problems/add-to-array-form-of-integer/solution/989-ji-zhu-zhe-ge-jia-fa-mo-ban-miao-sha-8y9r/

while ( A 没完 || B 没完)
    A 的当前位
    B 的当前位

    和 = A 的当前位 + B 的当前位 + 进位carry

    当前位 = 和 % 10;
    进位 = 和 / 10;

判断是否还有进位

复杂度分析：
- 时间复杂度: O(n)
- 空间复杂度: O(n)
*/

# include <stdio.h>
# include <iostream>
# include <vector>
# include <stack>

using namespace std;

class Solution {
public: // 逐位相加法，使用加法模板
    vector<int> addToArrayForm(vector<int>& num, int k) {
        int sum = 0;
        int carry = 0;
        int n = num.size()-1;
        vector<int> res;
        while(n >=0 || k != 0){
            int remainder = k % 10; // k 的当前位
            if(n>=0) sum = num[n] + remainder + carry;
            else sum = remainder + carry;
            carry = sum / 10;  // 进位计算
            sum %= 10;  // 当前位计算
            res.push_back(sum);
            k /= 10;
            n -= 1;
        }
        if(carry != 0) res.push_back(carry);  // 判断是否还有进位

        reverse(res.begin(), res.end());  // 反转数组
        return res;
    }
};
