// leetcode 394. 字符串解码

/*
解题思路：本题难点在于括号内嵌套括号，需要从内向外生成与拼接字符串，这与栈的先入后出特性对应。
参考：https://leetcode-cn.com/problems/decode-string/solution/decode-string-fu-zhu-zhan-fa-di-gui-fa-by-jyd/
复杂度分析：

- 时间复杂度 O(N)： s；
- 空间复杂度 O(N)：辅助栈在极端情况下需要线性空间，例如 2[2[2[a]]]。
*/
# include <stdio.h>
# include <iostream>
# include <vector>
# include <stack>
# include <algorithm>
#include <utility>
# include <string>

using namespace std;

class Solution {
public:

    string decodeString(string s) {
        stack<pair<string, int>> s1;
        string res = "";
        int num = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] >= '0' && s[i] <= '9') {
                num *= 10;
                num += (s[i] - '0');
            }
            else if (s[i] == '[') {
                s1.push(make_pair(res, num));
                num = 0;
                res = "";
            }
            else if (s[i] == ']') {
                auto cur_num = s1.top().second;
                auto latest_res = s1.top().first;
                s1.pop();
                for (int j = 0; j < cur_num; j++) latest_res = latest_res + res; // res 加 n 次
                res = latest_res;
            }
            else {
                res += s[i];
            }
        }
        return res;
    }
};

int main(){
    string s = "3[a]2[bc]";
    Solution solution;
    auto res = solution.decodeString(s);
    cout << res << endl;
    return  0;
}