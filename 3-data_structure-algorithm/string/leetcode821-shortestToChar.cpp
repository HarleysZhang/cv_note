// leetcode821. 字符的最短距离
// 给你一个字符串 s 和一个字符 c ，且 c 是 s 中出现过的字符。

// 返回一个整数数组 answer ，其中 answer.length == s.length 
// 且 answer[i] 是 s 中从下标 i 到离它 最近 的字符 c 的 距离 。

// 两个下标 i 和 j 之间的 距离 为 abs(i - j) ，其中 abs 是绝对值函数。

/*
解题思路：
1，两次遍历
从左向右遍历，记录上一个字符 C 出现的位置 prev，那么答案就是 i - prev。
从右想做遍历，记录上一个字符 C 出现的位置 prev，那么答案就是 prev - i。
2，哈希表法
- 获取 s 中所有目标字符 c 的位置，并提前存储在数组 c_indexs 中。
- 遍历字符串 s 中的每个字符，如果和 c 不相等，就到 c_indexs 中找距离当前位置最近的下标。
复杂度分析：
- 时间复杂度: O(n)
- 空间复杂度: O(1)
*/

# include <stdio.h>
# include <iostream>
# include <vector>
# include <stack>
# include <algorithm>

using namespace std;

/*
* 打印一维向量（矩阵）的元素
*/
void print_vector(vector<int> arr) {
    cout << "The size of marray is " << arr.size() << std::endl;
    // 迭代器遍历
    // vector<vector<int >>::iterator iter;
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter)
    {
        cout << (*iter) << " ";

    }
    cout << std::endl;
    // cout << "print success" << endl;
}

class Solution {
public: // 1，两次遍历法
    vector<int> shortestToChar(string s, char c) {
        vector <int> ret;
        int prev = -10000;
        int distance = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == c) prev = i;
            distance = i - prev;
            ret.push_back(distance);
        }
        prev = 10000;
        for (int i = s.size() - 1; i >= 0; i--) {
            if (s[i] == c) prev = i;
            distance = prev - i;
            ret[i] = min(ret[i], distance);
        }
        return ret;
    }
    // 解法2：空间换时间，时间复杂度 O(n*k)
    vector<int> shortestToChar2(string s, char c) {
        int n = s.size();
        vector<int> c_indexs;
        // Initialize a vector of size n with default value 0.
        vector<int> ret(n, 0);

        for (int i = 0; i < n; i++) {
            if (s[i] == c) c_indexs.push_back(i);
        }

        for (int i = 0; i < n; i++) {
            int distance = 10000;
            if (s[i] == c) ret[i] = 0;
            else {
                for (int j = 0; j < c_indexs.size(); j++) {
                    int temp = abs(c_indexs[j] - i);
                    if (temp < distance) distance = temp;
                }
                ret[i] = distance;
            }
            
        }
        return ret;
    }
};

int main() {
    string s = "loveleetcode";
    char c = 'e';
    Solution solution;
    auto ret = solution.shortestToChar2(s, c);
    print_vector(ret);
}