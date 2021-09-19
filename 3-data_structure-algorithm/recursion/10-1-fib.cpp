// 剑指offer 10.1 斐波那契数列 https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/

/*
解题思路：
1，记忆化递归
2，迭代法
*/

# include <stdio.h>
# include <iostream>
# include <vector>
# include <stack>
# include <algorithm>

using namespace std;
// 剑指 offer 10-1. 斐波那契数列

class Solution {
private:
    static const int mod = 1e9 + 7;
    int m = 101;
    vector<int> vec = vector<int>(101, -1);  // c++11 之后，类 private成员初始化方式
public:
    // 1，直接递归会超出时间限制，需要使用记忆化递归
    constexpr int fib(int n) {
        if (n == 0) return 0;
        if (n == 1 || n == 2) return 1;

        if (vec[n] != -1) return vec[n];
        vec[n] = (fib(n - 1) + fib(n - 2)) % mod;

        return vec[n];
    }
    // 2，迭代求解
    int fib(int n) {
        int arr[101];
        arr[0] = 0;
        arr[1] = 1;
        arr[2] = 1;
        for (int i = 2; i < n; i++) {
            arr[i+1] = (arr[i ] + arr[i - 1]) % mod;
        }
        return arr[n];
    }
};

int main() {
 
    Solution s1;
    int num = s1.fib(10);
    cout << num << endl;
    return 0;
}