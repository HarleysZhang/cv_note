// leetcode69. x 的平方根

// 实现 int sqrt(int x) 函数。计算并返回 x 的平方根，其中 x 是非负整数。
// 由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。
// 解题方法一：二分查找

#include<queue>
#include<algorithm>
#include<vector>
#include<stdio.h>
#include<iostream>
#include<math.h>

using namespace std;

class Solution {
public:
    int mySqrt(int x) {
        int low = 0, high = x, ret = -1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if ((long long)mid * mid <= x) {
                ret = mid;
                low = mid + 1;
            }
            else {
                high = mid - 1;
            }
        }
        return ret;
    }

    double mySqrt2(int x, int k) {
        double low = 0, high = x;
        double precision = pow(0.1, k);

        while (low <= high) {
            double mid = low + (high - low) / 2.0;
            if (abs(mid * mid - x) <= precision) {
                return mid;
            }
            else if (mid * mid > x) high = mid;
            else if(mid*mid < x)  low = mid;
        }

        return -1;
    }
};


int main() {
    Solution s1;
    cout << s1.mySqrt(11) << endl;  // 3
    cout << s1.mySqrt2(11, 6)  << endl;  // 3.31662
    return 0;
}