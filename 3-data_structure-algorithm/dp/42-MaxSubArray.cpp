// 剑指 offer 42: 连续子数组的最大和。https://leetcode-cn.com/problems/maximum-subarray/
// 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

/* 求解动态规划问题的 4 个步骤：
1，确定状态
2，找到转移公式
3，确定初始条件以及边界条件
4，计算结果。
*/

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    //1, 动态规划算法
    int maxSubArray2(vector<int>& nums) {
        int* dp = new int[nums.size()];
        dp[0] = nums[0];
        int maxSum = dp[0];
        for(int i=1; i < nums.size(); i++){
            dp[i] = max(dp[i-1], 0) + nums[i];
            maxSum = max(dp[i], maxSum);
        }
        return maxSum;
    }
    //1, 动态规划，优化空间
    int maxSubArray(vector<int>& nums) {
        int sum = nums[0];
        int maxSum = nums[0];
        for(int i=1; i < nums.size(); i++){
            sum = max(sum, 0) + nums[i];
            maxSum = max(sum, maxSum);
        }
        return maxSum;
    }
};

int main() {
    vector<int> nums = {-2,1,-3,4,-1,2,1,-5,4};
    Solution s1;
    int maxSum = s1.maxSubArray(nums);
    cout << maxSum << endl;
    return 0;
}