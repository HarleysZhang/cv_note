// 剑指 Offer 57. 和为s的两个数字。
// 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
// 解题思路：1，双指针法。

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
    // 双指针法，时间复杂度O(n)
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> ret;
        int i = 0, j = nums.size()-1;
        while(i<j){
            if (target-nums[i] > nums[j]) i += 1;
            else if(target-nums[i] < nums[j]) j -= 1;
            else{
                ret.push_back(nums[i]);
                ret.push_back(nums[j]);
                break;
            }
        }
        return ret;
    }
    // 两个 for 循环暴力遍历法，时间会超过
};

/*
* 打印一维向量（矩阵）的元素
*/
void print_vector(vector<int> arr) {
    cout << "The size of array is " << arr.size() << endl;
    // 迭代器遍历
    // vector<vector<int >>::iterator iter;
    for (auto iter = arr.cbegin(); iter != arr.cend(); ++iter)
    {
        cout << (*iter) << " ";
    }
    // cout << "print success" << endl;
}

int main(){
    vector<int> input = {2,7,11,15};
    int target = 9;
    Solution solution;
    auto ret = solution.twoSum(input, target);
    print_vector(ret);  // [2, 7]
}