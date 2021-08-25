// 剑指 offer 面试题59: 滑动窗口的最大值
// 给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。
/*解题思路：
方法一：对于每个滑动窗口，可以使用 O(k) 的时间遍历其中的每一个元素，找出其中的最大值。
对于长度为 n 的数组 nums 而言，窗口的数量为 n-k+1，算法的时间复杂度为 O((n-k+1)*k)=O(n*k)。
方法二：维护单调递减的双端队列！
如果发现队尾元素小于要加入的元素，则将队尾元素出队，直到队尾元素大于新元素时，再让新元素入队，从而维护一个单调递减的队列。
*/

#include<queue>
#include<algorithm>
#include<vector>
#include<stdio.h>
#include<iostream>

using namespace std;

class MyQueue { //单调队列（从大到小）
public:
    deque<int> que; // 使用deque来实现单调队列
    void pop(int value) {
        if (!que.empty() && value == que.front()) {
            que.pop_front();
        }
    }
    void push(int value) {
        while (!que.empty() && value > que.back()) {
            que.pop_back();
        }
        que.push_back(value);

    }
    int front() {
        return que.front();
    }
};


// 剑指 Offer 59 - I. 滑动窗口的最大值
class Solution {  
public:
    // 简单方法：遍历滑动窗口找最大值，合理选择区间
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ret;
        if (nums.size() == 0 && k == 0) return ret;
        for (int i = 0; i <= nums.size() - k; i++) {
            int maxNum = nums[i];
            for (int j = i; j < (i + k); j++) {
                if (nums[j] > maxNum)
                    maxNum = nums[j];
            }
            ret.push_back(maxNum);
        }
        return ret;
    }
    // 维护一个单调队列，队头是最大值
    vector<int> maxSlidingWindow2(vector<int>& nums, int k) {
        vector<int> ret;
        deque<int> window;  // 创建双端队列
        // 先将第一个窗口的值按照规则入队
        for (int i = 0; i < k; i++) {
            while (!window.empty() && window.back() < nums[i]) {
                window.pop_back();
            }
            window.push_back(nums[i]);
        }
        ret.push_back(window.front());
        
        for (int j = k; j < nums.size(); j++) {
            if (nums[j - k] == window.front()) window.pop_front();  // 模拟滑动窗口的移动
            while (!window.empty() && window.back() <= nums[j]) {
                window.pop_back();
            }
            window.push_back(nums[j]);
            ret.push_back(window.front());
        }
        return ret;
    }
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

int main() {
    Solution s1;
    vector<int> nums = { 1, 3, -1, -3, 5, 3, 6, 7 };
    int maxNum = *max_element(nums.begin(), nums.end());

    int k = 3;
    auto ret = s1.maxSlidingWindow2(nums, k);
    print_vector(ret);
}