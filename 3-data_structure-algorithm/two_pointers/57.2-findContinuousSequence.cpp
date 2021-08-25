// 剑指 Offer 57 - II. 和为s的连续正数序列
// 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
// 序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。


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
    // 暴力遍历法
    vector<vector<int>> findContinuousSequence(int target) {
        vector<vector<int>> m;
        for(int i = 1; i < target; i++){
            vector<int> v;
            int sum = 0;
            for(int j=i;j<target;j++){
                if(sum<target){
                    v.push_back(j);
                    sum += j;
                }
                else if(sum == target){
                    m.push_back(v);
                    break;
                }
                else break;
            }
        }
        return m;
    }
    // 滑动窗口法，时间复杂度O(n)，超过 63.42%
    vector<vector<int>> findContinuousSequence(int target) {
        vector<vector<int>> m; vector<int> v;
        int sum = 0, i=1, j=1;
        int limit = (target-1)/2;
        while(i <= limit){
            if(sum < target){  // 右边界向右移动
                sum += j;
                j++;
            }
            else if(sum > target){  // 左边界向右移动
                sum -= i;
                i++;
            }
            else{
                for(int k=i; k<j; k++){  // 保持符合条件的窗口内的元素值
                    v.push_back(k);
                }
                m.push_back(v);
                v.clear();
                sum -= i;
                i++;
            }
        }
        return m;
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

void print_matrix(vector<vector<int>> matrix){
    /*打印二维向量（矩阵）的元素
    */
    cout << "The size of matrix is" << "(" << matrix.size() << ", " << matrix[0].size() << ")" << std::endl;
    //迭代器遍历
    for(auto iter=matrix.cbegin();iter != matrix.cend(); ++iter)
    {
        for(int i = 0;i<(*iter).size();i++){
            cout << (*iter)[i] << " ";
        }
        cout << std::endl;
    }
     cout << "print success" << endl;
}

int main(){
    int target = 9;
    Solution solution;
    auto ret = solution.findContinuousSequence(target);
    print_matrix(ret);  // 
}