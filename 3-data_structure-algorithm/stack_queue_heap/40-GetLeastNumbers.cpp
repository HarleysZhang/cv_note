// 剑指 offer 面试题40：最小的 K 个数, 输入整数数组 arr ，找出其中最小的 k 个数。
// 例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
// 解题思路：
// 优先队列(最大堆，优先输出最大数): priority_queue<Type, Container, Functional>  // 默认定义最大堆
// priority_queue<int, vector<int>, greater<int> >p;  // 定义最小堆

// 1，数组原地排序法：对原数组从小到大排序后取出前 k 个数即可。时间复杂度：O(nlog n)，空间复杂度：O(log n)。
// 2，使用最大堆结构：时间复杂度：O(nlongk), 插入容量为k的大根堆时间复杂度为O(longk), 一共遍历n个元素；空间复杂度：O(k)。
// 3, 快速排序算法：TODO.

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
    // // stl 自带的 sort() 排序算法
    vector<int> getLeastNumbers1(vector<int>& arr, int k) {
        vector<int> ret(k, 0);
        sort(arr.begin(), arr.end());
        for(int i=0;i<k;++i){
            ret[i] = arr[i];
        }
        return ret;
    }
    // 大顶堆维护小顶堆的方法
    vector<int> getLeastNumbers2(vector<int>& arr, int k) {
        vector<int> vec(k,0);
        if(k==0)
            return vec;
        priority_queue<int> heap;  // 大顶堆，堆顶为最大值
        for(int i=0;i<(int)arr.size();i++){
            if(i<k){
                heap.push(arr[i]);
            }
            else{
                if(heap.top() > arr[i]){  // 使用大顶堆来维护最小堆
                    heap.pop();
                    heap.push(arr[i]);
                }
            }
        }
        for(int i=0;i<k;i++){
            vec[i] = heap.top();
            heap.pop();
        }
        return vec;
    }
    // 直接使用小顶堆
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int> vec(k,0);
        if(k==0)
            return vec;
        priority_queue<int, vector<int>, greater<int> > heap; // 小顶堆，堆顶为最小值
        priority_queue<int> heap2; // 大顶堆，堆顶为最大值
        for(int i=0;i<(int)arr.size();i++){
            heap.push(arr[i]);
        }
        for(int i=0;i<k;i++){
            vec[i] = heap.top();
            heap.pop();
        }
        return vec;
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

int main(){
    vector<int> arr = {2, 19, 9, 7,11,15};
    int k = 3;
    Solution solution;
    auto ret = solution.getLeastNumbers(arr, k);
    print_vector(ret);  // [2, 7, 11]
}