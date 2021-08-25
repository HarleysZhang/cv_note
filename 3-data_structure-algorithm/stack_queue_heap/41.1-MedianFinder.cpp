// 剑指 offer 面试题41.1：数据流中的中位数, 中位数是有序列表中间的数。如果列表长度是偶数，中位数则是中间两个数的平均值。。
// 例如，[2,3,4] 的中位数是 3；[2,3] 的中位数是 (2 + 3) / 2 = 2.5
// 解题思路：

#include<vector>
#include<string>
#include<stdio.h>
#include<iostream>
#include<stack>
#include<algorithm>
#include<queue>

using namespace std;

// 剑指 Offer 41. 数据流中的中位数
class MedianFinder {  // 大根堆+小根堆 解法，时间超过 99..38%，空间超过 18.07%
private:
    // 从左到右，数据依次从大到小
    priority_queue<int> right; // 大顶堆，堆顶为最大值
    priority_queue<int, vector<int>, greater<int> > left; // 小顶堆，堆顶为最小值
public:
    /** initialize your data structure here. */
    MedianFinder() {

    }
    
    void addNum(int num) {
        // 插入数据要始终保持两个堆处于平衡状态，即较大数在左边，较小数在右边
        // 两个堆元素个数不超过 1
        if(left.size() == right.size()){
            right.push(num);
            left.push(right.top());  // 保证左边堆插入的元素始终是右边堆的最大值
            right.pop();  // 删除堆顶元素
        }
        else{
            left.push(num);
            right.push(left.top());
            left.pop();
        }
    }
    
    double findMedian() {
        if(left.size() == right.size()) return (left.top() + right.top())*0.5;
        else return left.top()*1.0;
    }
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */

int main(){
    cout << "hello world" << endl;
    return 0;
}