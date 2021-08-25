#include <vector>
#include <string>
#include <stdio.h>
#include <iostream>
#include <stack>
#include <queue>
/*
// 剑指offer30: 队列的最大值
// 请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。
// 若队列为空，pop_front 和 max_value 需要返回 -1

// 解题思路：定义一个单调递减的辅助队列（双端队列）
*/

using namespace std;

class MaxQueue {
private:
    queue<int> que1;
    deque<int> que2;  // 辅助队列，头部位置存放最大值值
public:
    MaxQueue() {

    }
    int max_value() {
        if(que1.empty())
            return -1;
        return que2.front();
    }
    void push_back(int value) {
        // 维护单调递减队列
        while(!que2.empty() && que2.back() < value){
            que2.pop_back();  // 移除队尾元素直到队尾元素大于新添加元素
        }
        
        que2.push_back(value);
        que1.push(value);
    }
    int pop_front() {
        if(que1.empty()) return -1;
        else{
            int ans = que1.front();
            if( ans == que2.front()) que2.pop_front();
            que1.pop();
            return ans;
        }
    }
};

/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue* obj = new MaxQueue();
 * int param_1 = obj->max_value();
 * obj->push_back(value);
 * int param_3 = obj->pop_front();
 */