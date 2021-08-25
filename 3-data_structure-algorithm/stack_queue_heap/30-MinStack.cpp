#include <vector>
#include <string>
#include <stdio.h>
#include <iostream>
#include <stack>

/*
// 剑指offer30: 包含 min 函数的栈
// 定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，
// 调用 min、push 及 pop 的时间复杂度都是 O(1)。
*/

using namespace std;

class MinStack {  // 利用辅助栈

private:
    stack<int> stack1;
    stack<int> stack2;
public:
    /** initialize your data structure here. */
    MinStack() {

    }
    void push(int x) {
        stack1.push(x);
        if (stack2.empty()) {
            stack2.push(x);
        }
        else {
            if (x < stack2.top()) {
                stack2.push(x);
            }
            else {
                stack2.push(stack2.top());
            }
        }

    }
    void pop() {
        stack1.pop();
        stack2.pop();
    }

    int top() {
        return stack1.top();
    }

    int min() {
        return stack2.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->min();
 */

int main() {
    cout << "hello world" << endl;
    return 0;
}