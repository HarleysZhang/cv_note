// 剑指 offer 面试题9：用两个栈实现队列, 用两个栈来实现一个队列，完成队列的 Push 和 Pop 操作。
// in 栈用来处理入栈（push）操作，out 栈用来处理出栈（pop）操作。一个元素进入 in 栈之后，出栈的顺序被反转。
// 当元素要出栈时，需要先进入 out 栈，此时元素出栈顺序再一次被反转，因此出栈顺序就和最开始入栈顺序是相同的，
// 先进入的元素先退出，这就是队列的顺序。

#include<vector>
#include<string>
#include<stdio.h>
#include<iostream>
#include<stack>

using namespace std;

class Solution
{
public:
    void push(int node) {
        stack1.push(node);
    }
    int pop() {
        int res;
        if(stack2.empty()){
            while(!stack1.empty()){
                int temp = stack1.top();
                stack1.pop();
                stack2.push(temp);
            }
        }
        res = stack2.top();
        stack2.pop();
        return res;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};