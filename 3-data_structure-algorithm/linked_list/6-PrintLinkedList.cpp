// 剑指 offer 6: 从尾到头打印单链表 https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/
// 题目描述：输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

/*
解题思路1：使用栈的思想(Python 用 list 模拟栈, pop 弹出栈头元素)
// 栈具有后进先出的特点，在遍历链表时将值按顺序放入栈中，最后出栈的顺序即为逆序。
// > 和 C 语言不同，C++ 的结构体可以有构造函数！
*/

# include <stdio.h>
# include <iostream>
# include <vector>
# include <stack>

using namespace std;

class Solution{
public:
    vector<int> reversePrint(ListNode* head) {
        stack<int> values;  // 创建一个不包含任何元素的 stack 适配器，并采用默认的 deque 基础容器：
        vector<int> result;
        while (head != nullptr){
            values.push(head->val);
            head = head->next;
        }
        while(!values.empty()){
            result.push_back(values.top());
            values.pop();
        }
        return result;
    }
};

/*
* 打印一维向量（矩阵）的元素
*/
void print_vector(vector<int> arr){
    cout << "The size of marray is " << arr.size() << std::endl;
    // 迭代器遍历
    // vector<vector<int >>::iterator iter;
    for(auto iter=arr.cbegin();iter != arr.cend(); ++iter)
    {
        cout << (*iter) << " ";
        cout << std::endl;
    }
    // cout << "print success" << endl;
}

// Definition for singly-linked list.
struct ListNode {
    double val;
    ListNode *next;
    // 构造函数，使用初始化列表初始化字段，构造函数可以像常规函数一样，
    // 使用默认形参来定义，而为结点的后继指针提供一个默认的 nullptr 形参是很常见的。
    ListNode(int x): val(x), next(nullptr) {}
};

// 创建元素个数为 n 的单链表
ListNode* createLinkedList(int n){
    ListNode* head = new ListNode(0);
    ListNode* cur = head;
    for (int i = 1; i < n; i++){
        int val = i * 2;
        cur->next = new ListNode(val);
        cur = cur->next;
    }

    return head;

}

int main(){
    ListNode* head = createLinkedList(10);
    Solution s1;
    vector<int> ret = s1.reversePrint(head);
    print_vector(ret);
}


