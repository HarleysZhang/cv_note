// leetcode 24. 两两交换链表中的节点

// 给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。
// 你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

/*
解题思路：
1，迭代法：关键是高清如何交换两个相邻节点，然后迭代交换即可。

复杂度分析：
- 时间复杂度：O(n)
- 空间复杂度：O(1)
*/

# include <stdio.h>
# include <iostream>
# include <vector>
# include <stack>

using namespace std;

// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if(head == nullptr) return nullptr;
        else if(head->next == nullptr) return head;
        ListNode* temp = new ListNode(-1);
        temp ->next = head;
        ListNode* pre = temp;
        while(pre->next != nullptr && pre->next->next != nullptr) {

            ListNode* cur = pre->next;
            ListNode* next = pre->next->next;

            pre->next = cur->next;
            cur->next = next->next;
            next->next = cur;
            pre = cur;
        }
        return temp->next;
    }
};
