// 剑指 Offer 18. 删除链表的节点
// 题目描述：给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。返回删除后的链表的头节点。

/*解题思路：
1. 定位节点： 遍历链表，直到 head.val == val 时跳出，即可定位目标节点。
2. 修改引用： 设节点 cur 的前驱节点为 pre ，后继节点为 cur.next ；则执行 pre.next = cur.next ，即可实现删除 cur 节点。
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
    ListNode* deleteNode(ListNode* head, int val) {
        if(head->val == val) return head -> next;
        ListNode* pre = head; ListNode* cur = head->next;
        while(cur != nullptr && cur->val != val) {
            pre = cur; 
            cur = cur->next;
        }
        pre->next = cur->next;
        return head;
    }
};
