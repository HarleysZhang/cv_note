//leetcode 61. 旋转链表
/*
解题思路：
将原来的链表首尾相连变成环，然后找倒数第 k 个点作为新的表头，即原来的表头向右移动 (n-1)-(k%n) 次后断开。
*/

// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if (k == 0 || head == nullptr || head->next == nullptr) {
            return head;
        }
        ListNode* cur = head;
        int n = 1;
        while(cur -> next != nullptr){
            cur = cur -> next;
            n += 1;
        } 
        cur -> next = head; // 将链表首尾相连变成环
        cur = head;
        int move = (n-1)-(k % n);
        while(move--){
            cur = cur -> next;
        }
        ListNode* ret = cur -> next;
        cur -> next = nullptr; // cur 向右移动 move 次后，断掉连接
        return ret;
    }
};