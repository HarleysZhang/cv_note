// 剑指 Offer 18.2  删除链表中重复的节点
// 题目描述：在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 
// 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

/*解题思路：
迭代解法
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
    ListNode* deleteDuplication(ListNode* head) {
        ListNode *vhead = new ListNode(-1);
        vhead->next = head;
        ListNode* pre = vhead,*cur = head;
        while(cur){
            if(cur->next && cur->val==cur->next->val){
                cur = cur->next;
                while(cur->next && cur->val == cur->next->val){
                    cur = cur->next;
                }
                cur = cur -> next;
                pre->next = cur;
            }
            else{
                pre = cur;
                cur = cur->next;
            }
        }
        return vhead->next;
    }
};