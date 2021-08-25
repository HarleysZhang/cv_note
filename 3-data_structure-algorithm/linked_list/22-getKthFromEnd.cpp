// 剑指 Offer 22. 链表中倒数第k个节点

/*解题思路
双指针法。不用统计链表长度。前指针 former 先向前走 k 步。
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
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode* former = head;
        ListNode* latter = head;
        for(int i=0;i<k;i++){
            former = former->next;
        }
        while(former != NULL){
            former = former->next;
            latter = latter->next;
        }
        return latter;
    }
};