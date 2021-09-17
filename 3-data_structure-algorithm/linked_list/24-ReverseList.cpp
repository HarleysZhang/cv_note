// 剑指 offer 24: 反转一个单链表。https://leetcode-cn.com/problems/reverse-linked-list/

# include <stdio.h>
# include <iostream>
# include <vector>
# include <stack>

using namespace std;

struct ListNode {
	int val;
	struct ListNode *next;
	ListNode(int x) :
			val(x), next(NULL) {
	}
};

class Solution {
public: // 迭代法
    ListNode* reverseList(ListNode* head) {
         // 判断链表为空或长度为1的情况
        if(head == nullptr || head->next == nullptr){
            return head;
        }
        ListNode* pre = nullptr; // 当前节点的前一个节点
        ListNode* next = nullptr; // 当前节点的下一个节点
        while( head != nullptr){
            next = head->next; // 记录当前节点的下一个节点位置；
            head->next = pre; // 让当前节点指向前一个节点位置，完成反转
            pre = head; // pre 往右走
            head = next;// 当前节点往右继续走
        }
        return pre;
    }
};

class Solution {
public:  // 迭代法
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr, *cur = head;
        while(cur != nullptr){
            ListNode *temp = cur -> next;  // 暂存后继节点 cur.next
            cur->next = pre;               // 修改 next 引用指向
            pre = cur;
            cur = temp;
        }
        return pre;
    }
};

