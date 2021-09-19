// leetcode 160. 相交链表 https://leetcode-cn.com/problems/intersection-of-two-linked-lists/
/*
### 解题思路

**双指针法**：
设节点指针 A 指向头节点 headA, 节点指针 B 指向头节点 headB
1. A 先遍历完链表 headA，然后遍历 headB;
2. B 先遍历完链表 headB，然后遍历 headA;

只要有公共节点，总路程数，或者说 A 经过的节点数和 B 经过的节点数是一样的，
如果没有公共节点，只有当 A 和 B都变成了 nullptr的时候，两者最终走的路程才是一样的。 
然后只需比较 A和 B是否相等，相等的那个位置即为公共节点，因为此使，两者走的步数开始相等了。

```
### 复杂度分析：
- 时间复杂度：O(n)
- 空间复杂度：O(1)
*/


class Solution {
public:
    // 双指针法
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode* A =  headA;
        ListNode* B = headB;
        while(A != B){
            if(A != nullptr) A = A->next;
            else A = headB;
            if (B != nullptr) B = B->next;
            else B = headA;
        }
        return A;
    }
    // 哈希表法，哈希表中存储链表节点指针
    ListNode *getIntersectionNode2(ListNode *headA, ListNode *headB) {
        unordered_set<ListNode *> visited;
        ListNode* temp = headA;
        while(temp != nullptr){
            visited.insert(temp);
            temp = temp -> next;
        }
        temp = headB;
        while(temp != nullptr){
            // count 方法判断哈希表中是否存在 temp 关键字
            if(visited.count(temp)) return temp;
            else temp = temp -> next;
        }
        return nullptr;
    }
    // vector 法，vector 中元素为链表节点指针
    ListNode *getIntersectionNode3(ListNode *headA, ListNode *headB) {
        vector<ListNode *> visited;
        ListNode* temp = headA;
        while(temp != nullptr){
            visited.push_back(temp);
            temp = temp -> next;
        }
        temp = headB;
        while(temp != nullptr){
            // find 函数查找 vector 中是否存在 temp 元素
            if(visited.end() != find(visited.begin(), visited.end(), temp)) return temp;
            else temp = temp -> next;
        }
        return nullptr;
    }
};