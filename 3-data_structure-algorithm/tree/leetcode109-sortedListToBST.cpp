// leetcode109. 有序链表转换二叉搜索树 https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree
// 给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。
// 本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。

/*
解题思路：
1，将单调递增链表转化为数组，然后分治递归
2，快慢指针找链表的中间节点，然后递归
复杂度分析：
- 时间复杂度: O(n)
- 空间复杂度: O(n)
*/

# include <stdio.h>
# include <iostream>
# include <vector>
# include <stack>
# include <algorithm>

using namespace std;

// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    // 分治递归
    TreeNode* sortedListToBST(ListNode* head) {
        vector<int> vec;
        for(auto it = head; it!=nullptr ; it=it->next ){
            vec.push_back( it->val );
        }
        return recur(vec, 0, vec.size()-1);
    }

    TreeNode* recur(vector<int> &arr, int left, int right){
        if(left > right) return nullptr;
        int mid = right + (left-right)/2;  // 数组中间位置的索引
        TreeNode* node = new TreeNode(arr[mid]);
        node -> left = recur(arr, left, mid - 1);
        node -> right = recur(arr, mid + 1, right);

        return node;
    }
};