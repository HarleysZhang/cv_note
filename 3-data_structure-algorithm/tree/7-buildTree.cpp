// 剑指offer07. 重建二叉树
/*
输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
解题思路：
1，递归法
+ 中序遍历的结果可以获取左右子树的元素个数；
+ 前序遍历结果可以获取树的根节点 node 的值。
*/

#include <stdio.h>
#include <string.h>
#include <vector>
#include <map>
#include <unordered_map>

using namespace std;

// Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 };


class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        // 哈希表 dic 存储中序遍历的值与索引的映射
        for(int i=0; i<inorder.size(); ++i){
            index[inorder[i]] = i;
        }
        auto root = recur(preorder, 0, 0, inorder.size());
        return root;
    }

private:
    unordered_map<int,int> index;
    TreeNode* recur(vector<int>& preorder, int root, int left, int right){
        if (left > right) return nullptr;
        int i = index[preorder[left]]; // 获取中序遍历中根节点值的索引
        TreeNode* node = new TreeNode(preorder[left]);
        node->left = recur(preorder, root+1, left, i-1);
        node->right = recur(preorder, root+i-left+1, i+1, right);

        return node;

    }
};