#include <stdio.h>
#include <vector>
#include <string>
#include <stack>

using namespace std;


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
    // 前序遍历
    void preOrder(TreeNode* root, vector<int> &ret){
        if(root == nullptr) return ;
        ret.push_back(root->val);
        preOrder(root->left, ret);
        preOrder(root->right, ret);
    }
    // 中序遍历
    void inOrder(TreeNode* root, vector<int> &ret){
        if(root == nullptr) return ;
        inOrder(root->left, ret);
        ret.push_back(root->val);
        inOrder(root->right, ret);
    }
    // 后续遍历
    void postOrder(TreeNode* root, vector<int> &ret){
        if(root == nullptr) return ;
        postOrder(root->left, ret);
        postOrder(root->right, ret);
        ret.push_back(root->val);
    }

    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        preOrder(root, res);
        return res;
    }
    vector<int> inOrderTraversal(TreeNode* root) {
        vector<int> res;
        inOrder(root, res);
        return res;
    }
    vector<int> postOrderTraversal(TreeNode* root) {
        vector<int> res;
        postOrder(root, res);
        return res;
    }
};
