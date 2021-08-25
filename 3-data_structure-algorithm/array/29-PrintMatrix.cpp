// 剑指offer 题29：顺时针打印矩阵
// 按顺时针的方向，从外到里打印矩阵的值。
// 下图的矩阵打印结果为：1, 2, 3, 4, 8, 12, 16, 15, 14, 13, 9, 5, 6, 7, 11, 10
// 解题思路：从左到右，从上到下，检查一次是否遍历完，从右到左，从下到上

#include<vector>
#include<string>
#include<stdio.h>
#include<iostream>

using namespace std;

class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        // left to right, top to bottom
        if(matrix.empty()) return {};
        vector<int> res;
        int l = 0, r = matrix[0].size()-1, t = 0, b = matrix.size()-1;
        int nums = (r+1) * (b+1);
        while(res.size() != nums){
            for(int i=l; i<=r; i++)  // 从左往右遍历：行索引不变，列索引增加
                res.push_back(matrix[t][i]);
            t++;
            for(int j=t; j<=b; j++)  // 从上到下遍历:列索引不变，行索引增加
                res.push_back(matrix[j][r]);
            r--;
            // 检查一次是否遍历完
            if(res.size() == nums)  break;
            for(int m=r; m>=l; m--)  // 从右往左遍历：行索引不变，列索引减少
                res.push_back(matrix[b][m]);
            b--;
            for(int n=b; n>=t; n--)  // 从下往上遍历：列索引不变，行索引减少
                res.push_back(matrix[n][l]);
            l++;            
        } 
        return res;
    }
};