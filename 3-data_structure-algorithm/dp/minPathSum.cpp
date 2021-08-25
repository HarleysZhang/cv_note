// leetcode 64
// 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
// 说明：每次只能向下或者向右移动一步。

#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// 回溯法：会超出时间限制
class Solution {
private:
    int minDist = 10000;
    void minDistBT(vector<vector<int>>& grid, int i, int j, int dist, int m, int n) {
        if (i == 0 && j == 0) dist = grid[0][0];
        if (i == m-1 && j == n-1) {
            if (dist < minDist) minDist = dist;
            return;
        }
        if (i < m-1) {
            minDistBT(grid, i + 1, j, dist + grid[i+1][j], m, n);  // 向右走
        }
        if (j < n-1) {
            minDistBT(grid, i, j + 1, dist + grid[i][j+1], m, n);  // 向下走
        }
    }

public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        int dist = 0;
        minDistBT(grid, 0, 0, dist, m, n);
        return minDist;
    }
};

// 动态规划：状态转移表法
class Solution2 {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int> > states(m, vector<int>(n, 0));
        // 第一个阶段初始化
        int sum = 0;
        for (int i = 0; i < n; i++) {  // 初始化 states 的第一行数据
            sum += grid[0][i];
            states[0][i] = sum;
        }
        sum = 0;
        for (int j = 0; j < m; j++) {  // 初始化 states 的第一列数据
            sum += grid[j][0];
            states[j][0] = sum;
        }

        // 分阶段求解，下层状态的值是基于上一层状态来的
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                states[i][j] = std::min(states[i - 1][j] + grid[i][j], states[i][j - 1] + grid[i][j]);
            }
        }
        return states[m - 1][n - 1];
    }
};

// 记忆化递归
class Solution3 {
private:
    int minDist(int i, int j, vector<vector<int> >& matrix, vector<vector<int> >& mem) { // 调用minDist(n-1, n-1);
        if (i == 0 && j == 0) return matrix[0][0];
        if (mem[i][j] > 0) return mem[i][j];

        int minUp = 10000;
        if (i - 1 >= 0) minUp = minDist(i - 1, j, matrix, mem);
        int minLeft = 10000;
        if (j - 1 >= 0) minLeft = minDist(i, j - 1, matrix, mem);
        int currMinDist = matrix[i][j] + std::min(minUp, minLeft);

        mem[i][j] = currMinDist;

        return currMinDist;
    }
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int> > mem(m, vector<int>(n, -1));

        return minDist(m - 1, n - 1, grid, mem);
    }
};

void print_matrix(vector<vector<int>> matrix) {
    /*打印二维向量（矩阵）的元素
    */
    cout << "The size of matrix is" << "(" << matrix.size() << ", " << matrix[0].size() << ")" << std::endl;
    //迭代器遍历
    // vector<vector<int >>::iterator iter;
    for (auto iter = matrix.cbegin(); iter != matrix.cend(); ++iter)
    {
        for (int i = 0; i < (*iter).size(); i++) {
            cout << (*iter)[i] << " ";
        }
        cout << std::endl;
    }
    // cout << "print success" << endl;
}

int main() {
    vector<vector<int>> grid = { {1,2,3},{4,5,6}};
    print_matrix(grid);
    Solution3 minDist;
    int minDistance = minDist.minPathSum(grid);
    cout << "Min distance is " << minDistance << endl;
}
