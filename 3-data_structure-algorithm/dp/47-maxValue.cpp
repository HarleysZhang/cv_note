// 剑指 Offer 47. 礼物的最大价值

class Solution {  // 状态转移方程法
private:
    int minDist(int i, int j, vector<vector<int> >& matrix, vector<vector<int> >& mem) { // 调用minDist(n-1, n-1);
        if (i == 0 && j == 0) return matrix[0][0];
        if (mem[i][j] > 0) return mem[i][j];

        int minUp = -10000;
        if (i - 1 >= 0) minUp = minDist(i - 1, j, matrix, mem);
        int minLeft = -10000;
        if (j - 1 >= 0) minLeft = minDist(i, j - 1, matrix, mem);
        int currMinDist = matrix[i][j] + std::max(minUp, minLeft);

        mem[i][j] = currMinDist;

        return currMinDist;
    }
public:
    int maxValue(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int> > mem(m, vector<int>(n, -1));

        return minDist(m - 1, n - 1, grid, mem);
    }
};