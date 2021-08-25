#include <vector>
#include <stdio.h>
#include <string>
#include <iostream>

using namespace std;

vector<vector<int>> ret;
void printQueens2(vector<string> result);
void printQueens(vector<vector<string>> result);

// leetcode N皇后-回溯解法
class Solution {
private:
    vector<vector<string>> result;
    void backtracking(int n, int row, vector<string>& chessboard){
        if(row == n) {
            result.push_back(chessboard);
            return;
        }
        for(int column=0; column < n; column++){
            if (isOK(row, column, n, chessboard)){
                chessboard[row][column] = 'Q';  // 放置皇后
                backtracking(n, row+1, chessboard);
                chessboard[row][column] = '.';  // 回溯，撤销处理结果
            }
        }
    }
    // 判断 row 行 column 列放置皇后是否合适
    bool isOK(int row, int column, int n, vector<string>& chessboard){
        
        int leftup = column - 1; int rightup = column + 1;  // 左上角和右上角

        for(int i = row-1; i>=0; i--){  // 逐行网上考察每一行
            // 判断第 i 行的 column 列是否有棋子
            if(chessboard[i][column] == 'Q') {
                return false;
            }
            // 考察左上对角线：判断第i行leftup列是否有棋子   
            if(leftup >=0 ){
                if(chessboard[i][leftup] == 'Q') return false;
            }
            // 考察左上对角线：判断第i行rightup列是否有棋子
            if(rightup < n){
                if(chessboard[i][rightup] == 'Q') return false;
            }
            --leftup;
            ++rightup;
        }
        return true;
    }  

public:
    vector<vector<string>> solveNQueens(int n) {
        result.clear();
        std::vector<std::string> chessboard(n, std::string(n, '.'));
       
        backtracking(n, 0, chessboard);
        return result;
    }
};

void printQueens(vector<vector<string>> result){
    for(int i = 0; i < result.size(); ++i){
        for(int j = 0; j < result[0].size();++j){
            cout << result[i][j] << "\n";
        }
        cout << "The " << i+1 << " solution of chessboard" << "\n";
    }
}

void printQueens2(vector<string> result){
    for(int i = 0; i < result.size(); ++i){
        cout << result[i] << "\n" << "\n";
    }   
}

int main(){

    Solution solve_nqueens;
    auto ret = solve_nqueens.solveNQueens(4);
    printQueens(ret);
}
