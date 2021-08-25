// 剑指面试题4：二维数组中的查找
// 题目：在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按
// 照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个
// 整数，判断数组中是否含有该整数
//
#include<iostream>
#include<stdlib.h>
#include<vector>

using namespace std;

vector<vector<int>> init_matrix(int mm, int nn){
    /*初始化指定行和列的二维矩阵
    */
    int random_integer;

    vector<vector<int>> matrix(mm, vector<int>(nn,0)); // 根据初始化二维数组大小为 mm*nn，所有元素为0
    cout << "The size of init matrix is " << "(" << matrix.size() << ", " << matrix[0].size() << ")" << std::endl;
    for (int i=0; i < matrix.size();i++)
    {
        for(int j=0;j < matrix[i].size();j++)
        {
            random_integer = rand() % 128;
            matrix[i][j] = random_integer;  // 利用下标给二维数组赋值
            // matrix[i].push_back(random_integer);  // 利用push_back给vactor添加元素
        }
    }
    return matrix;
}

void print_matrix(vector<vector<int>> matrix){
    /*打印二维向量（矩阵）的元素
    */
    cout << "The size of matrix is" << "(" << matrix.size() << ", " << matrix[0].size() << ")" << std::endl;
    //迭代器遍历
    for(auto iter=matrix.cbegin();iter != matrix.cend(); ++iter)
    {
        for(int i = 0;i<(*iter).size();i++){
            cout << (*iter)[i] << " ";
        }
        cout << std::endl;
    }
     cout << "print success" << endl;
}

bool Find(int target, vector<vector<int>> matrix){
    if(matrix.size() == 0)
        return false;
    int rows = matrix.size();
    int cols = (*matrix.begin()).size();
    int r = 0, c = cols -1; // 从右上角开始
    while(r<=rows-1 && c >>0){
        if(target == matrix[r][c])
            return true;
        else if(target > matrix[r][c])
            r++;
        else
            c--;
    }
    return false;
}
int main(){
    vector<vector<int>> A;
    A = init_matrix(10,5);
    print_matrix(A);
    bool flag = Find(54, A);
    cout << flag;
}

