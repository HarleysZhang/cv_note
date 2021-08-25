// 0-1 背包问题的动态规划解法
#include <stdio.h>
#include <iostream>
#include <vector>

using namespace std;

int maxW = 0;
int weight[6] = {2,2,4,6,3};  // 物品重量
int n = 5;  // 物品个数 
int w = 9;  // 背包承受的最大重量
bool mem[5][10];  // 备忘录，默认值false

// 记忆化递归算法实现
class SolutionBacktracking{
public:
    void f(int i, int cw){  // i 表示放第 i 个物品，cw 表示当前装进背包的物品的重量和
        if (cw == w || i == n) { // cw==w表示装满了，i==n表示物品都考察完了
            if(cw > maxW)  maxW = cw;
            return;
        }
        if(mem[i][cw]) return;  // 重复状态
        mem[i][cw] = true; // 记录状态

        f(i+1, cw);  // 不放第 i 个物品
        if(cw+weight[i] <= w)
            f(i+1, cw+weight[i]);  // 放第 i 个物品
    }
};


class SolutionDP1{
public:
    // weight:物品重量，n:物品个数，w:背包可承载重量
    int knapsack1(int weight[], int n, int w){
        vector<vector<bool> >states(n, vector<bool>(w+1, false));
        states[0][0] = true;  // 第一个物品不放进背包
        if(weight[0] <= w) states[0][weight[0]] = true;  // 第一个物品放进背包
        // 动态规划-分阶段
        for(int i=1; i<n;i++){
            for(int j=0; j<w; j++)  {  // 第 i 个物品不放进背包{}
                if(states[i-1][j]) states[i][j] = states[i-1][j];
            }
            for(int j=0; j<=w-weight[i];j++){
                if(states[i-1][j]) states[i][j+weight[i]] = true;
            }
        }

        // 在最后一层变量找到最接近 w 的重量并输出结果
        for(int i=w; i>0; i--){  
            if(states[n-1][i]) return i;
        }
        return 0;
    }
};

class SolutionDP2{
public:
    // weight:物品重量，n:物品个数，w:背包可承载重量
    int knapsack2(int weight[], int n, int w){
        // bool states[w+1];
        vector<bool> states(w+1, false);

        states[0] = true;  // 第一个物品不放进背包
        if(weight[0] < w) states[weight[0]] = true;  // 第一个物品放进背包
        
        // 动态规划-分阶段
        for(int i=1; i<n;i++){
            for(int j=w-weight[i]; j>=0; j--)  {  // 第 i 个物品放进背包
                if(states[j]) states[j+weight[i]] = true;
            }
        }

        // 在最后一层变量找到最接近 w 的重量并输出结果
        for(int i=w;i>0;i--){  
            if(states[i]) return i;
        }
        return 0;
    }
};

int main(){
    SolutionBacktracking knspsack;  // 创建背包类对象
    knspsack.f(0, 0);
    // The max weight of items is 9
    cout << "The max weight of items is "<< maxW << endl; 

    SolutionDP2 knspsack2;
    int maxW2 = knspsack2.knapsack2(weight, n, w);
    cout << "The max weight of items is "<< maxW2 << endl; 

    SolutionDP1 knspsack1;
    int maxW1 = knspsack1.knapsack1(weight, n, w);
    cout << "The max weight of items is "<< maxW1 << endl; 
    return 0;
}

