// 剑指offer 63. 股票的最大利润

// 假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

/*
解题思路：

1，贪心法：
假设每天的股价都是最低价，每天都计算股票卖出去后的利润。一次 for 循环，时间复杂度：O(n)
2，暴力法：
两次 for 循环，时间复杂度 O(n^2)
*/

# include <stdio.h>
# include <iostream>
# include <vector>
# include <stack>
# include <algorithm>

using namespace std;

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        // 贪心算法：一次遍历
        int inf = 1e9; // 表示“无穷大”
        int minprice = inf, maxprofit = 0;
        for(int price: prices){
            maxprofit = max(maxprofit, (price-minprice)); // 假设每天都是最低价
            minprice = min(minprice, price);
        }
        return maxprofit;
    }
};

int main(){
    vector<int> prices = {7,1,5,3,6,4};
    Solution s1;
    int max_profit = s1.maxProfit(prices);
    cout << max_profit << endl;
    return 0;
}