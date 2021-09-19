// 剑指 Offer 10- II. 青蛙跳台阶问题



# include <stdio.h>
# include <iostream>
# include <vector>
# include <stack>
# include <algorithm>

using namespace std;

class Solution {
private:
    static const int mod = 1e9 + 7;
public:
    // 动态规划法 
    int numWays(int n) {
        int dp[n+1];
        if( n == 0 || n == 1) return 1;
        dp[0] = 1;
        dp[1] = 1;
        for(int i=2; i<=n; i++){
            dp[i] = (dp[i-1] + dp[i-2]) % mod;
        }
        return dp[n];
    }
    // 递归法
    int numWays2(int n) {
        if(n == 1) return 1;
        if(n == 2) return 2;
        return numWays2(n-1) + numWays2(n-2);
    }
};

int main() {
 
    Solution s1;
    int num = s1.numWays(7);
    cout << num << endl;  // 21
    return 0;
}
