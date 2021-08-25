// 剑指 offer31: 栈的压入、弹出序列
/*
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。
例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，
但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。
*/

#include <vector>
#include <string>
#include <stdio.h>
#include <iostream>
#include <stack>

using namespace std;

// 剑指offer31: 栈的压入、弹出序列
class Solution { // 辅助栈解法，时间超过 77.62%，空间超过 74.84%
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        int k = 0;
        stack<int> st;
        for(int i=0; i<pushed.size();i++){
            st.push(pushed[i]);
            for(int j=k;j<=i;j++){
                int temp = popped[j];
                if(temp == st.top()){
                    k++;
                    st.pop();
                }
                else{
                    break;
                }
            }
        }
        if(k==popped.size()) return true;
        else return false;
    }
};

int main() {
    cout << "hello world" << endl;
    return 0;
}

