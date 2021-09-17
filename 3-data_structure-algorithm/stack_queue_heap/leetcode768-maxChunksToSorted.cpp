// leetcode 768. 最多能完成排序的块 II

/*
这个问题和“最多能完成排序的块”相似，但给定数组中的元素可以重复，输入数组最大长度为 2000，其中的元素最大为 10**8。
arr 是一个可能包含重复元素的整数数组，我们将这个数组分割成几个“块”，并将这些块分别进行排序。
之后再连接起来，使得连接的结果和按升序排序后的原数组相同。我们最多能将数组分成多少块？
*/

/*
解题思路
1，辅助栈法：栈中存放每个块内元素的最大值，栈的 size() 即为最多分块数。

题中隐含结论：
- 下一个分块中的所有数字都会大于等于上一个分块中的所有数字，即后面块中的最小值也大于前面块中最大值。
- 只有分的块内部可以排序，块与块之间的相对位置是不能变的。
- 直观上就是找到从左到右开始不减少（增加或者不变）的地方并分块。
- 要后面有较小值，那么前面大于它的都应该在一个块里面。

复杂度分析：
- 时间复杂度: O(n)
- 空间复杂度: O(1)

*/

# include <stdio.h>
# include <iostream>
# include <vector>
# include <stack>
# include <algorithm>

using namespace std;

class Solution {
public:
    int maxChunksToSorted(vector<int>& arr) {
        stack<int> ret; // 创建单调栈
        // 单调栈中只保留每个分块的最大值
        for (int i = 0; i < arr.size(); i++) {
            // 遇到一个比栈顶小的元素，而前面的块不应该有比 arr[i] 小的
            // 而栈中每一个元素都是一个块，并且栈的存的是块的最大值，因此栈中比 arr[i] 小的值都需要 pop 出来
            if (!ret.empty() && arr[i] < ret.top()) {
                int temp = ret.top();
                // 维持栈的单调递增
                while (!ret.empty() && arr[i] < ret.top()) {
                    ret.pop();
                }
                ret.push(temp);
            }
            else {
                ret.push(arr[i]);
            }
        }
        int m = ret.size();
        return m;
    }
};

int main() {
    vector<int> v1 = { 2,1,3,4,4};
    Solution s;
    auto n = s.maxChunksToSorted(v1);
    cout << n;
    return 0;
}