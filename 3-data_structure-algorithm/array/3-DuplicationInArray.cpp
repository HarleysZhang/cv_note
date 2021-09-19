// 剑指offer面试题3（一）：找出数组中重复的数字
// 题目：在一个长度为n的数组里的所有数字都在0到n-1的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，
// 也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。例如，如果输入长度为7的数组{2, 3, 1, 0, 2, 5, 3}，
// 那么对应的输出是重复的数字2或者3。

/*
// 方法一：直接排序，然后遍历，思路很简单但是执行起来比较麻烦
// 方法二：哈希表，就是找另一个数组，把nums的元素一个一个放进去，放进去之前判断里面有没有，如果里面已经有了那就遇到重复元素，结束。
// 贼麻烦！时间复杂度O(N2),空间O(N)
// 方法三：原地置换。思路是重头扫描数组，遇到下标为i的数字如果不是i的话，（假设为m),那么我们就拿与下标m的数字交换。
在交换过程中，如果有重复的数字发生，那么终止返回ture
*/
#include <set>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     * @param numbers int整型vector 
     * @return int整型
     */
    // 哈希表法
    int duplicate(vector<int>& numbers) {
        // write code here
        multiset<int> set1;          
        for(auto i: numbers){
            set1.insert(i);
            if (set1.count(i) > 1)
                return i;
        }
        return -1;
    }
    // 原地置换法
    int findRepeatNumber(vector<int>& numbers) {
        int n = numbers.size();
        for(int i=0; i<n; i++){
            // 如果遇到下标i与nums[i]不一样，那么就要把这个nums[i]换到它应该去的下标下面
            if(numbers[i] != i){
                if(numbers[i] == numbers[numbers[i]])  // 如果那么下标下面已经被占了，那么就找到了重复值，结束！
                    return numbers[i];
                else
                    swap(numbers[i],numbers[numbers[i]]);
            }
        }
        return 0;
    }
};

void swap(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}