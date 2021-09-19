- [一，常见数据结构](#一常见数据结构)
  - [1，数组](#1数组)
    - [3-找出数组中重复的数字](#3-找出数组中重复的数字)
    - [4-二维数组中的查找](#4-二维数组中的查找)
    - [5-替换空格](#5-替换空格)
    - [29-顺时针打印矩阵](#29-顺时针打印矩阵)
    - [50-第一个只出现一次的字符位置](#50-第一个只出现一次的字符位置)
    - [leetcode 989-数组形式的整数加法](#leetcode-989-数组形式的整数加法)
  - [2，链表](#2链表)
    - [6-从尾到头打印单链表](#6-从尾到头打印单链表)
    - [18.1-删除链表的节点](#181-删除链表的节点)
    - [18.2 删除链表中重复的节点](#182-删除链表中重复的节点)
    - [22-链表中倒数第k个节点](#22-链表中倒数第k个节点)
    - [23-链表中环的入口结点](#23-链表中环的入口结点)
    - [24-反转一个单链表](#24-反转一个单链表)
    - [52-两个链表的第一个公共节点](#52-两个链表的第一个公共节点)
    - [leetcode 61-旋转链表](#leetcode-61-旋转链表)
    - [leetcode 24-两两交换链表中的节点](#leetcode-24-两两交换链表中的节点)
  - [3，栈队列堆](#3栈队列堆)
    - [9-用两个栈实现队列](#9-用两个栈实现队列)
    - [30-包含 min 函数的栈](#30-包含-min-函数的栈)
    - [31-栈的压入、弹出序列](#31-栈的压入弹出序列)
    - [40-最小的 K 个数](#40-最小的-k-个数)
    - [41.1-数据流中的中位数](#411-数据流中的中位数)
    - [41.2-字符流中第一个不重复的字符](#412-字符流中第一个不重复的字符)
    - [59-滑动窗口的最大值](#59-滑动窗口的最大值)
    - [59.2-队列的最大值](#592-队列的最大值)
    - [leetcode 768-最多能完成排序的块 II](#leetcode-768-最多能完成排序的块-ii)
  - [4，字符串](#4字符串)
    - [leetcode 58-最后一个单词的长度](#leetcode-58-最后一个单词的长度)
    - [leetcode 557-反转字符串中的单词 III](#leetcode-557-反转字符串中的单词-iii)
    - [leetcode 1805-字符串中不同整数的数目](#leetcode-1805-字符串中不同整数的数目)
    - [leetcode 1816-截断句子](#leetcode-1816-截断句子)
    - [leetcode 394-字符串解码](#leetcode-394-字符串解码)
    - [leetcode 821-字符的最短距离](#leetcode-821-字符的最短距离)
  - [5，哈希表](#5哈希表)
  - [6，二叉树](#6二叉树)
    - [6.1，Offer 07-重建二叉树](#61offer-07-重建二叉树)
    - [6.2，leetcode 104-二叉树的最大深度](#62leetcode-104-二叉树的最大深度)
    - [55.2-平衡二叉树](#552-平衡二叉树)
    - [leetcode 109-有序链表转换二叉搜索树](#leetcode-109-有序链表转换二叉搜索树)
  - [7，图](#7图)
- [二，算法](#二算法)
  - [1，递归](#1递归)
    - [10-1. 斐波那契数列](#10-1-斐波那契数列)
  - [2，二分查找](#2二分查找)
  - [3，排序](#3排序)
  - [4，贪心](#4贪心)
    - [63-股票的最大利润](#63-股票的最大利润)
  - [5，分治](#5分治)
  - [6，回溯](#6回溯)
  - [7，动态规划](#7动态规划)
    - [10.2-青蛙跳台阶问题](#102-青蛙跳台阶问题)
    - [42-连续子数组的最大和](#42-连续子数组的最大和)
    - [47-礼物的最大价值](#47-礼物的最大价值)
    - [48-最长不含重复字符的子字符串](#48-最长不含重复字符的子字符串)
    - [66-构建乘积数组](#66-构建乘积数组)

## 一，常见数据结构

### 1，数组

#### 3-找出数组中重复的数字

[剑指 Offer 03. 数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

**解题方法**：

1. 直接排序，然后遍历，思路很简单但是执行起来比较麻烦
2. 哈希表，就是找另一个数组，把nums的元素一个一个放进去，放进去之前判断里面有没有，如果里面已经有了那就遇到重复元素，结束。
3. 原地置换。思路是重头扫描数组，遇到下标为 i 的数字如果不是 i 的话（假设为m), 那么我们就拿与下标 m 的数字交换。在交换过程中，如果有重复的数字发生，那么终止返回 ture。

**C++代码**：

```cpp
class Solution {
private:
    void swap(int &a, int &b)
    {
        int temp = a;
        a = b;
        b = temp;
    }
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
```

#### 4-二维数组中的查找

[剑指offer 04. 二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

示例:现有矩阵 matrix 如下：

```c
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```

- 给定 target = 5，返回 true。
- 给定 target = 20，返回 false。

`c++` 代码如下：

```cpp
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
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
};
```

#### 5-替换空格

[剑指 offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof)

请实现一个函数，把字符串 s 中的每个空格替换成"%20"。

示例 1：
```shell
输入：s = "We are happy."
输出："We%20are%20happy."
```

限制：`0 <= s 的长度 <= 10000`

**解题方法**：
题解：双指针法： p2 指针指向扩容之后的 string 最后一位，p1 指向原指针最后一位，遍历指针，如果 p1 遇到空格，就将 p2 向前移动三次并赋值为'%20'，没有，则将 p1 字符赋值给 p2 字符。

**C++代码**：

```cpp

class Solution {
public:
    string replaceSpace(string s) {
        int count = 0, len = s.size();
        for(char& c:s){
            if(c == ' ') count++;
        }
        s.resize(len + 2*count);
        cout << count;
        for(int i = len-1, j=s.size()-1; i<j; i--,j--){
            if(s[i] == ' '){
                cout << s[i];
                s[j] = '0';
                s[j-1] = '2';
                s[j-2] = '%';
                j -= 2;
            }
            else{
                s[j] = s[i];
            }
        }
        return s;
    }
};
```

#### 29-顺时针打印矩阵

[剑指 Offer 29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof)

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

示例 1：
```shell
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
```

示例 2：
```shell
输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
```

限制：
- `0 <= matrix.length <= 100`
- `0 <= matrix[i].length <= 100`

**解题方法**：

从左到右，从上到下，检查一次是否遍历完，从右到左，从下到上

**C++代码**：

```cpp
class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        // left to right, top to bottom
        if(matrix.empty()) return {};
        vector<int> res;
        int l = 0, r = matrix[0].size()-1, t = 0, b = matrix.size()-1;
        int nums = (r+1) * (b+1);
        while(res.size() != nums){
            for(int i=l; i<=r; i++)  // 从左往右遍历：行索引不变，列索引增加
                res.push_back(matrix[t][i]);
            t++;
            for(int j=t; j<=b; j++)  // 从上到下遍历:列索引不变，行索引增加
                res.push_back(matrix[j][r]);
            r--;
            // 检查一次是否遍历完
            if(res.size() == nums)  break;
            for(int m=r; m>=l; m--)  // 从右往左遍历：行索引不变，列索引减少
                res.push_back(matrix[b][m]);
            b--;
            for(int n=b; n>=t; n--)  // 从下往上遍历：列索引不变，行索引减少
                res.push_back(matrix[n][l]);
            l++;            
        } 
        return res;
    }
};
```

#### 50-第一个只出现一次的字符位置

[剑指offer 题50. 第一个只出现一次的字符位置](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof)

在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

示例 1:
```shell
输入：s = "abaccdeff"
输出：'b'
```

示例 2:
```shell
输入：s = "" 
输出：' '
```

限制：`0 <= s 的长度 <= 50000`

**解题方法**：

哈希表法。map：基于红黑树，元素有序存储; unordered_map：基于散列表，元素无序存储

**C++代码**：

```cpp
class Solution {
public:
    char firstUniqChar(string s) {
        unordered_map<char, bool> dic;
        for(char c:s){
            dic[c] = dic.find(c) == dic.end();
        }
        for(char c:s){
            if(dic[c] == true)
                return c;
        }
        return ' ';
    }
    
    char FirstNotRepeatingChar(string s) {
        unordered_map<char, bool> dic;
        for(char c:s){
            dic[c] = dic.find(c) == dic.end();
        }
        for(int i=0; i<s.size();i++){
            if(dic[s[i]] == true)
                return i;
        }
        return -1;
    }
};
```

#### leetcode 989-数组形式的整数加法

[leetcode 989-数组形式的整数加法](https://leetcode-cn.com/problems/add-to-array-form-of-integer)

对于非负整数 X 而言，X 的数组形式是每位数字按从左到右的顺序形成的数组。例如，如果 X = 1231，那么其数组形式为 [1,2,3,1]。

给定非负整数 X 的数组形式 A，返回整数 X+K 的数组形式。

示例 1：
```shell
输入：A = [1,2,0,0], K = 34
输出：[1,2,3,4]
解释：1200 + 34 = 1234
```
示例 2：
```shell
输入：A = [2,7,4], K = 181
输出：[4,5,5]
解释：274 + 181 = 455
```

限制：

1. 1 <= A.length <= 10000
2. 0 <= A[i] <= 9
3. 0 <= K <= 10000
4. 如果 A.length > 1，那么 A[0] != 0

**解题方法**：

两数相加形式的题目，可用以下加法公式[模板](https://leetcode-cn.com/problems/add-to-array-form-of-integer/solution/989-ji-zhu-zhe-ge-jia-fa-mo-ban-miao-sha-8y9r/)。

```shell
当前位 = (A 的当前位 + B 的当前位 + 进位carry) % 10

while ( A 没完 || B 没完)
    A 的当前位
    B 的当前位

    和 = A 的当前位 + B 的当前位 + 进位carry

    当前位 = 和 % 10;
    进位 = 和 / 10;

判断是否还有进位
```

复杂度分析：
- 时间复杂度: $O(n)$
- 空间复杂度: $O(n)$

**C++代码**：

```cpp

class Solution {
public: // 逐位相加法，使用加法模板
    vector<int> addToArrayForm(vector<int>& num, int k) {
        int sum = 0;
        int carry = 0;
        int n = num.size()-1;
        vector<int> res;
        while(n >=0 || k != 0){
            int remainder = k % 10; // k 的当前位
            if(n>=0) sum = num[n] + remainder + carry;
            else sum = remainder + carry;
            carry = sum / 10;  // 进位计算
            sum %= 10;  // 当前位计算
            res.push_back(sum);
            k /= 10;
            n -= 1;
        }
        if(carry != 0) res.push_back(carry);  // 判断是否还有进位

        reverse(res.begin(), res.end());  // 反转数组
        return res;
    }
};
```

### 2，链表

#### 6-从尾到头打印单链表

[剑指 offer 6: 从尾到头打印单链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

示例 1：
```shell
输入：head = [1,3,2]
输出：[2,3,1]
```

限制：`0 <= 链表长度 <= 10000`

**解题方法**：

使用栈的思想(Python 用 list 模拟栈, pop 弹出栈头元素)，栈具有后进先出的特点，在遍历链表时将值按顺序放入栈中，最后出栈的顺序即为逆序。注意和 C 语言不同，C++ 的结构体可以有构造函数！

**C++代码**：

```cpp
class Solution{
public:
    vector<int> reversePrint(ListNode* head) {
        stack<int> values;  // 创建一个不包含任何元素的 stack 适配器，并采用默认的 deque 基础容器：
        vector<int> result;
        while (head != nullptr){
            values.push(head->val);
            head = head->next;
        }
        while(!values.empty()){
            result.push_back(values.top());
            values.pop();
        }
        return result;
    }
};
```

#### 18.1-删除链表的节点

[剑指 Offer 18.1 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。返回删除后的链表的头节点。（注意：此题对比原题有改动）

示例 1:
```shell
输入: head = [4,5,1,9], val = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
```
示例 2:
```shell
输入: head = [4,5,1,9], val = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
```

说明：
- 题目保证链表中节点的值互不相同
- 若使用 C 或 C++ 语言，你不需要 free 或 delete 被删除的节点

**解题思路**：

1. 定位节点： 遍历链表，直到 head.val == val 时跳出，即可定位目标节点。
2. 修改引用： 设节点 cur 的前驱节点为 pre ，后继节点为 cur.next ；则执行 pre.next = cur.next ，即可实现删除 cur 节点。

**C++代码**：

```cpp
class Solution {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        if(head->val == val) return head -> next;
        ListNode* pre = head; ListNode* cur = head->next;
        while(cur != nullptr && cur->val != val) {
            pre = cur; 
            cur = cur->next;
        }
        pre->next = cur->next;
        return head;
    }
};
```

#### 18.2 删除链表中重复的节点

[剑指 Offer 18.2  删除链表中重复的节点]()

题目描述：在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

**解题方法：**

迭代解法

**C++代码**：

```cpp
class Solution {
public:
    ListNode* deleteDuplication(ListNode* head) {
        ListNode *vhead = new ListNode(-1);
        vhead->next = head;
        ListNode* pre = vhead,*cur = head;
        while(cur){
            if(cur->next && cur->val==cur->next->val){
                cur = cur->next;
                while(cur->next && cur->val == cur->next->val){
                    cur = cur->next;
                }
                cur = cur -> next;
                pre->next = cur;
            }
            else{
                pre = cur;
                cur = cur->next;
            }
        }
        return vhead->next;
    }
};
```

#### 22-链表中倒数第k个节点

[剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof)

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。

示例：

```shell
给定一个链表: 1->2->3->4->5, 和 k = 2.
返回链表 4->5.
```

**解题方法：**

双指针法。不用统计链表长度。前指针 former 先向前走 k 步。

**C++代码**：

```cpp

// Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode* former = head;
        ListNode* latter = head;
        for(int i=0;i<k;i++){
            former = former->next;
        }
        while(former != NULL){
            former = former->next;
            latter = latter->next;
        }
        return latter;
    }
};
```

#### 23-链表中环的入口结点

[链表中环的入口结点](https://www.nowcoder.com/practice/253d2c59ec3e4bc68da16833f79a38e4?tpId=13&tqId=11208&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking&from=cyc_github)

给一个长度为 n 的链表，若其中包含环，请找出该链表的环的入口结点，否则，返回 null。

- 输入描述：输入分为2段，第一段是入环前的链表部分，第二段是链表环的部分，后台将这2个会组装成一个有环或者无环单链表
- 返回值描述：返回链表的环的入口结点即可。而我们后台程序会打印这个节点

示例1：

```shell
输入：{1,2},{3,4,5}
返回值：3
说明：返回环形链表入口节点，我们后台会打印该环形链表入口节点，即3   
```

**解题思路：**

**采用双指针解法**，一快一慢指针。快指针每次跑两个element，慢指针每次跑一个。如果存在一个圈，总有一天，快指针是能追上慢指针的。

**C++代码**：

```cpp
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead) {
        ListNode* fast = pHead;
        ListNode* slow = pHead;
        
        while( fast && fast->next) {  // 找到 fast 指针和 slow 指针相遇位置
            fast = fast->next->next;
            slow = slow->next;
            if(fast == slow ) break;
        }
        if (!fast || !fast->next) return nullptr;
        fast = pHead; // fast 指针指向头节点，slow 指针原地不变
        while(fast != slow ) {  // 两个指针重新相遇于环的入口点
            fast = fast->next;
            slow = slow->next;
        }
        return fast;
    }
};
```

#### 24-反转一个单链表

[剑指 offer 24: 反转一个单链表](https://leetcode-cn.com/problems/reverse-linked-list/)

给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

示例1：
```shell
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
```

**解题思路：**

双指针迭代法。

**C++代码**：

```cpp
class Solution {
public: // 双指针迭代法
    ListNode* reverseList(ListNode* head) {
         // 判断链表为空或长度为1的情况
        if(head == nullptr || head->next == nullptr){
            return head;
        }
        ListNode* pre = nullptr; // 当前节点的前一个节点
        ListNode* next = nullptr; // 当前节点的下一个节点
        while( head != nullptr){
            next = head->next; // 记录当前节点的下一个节点位置；
            head->next = pre; // 让当前节点指向前一个节点位置，完成反转
            pre = head; // pre 往右走
            head = next;// 当前节点往右继续走
        }
        return pre;
    }
};
```

#### 52-两个链表的第一个公共节点

[剑指 offer 52: 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

输入两个链表，找出它们的第一个公共节点。

**解题方法**：

1，**双指针法**：设节点指针 A 指向头节点 headA, 节点指针 B 指向头节点 headB。

1. A 先遍历完链表 headA，然后遍历 headB;
2. B 先遍历完链表 headB，然后遍历 headA;

只要有公共节点，总路程数，或者说 A 经过的节点数和 B 经过的节点数是一样的，
如果没有公共节点，只有当 A 和 B都变成了 nullptr的时候，两者最终走的路程才是一样的。 
然后只需比较 A和 B是否相等，相等的那个位置即为公共节点，因为此使，两者走的步数开始相等了。

2，**栈特性解法**。两个链表从公共结点开始后面都是一样的，顺着链表从后向前查找，很容易就能查找到链表的公共结点（第一个不相同的结点的下一个结点即所求）；而从后向前的特性自然联想到栈。

3，**哈希表法**。

复杂度分析：

- 时间复杂度：O(n)
- 空间复杂度：O(1)

**C++代码**：

```cpp
class Solution {
public:
    // 双指针法
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode* A =  headA;
        ListNode* B = headB;
        while(A != B){
            if(A != nullptr) A = A->next;
            else A = headB;
            if (B != nullptr) B = B->next;
            else B = headA;
        }
        return A;
    }
    // 哈希表法，哈希表中存储链表节点指针
    ListNode *getIntersectionNode2(ListNode *headA, ListNode *headB) {
        unordered_set<ListNode *> visited;
        ListNode* temp = headA;
        while(temp != nullptr){
            visited.insert(temp);
            temp = temp -> next;
        }
        temp = headB;
        while(temp != nullptr){
            // count 方法判断哈希表中是否存在 temp 关键字
            if(visited.count(temp)) return temp;
            else temp = temp -> next;
        }
        return nullptr;
    }
    // vector 法，vector 中元素为链表节点指针
    ListNode *getIntersectionNode3(ListNode *headA, ListNode *headB) {
        vector<ListNode *> visited;
        ListNode* temp = headA;
        while(temp != nullptr){
            visited.push_back(temp);
            temp = temp -> next;
        }
        temp = headB;
        while(temp != nullptr){
            // find 函数查找 vector 中是否存在 temp 元素
            if(visited.end() != find(visited.begin(), visited.end(), temp)) return temp;
            else temp = temp -> next;
        }
        return nullptr;
    }
    // 栈特性解法
    ListNode *getIntersectionNode4(ListNode *headA, ListNode *headB) {
        ListNode *l1 = headA;
        ListNode *l2 = headB;
        stack<ListNode* > st1, st2;
        while(headA != nullptr){
            st1.push(headA);
            headA = headA->next;
        }
        while(headB != nullptr){
            st2.push(headB);
            headB = headB->next;
        }
        ListNode* ans = nullptr;
        while(!st1.empty()&&!st2.empty()&&st1.top()==st2.top()){
            ans = st1.top();
            st1.pop();
            st2.pop();
        }
        return ans;
    } 
};
```

#### leetcode 61-旋转链表

[leetcode 61-旋转链表](https://leetcode-cn.com/problems/rotate-list/)

给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置。

示例1

```shell
输入：head = [1,2,3,4,5], k = 2
输出：[4,5,1,2,3]
```

**解题思路：**

将原来的链表首尾相连变成环，然后找倒数第 k 个点作为新的表头，即原来的表头向右移动 (n-1)-(k%n) 次后断开。

**C++代码**：

```cpp
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if (k == 0 || head == nullptr || head->next == nullptr) {
            return head;
        }
        ListNode* cur = head;
        int n = 1;
        while(cur -> next != nullptr){
            cur = cur -> next;
            n += 1;
        } 
        cur -> next = head; // 将链表首尾相连变成环
        cur = head;
        int move = (n-1)-(k % n);
        while(move--){
            cur = cur -> next;
        }
        ListNode* ret = cur -> next;
        cur -> next = nullptr; // cur 向右移动 move 次后，断掉连接
        return ret;
    }
};
```
#### leetcode 24-两两交换链表中的节点

[leetcode 24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

**解题思路：**

1，迭代法：关键是高清如何交换两个相邻节点，然后迭代交换即可。

复杂度分析：

- 时间复杂度：O(n)
- 空间复杂度：O(1)

**C++代码**：

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if(head == nullptr) return nullptr;
        else if(head->next == nullptr) return head;
        ListNode* temp = new ListNode(-1);
        temp ->next = head;
        ListNode* pre = temp;
        while(pre->next != nullptr && pre->next->next != nullptr) {

            ListNode* cur = pre->next;
            ListNode* next = pre->next->next;

            pre->next = cur->next;
            cur->next = next->next;
            next->next = cur;
            pre = cur;
        }
        return temp->next;
    }
};
```

### 3，栈队列堆

#### 9-用两个栈实现队列

[剑指 offer 面试题9](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

用两个栈实现队列, 用两个栈来实现一个队列，完成队列的 Push 和 Pop 操作。

**解题思路**：

- in 栈用来处理入栈（push）操作，out 栈用来处理出栈（pop）操作。一个元素进入 in 栈之后，出栈的顺序被反转。
- 当元素要出栈时，需要先进入 out 栈，此时元素出栈顺序再一次被反转，因此出栈顺序就和最开始入栈顺序是相同的，先进入的元素先退出，这就是队列的顺序。

**C++代码**：

```cpp
class Solution
{
public:
    void push(int node) {
        stack1.push(node);
    }
    int pop() {
        int res;
        if(stack2.empty()){
            while(!stack1.empty()){
                int temp = stack1.top();
                stack1.pop();
                stack2.push(temp);
            }
        }
        res = stack2.top();
        stack2.pop();
        return res;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};
```

#### 30-包含 min 函数的栈

[30-包含 min 函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

**解题思路：**

- 数据栈 A ： 栈 A 用于存储所有元素，保证入栈 push() 函数、出栈 pop() 函数、获取栈顶 top() 函数的正常逻辑。
- 辅助栈 B ： 栈 B 中存储栈 A 中所有 非严格降序 的元素，则栈 A 中的最小元素始终对应栈 B 的栈顶元素，即 min() 函数只需返回栈 B 的栈顶元素即可。

**C++代码**：

```cpp
class MinStack {  // 利用辅助栈
private:
    stack<int> stack1;
    stack<int> stack2;
public:
    /** initialize your data structure here. */
    MinStack() {

    }
    void push(int x) {
        stack1.push(x);
        if (stack2.empty()) {
            stack2.push(x);
        }
        else {
            if (x < stack2.top()) {
                stack2.push(x);
            }
            else {
                stack2.push(stack2.top());
            }
        }
    }
    void pop() {
        stack1.pop();
        stack2.pop();
    }

    int top() {
        return stack1.top();
    }

    int min() {
        return stack2.top();
    }
};
```

#### 31-栈的压入、弹出序列

[31-栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof)

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 `{1,2,3,4,5}` 是某栈的压栈序列，序列 `{4,5,3,2,1}` 是该压栈序列对应的一个弹出序列，但 `{4,3,5,1,2}` 就不可能是该压栈序列的弹出序列。

示例 1：
```shell
输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
```

示例 2：
```shell
输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。
```

提示：
1. 0 <= pushed.length == popped.length <= 1000
2. 0 <= pushed[i], popped[i] < 1000
3. pushed 是 popped 的排列。

**解题思路：**

**C++代码实现**：

```cpp
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
```

#### 40-最小的 K 个数

[40-最小的 K 个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入 `4、5、1、6、2、7、3、8` 这 `8` 个数字，则最小的 `4` 个数字是 `1、2、3、4`。

示例 1：
```shell
输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
```

示例 2：
```shell
输入：arr = [0,1,2,1], k = 1
输出：[0]
```

限制：
- 0 <= k <= arr.length <= 10000
- 0 <= arr[i] <= 10000

**解题方法：**

1. 数组原地排序法：对原数组从小到大排序后取出前 k 个数即可。时间复杂度：O(nlog n)，空间复杂度：O(log n)。
2. 使用最大堆结构：优先队列(最大堆，优先输出最大数)。时间复杂度：O(nlongk), 插入容量为k的大根堆时间复杂度为O(longk), 一共遍历n个元素；空间复杂度：O(k)。
3. 快速排序算法：TODO.

**C++代码**：

```cpp
// priority_queue<Type, Container, Functional>  // 默认定义最大堆
// priority_queue<int, vector<int>, greater<int> >p;  // 定义最小堆

class Solution {
public:
    // // stl 自带的 sort() 排序算法
    vector<int> getLeastNumbers1(vector<int>& arr, int k) {
        vector<int> ret(k, 0);
        sort(arr.begin(), arr.end());
        for(int i=0;i<k;++i){
            ret[i] = arr[i];
        }
        return ret;
    }
    // 大顶堆维护小顶堆的方法
    vector<int> getLeastNumbers2(vector<int>& arr, int k) {
        vector<int> vec(k,0);
        if(k==0)
            return vec;
        priority_queue<int> heap;  // 大顶堆，堆顶为最大值
        for(int i=0;i<(int)arr.size();i++){
            if(i<k){
                heap.push(arr[i]);
            }
            else{
                if(heap.top() > arr[i]){  // 使用大顶堆来维护最小堆
                    heap.pop();
                    heap.push(arr[i]);
                }
            }
        }
        for(int i=0;i<k;i++){
            vec[i] = heap.top();
            heap.pop();
        }
        return vec;
    }
    // 直接使用小顶堆
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        vector<int> vec(k,0);
        if(k==0)
            return vec;
        priority_queue<int, vector<int>, greater<int> > heap; // 小顶堆，堆顶为最小值
        priority_queue<int> heap2; // 大顶堆，堆顶为最大值
        for(int i=0;i<(int)arr.size();i++){
            heap.push(arr[i]);
        }
        for(int i=0;i<k;i++){
            vec[i] = heap.top();
            heap.pop();
        }
        return vec;
    }
};
```

#### 41.1-数据流中的中位数

[41.1-数据流中的中位数](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof)

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。例如，[2,3,4] 的中位数是 3；[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：
- `void addNum(int num)` - 从数据流中添加一个整数到数据结构中。
- `double findMedian()` - 返回目前所有元素的中位数。

**解题方法**：

数据流左半边的数用大顶堆，右半边的数用小顶堆，中位数由两个堆的堆顶元素求得。

**C++代码**：
```cpp
class MedianFinder {  // 大根堆+小根堆 解法，时间超过 99..38%，空间超过 18.07%
private:
    // 从左到右，数据依次从大到小
    priority_queue<int> right; // 大顶堆，堆顶为最大值
    priority_queue<int, vector<int>, greater<int> > left; // 小顶堆，堆顶为最小值

public:
    /** initialize your data structure here. */
    MedianFinder() {

    }
    void addNum(int num) {
        // 插入数据要始终保持两个堆处于平衡状态，即较大数在左边，较小数在右边
        // 两个堆元素个数不超过 1
        if(left.size() == right.size()){
            right.push(num);
            left.push(right.top());  // 保证左边堆插入的元素始终是右边堆的最大值
            right.pop();  // 删除堆顶元素
        }
        else{
            left.push(num);
            right.push(left.top());
            left.pop();
        }
    }
    double findMedian() {
        if(left.size() == right.size()) return (left.top() + right.top())*0.5;
        else return left.top()*1.0;
    }
};
```

#### [41.2-字符流中第一个不重复的字符]()

请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

**解题思路：**

对于“重复问题”，惯性思维应该想到哈希或者set。对于“字符串问题”，大多会用到哈希。因此一结合，应该可以想到，判断一个字符是否重复，可以选择用哈希，在 `c++` 中，可以选择用 `unordered_map<char, int>`。

```cpp
class Solution
{
public:
  //Insert one char from stringstream
    queue<char> q;
    unordered_map<char, int> mp;
    void Insert(char ch)
    {
         // 如果是第一次出现，则添加到队列中
         if (mp.find(ch) == mp.end()) {
             q.push(ch);
         }
         // 不管是不是第一次出现，都进行计数
         ++mp[ch];
    }
    // return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        while (!q.empty()) {
            char ch = q.front();
            // 拿出头部，如果是第一次出现，则返回
            if (mp[ch] == 1) {
                return ch;
            }
            // 不是第一次出现，则弹出，然后继续判断下一个头部
            else {
                q.pop();
            }
        }
        return '#';
    }
};
```

#### 59-滑动窗口的最大值

[剑指offer 59-滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

给定一个数组 `nums` 和滑动窗口的大小 `k`，请找出所有滑动窗口里的最大值。

示例:

```shell
输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7] 
解释: 

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

**解题方法：**

1. 对于每个滑动窗口，可以使用 O(k) 的时间遍历其中的每一个元素，找出其中的最大值。对于长度为 n 的数组 nums 而言，窗口的数量为 n-k+1，算法的时间复杂度为 $O((n-k+1)\ast k)=O(n\ast k)$。
2. 维护单调递减的双端队列！如果发现队尾元素小于要加入的元素，则将队尾元素出队，直到队尾元素大于新元素时，再让新元素入队，从而维护一个单调递减的队列。

**C++代码**：

```cpp
class Solution {  
public:
    // 简单方法：遍历滑动窗口找最大值，合理选择区间
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ret;
        if (nums.size() == 0 && k == 0) return ret;
        for (int i = 0; i <= nums.size() - k; i++) {
            int maxNum = nums[i];
            for (int j = i; j < (i + k); j++) {
                if (nums[j] > maxNum)
                    maxNum = nums[j];
            }
            ret.push_back(maxNum);
        }
        return ret;
    }
    // 维护一个单调队列，队头是最大值
    vector<int> maxSlidingWindow2(vector<int>& nums, int k) {
        vector<int> ret;
        deque<int> window;  // 创建双端队列
        // 先将第一个窗口的值按照规则入队
        for (int i = 0; i < k; i++) {
            while (!window.empty() && window.back() < nums[i]) {
                window.pop_back();
            }
            window.push_back(nums[i]);
        }
        ret.push_back(window.front());
        
        for (int j = k; j < nums.size(); j++) {
            if (nums[j - k] == window.front()) window.pop_front();  // 模拟滑动窗口的移动
            while (!window.empty() && window.back() <= nums[j]) {
                window.pop_back();
            }
            window.push_back(nums[j]);
            ret.push_back(window.front());
        }
        return ret;
    }
};
```

#### 59.2-队列的最大值

[剑指offer 59.2-队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof)

请定义一个队列并实现函数 `max_value` 得到队列里的最大值，要求函数 `max_value`、`push_back` 和 `pop_front` 的均摊时间复杂度都是 $O(1)$。

若队列为空，`pop_front` 和 `max_value` 需要返回 `-1`。

示例 1：

```shell
输入: 
["MaxQueue","push_back","push_back","max_value","pop_front","max_value"]
[[],[1],[2],[],[],[]]
输出: [null,null,null,2,1,2]
```

**解题思路：**

定义一个单调递减的辅助队列（双端队列）

**C++代码实现**：

```cpp
class MaxQueue {
private:
    queue<int> que1;
    deque<int> que2;  // 辅助队列，头部位置存放最大值值
public:
    MaxQueue() {

    }
    int max_value() {
        if(que1.empty())
            return -1;
        return que2.front();
    }
    void push_back(int value) {
        // 维护单调递减队列
        while(!que2.empty() && que2.back() < value){
            que2.pop_back();  // 移除队尾元素直到队尾元素大于新添加元素
        }
        
        que2.push_back(value);
        que1.push(value);
    }
    int pop_front() {
        if(que1.empty()) return -1;
        else{
            int ans = que1.front();
            if( ans == que2.front()) que2.pop_front();
            que1.pop();
            return ans;
        }
    }
};

/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue* obj = new MaxQueue();
 * int param_1 = obj->max_value();
 * obj->push_back(value);
 * int param_3 = obj->pop_front();
 */
```

#### leetcode 768-最多能完成排序的块 II

[leetcode 768-最多能完成排序的块 II](https://leetcode-cn.com/problems/max-chunks-to-make-sorted-ii/?utm_source=LCUS&utm_medium=ip_redirect&utm_campaign=transfer2china)

这个问题和“最多能完成排序的块”相似，但给定数组中的元素可以重复，输入数组最大长度为 2000，其中的元素最大为 10**8。

`arr` 是一个可能包含重复元素的整数数组，我们将这个数组分割成几个“块”，并将这些块分别进行排序。之后再连接起来，使得连接的结果和按升序排序后的原数组相同。我们最多能将数组分成多少块？

示例 1:
```shell
输入: arr = [5,4,3,2,1]
输出: 1
解释:
将数组分成2块或者更多块，都无法得到所需的结果。
例如，分成 [5, 4], [3, 2, 1] 的结果是 [4, 5, 1, 2, 3]，这不是有序的数组。
```
**解题思路：**

1，辅助栈法：栈中存放每个块内元素的最大值，栈的 size() 即为最多分块数。

题中隐含结论：
- 下一个分块中的所有数字都会大于等于上一个分块中的所有数字，即后面块中的最小值也大于前面块中最大值。
- 只有分的块内部可以排序，块与块之间的相对位置是不能变的。
- 直观上就是找到从左到右开始不减少（增加或者不变）的地方并分块。
- 要后面有较小值，那么前面大于它的都应该在一个块里面。

复杂度分析：
- 时间复杂度: O(n)
- 空间复杂度: O(1)

**C++代码**：

```cpp
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
```

### 4，字符串

#### leetcode 58-最后一个单词的长度

[leetcode 58-最后一个单词的长度](https://leetcode-cn.com/problems/length-of-last-word/solution/leetcode58ti-jie-ti-si-lu-by-litao/)

给你一个字符串 s，由若干单词组成，单词前后用一些空格字符隔开。返回字符串中最后一个单词的长度。单词：是指仅由字母组成、不包含任何空格字符的最大子字符串。

**C++代码**：

```cpp
class Solution {
public:
    int lengthOfLastWord(string s) {
        s +=  ' ';
        vector<string> res; // 存放字符串的数组
        string temp; // 临时字符串
        for(char ch:s){
            if(ch == ' '){
                if(!temp.empty()){
                    res.push_back(temp);
                    temp.clear();
                }
            }
            else{
                temp += ch;
            }
        }
        string last_word = res.back();  // 数组最后一个元素
        return last_word.size();
    }
};
```

#### leetcode 557-反转字符串中的单词 III

[leetcode 557. 反转字符串中的单词 III](https://leetcode-cn.com/problems/reverse-words-in-a-string-iii)

给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。

示例：
```shell
输入："Let's take LeetCode contest"
输出："s'teL ekat edoCteeL tsetnoc"
```

**提示**：在字符串中，每个单词由单个空格分隔，并且字符串中不会有任何额外的空格。

**C++代码**：

```cpp
class Solution {
public:
    string reverseWords(string s) {
        s +=  ' ';
        vector<string> res; // 存放字符串的数组
        string temp; // 临时字符串
        for(char ch:s){
            if(ch == ' '){
                if(!temp.empty()){
                    res.push_back(temp);
                    temp.clear();
                }
            }
            else{
                temp += ch;
            }
        }
        s.clear();
        for(string &str: res){
            reverse(str.begin(), str.end());
            s += str + ' ';
        }
        s.pop_back();
        return s;
    }
};
```

#### leetcode 1805-字符串中不同整数的数目

[leetcode 1805-字符串中不同整数的数目](ttps://leetcode-cn.com/problems/number-of-different-integers-in-a-string/)

给你一个字符串 `word` ，该字符串由数字和小写英文字母组成。

请你用空格替换每个不是数字的字符。例如，"a123bc34d8ef34" 将会变成 " 123  34 8  34" 。注意，剩下的这些整数为（相邻彼此至少有一个空格隔开）："123"、"34"、"8" 和 "34" 。

返回对 word 完成替换后形成的 不同 整数的数目。只有当两个整数的 不含前导零 的十进制表示不同， 才认为这两个整数也不同。

示例 1：
```shell
输入：word = "a123bc34d8ef34"
输出：3
解释：不同的整数有 "123"、"34" 和 "8" 。注意，"34" 只计数一次。
```

**C++代码**：

```cpp

class Solution {
public:
    int numDifferentIntegers(string word) {
        set<string> s;
        word += 'a';
        string temp; // 临时字符串
        for(char ch:word){
            // 如果遇到字母且临时字符串非空，就把它加入集合并重置临时字符串
            if(isalpha(ch)){   
                if(!temp.empty()){
                    s.insert(temp);
                    temp.clear();
                }
            }
            else{
                if(temp == "0") temp.clear();  // "001" 和 "1" 是等值的
                temp += ch;
            }
        }
        return s.size();
    }
};
```

#### leetcode 1816-截断句子

[leetcode 1816-截断句子](https://leetcode-cn.com/problems/truncate-sentence)

句子 是一个单词列表，列表中的单词之间用单个空格隔开，且不存在前导或尾随空格。每个单词仅由大小写英文字母组成（不含标点符号）。

例如，"Hello World"、"HELLO" 和 "hello world hello world" 都是句子。给你一个句子 s​​​​​​ 和一个整数 k​​​​​​ ，请你将 s​​ 截断 ​，​​​使截断后的句子仅含 前 k​​​​​​ 个单词。返回 截断 s​​​​​​ 后得到的句子。

示例 1：
```shell
输入：s = "Hello how are you Contestant", k = 4
输出："Hello how are you"
解释：
s 中的单词为 ["Hello", "how" "are", "you", "Contestant"]
前 4 个单词为 ["Hello", "how", "are", "you"]
因此，应当返回 "Hello how are you"
```

**C++代码**：

```cpp

class Solution {
public:
    string truncateSentence(string s, int k) {
        s +=  ' ';
        vector<string> res; // 存放字符串的数组
        string temp; // 临时字符串
        for(char ch:s){
            if(ch == ' '){
                res.push_back(temp);
                temp.clear();
            }
            else{
                temp += ch;
            }
        }
        s.clear();
        for(int i=0; i< k;i++){
            s += res[i] + ' ';
        }
        s.pop_back();
        return s;
    }
};
```

#### leetcode 394-字符串解码

[leetcode 394. 字符串解码](https://leetcode-cn.com/problems/decode-string/solution/decode-string-fu-zhu-zhan-fa-di-gui-fa-by-jyd/)

给定一个经过编码的字符串，返回它解码后的字符串。编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。

示例 1：
```shell
输入：s = "3[a]2[bc]"
输出："aaabcbc"
```

**解题思路**：

本题难点在于括号内嵌套括号，需要从内向外生成与拼接字符串，这与栈的先入后出特性对应。

复杂度分析：

- 时间复杂度 O(N)： s；
- 空间复杂度 O(N)：辅助栈在极端情况下需要线性空间，例如 2[2[2[a]]]。

**C++代码**：

```cpp
class Solution {
public:
    string decodeString(string s) {
        stack<pair<string, int>> s1;
        string res = "";
        int num = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] >= '0' && s[i] <= '9') {
                num *= 10;
                num += (s[i] - '0');
            }
            else if (s[i] == '[') {
                s1.push(make_pair(res, num));
                num = 0;
                res = "";
            }
            else if (s[i] == ']') {
                auto cur_num = s1.top().second;
                auto latest_res = s1.top().first;
                s1.pop();
                for (int j = 0; j < cur_num; j++) latest_res = latest_res + res; // res 加 n 次
                res = latest_res;
            }
            else {
                res += s[i];
            }
        }
        return res;
    }
};
```

#### leetcode 821-字符的最短距离

[leetcode 821-字符的最短距离](https://leetcode-cn.com/problems/shortest-distance-to-a-character/)

给你一个字符串 `s` 和一个字符 `c` ，且 `c` 是 `s` 中出现过的字符。

返回一个整数数组 answer ，其中 answer.length == s.length 且 answer[i] 是 s 中从下标 i 到离它 最近 的字符 c 的 距离 。

两个下标 i 和 j 之间的 距离 为 abs(i - j) ，其中 abs 是绝对值函数。

示例 1：
```shell
输入：s = "loveleetcode", c = "e"
输出：[3,2,1,0,1,0,0,1,2,2,1,0]
解释：字符 'e' 出现在下标 3、5、6 和 11 处（下标从 0 开始计数）。
距下标 0 最近的 'e' 出现在下标 3 ，所以距离为 abs(0 - 3) = 3 。
距下标 1 最近的 'e' 出现在下标 3 ，所以距离为 abs(1 - 3) = 2 。
对于下标 4 ，出现在下标 3 和下标 5 处的 'e' 都离它最近，但距离是一样的 abs(4 - 3) == abs(4 - 5) = 1 。
距下标 8 最近的 'e' 出现在下标 6 ，所以距离为 abs(8 - 6) = 2 
```

**解题方法：**

1，两次遍历

- 从左向右遍历，记录上一个字符 C 出现的位置 prev，那么答案就是 i - prev。
- 从右想做遍历，记录上一个字符 C 出现的位置 prev，那么答案就是 prev - i。

2，哈希表法

- 获取 s 中所有目标字符 c 的位置，并提前存储在数组 c_indexs 中。
- 遍历字符串 s 中的每个字符，如果和 c 不相等，就到 c_indexs 中找距离当前位置最近的下标。

复杂度分析：
- 时间复杂度: O(n)
- 空间复杂度: O(1)

**C++代码**：

```cpp
class Solution {
public: // 1，两次遍历法
    vector<int> shortestToChar(string s, char c) {
        vector <int> ret;
        int prev = -10000;
        int distance = 0;
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == c) prev = i;
            distance = i - prev;
            ret.push_back(distance);
        }
        prev = 10000;
        for (int i = s.size() - 1; i >= 0; i--) {
            if (s[i] == c) prev = i;
            distance = prev - i;
            ret[i] = min(ret[i], distance);
        }
        return ret;
    }
    // 解法2：空间换时间，时间复杂度 O(n*k)
    vector<int> shortestToChar2(string s, char c) {
        int n = s.size();
        vector<int> c_indexs;
        // Initialize a vector of size n with default value 0.
        vector<int> ret(n, 0);

        for (int i = 0; i < n; i++) {
            if (s[i] == c) c_indexs.push_back(i);
        }

        for (int i = 0; i < n; i++) {
            int distance = 10000;
            if (s[i] == c) ret[i] = 0;
            else {
                for (int j = 0; j < c_indexs.size(); j++) {
                    int temp = abs(c_indexs[j] - i);
                    if (temp < distance) distance = temp;
                }
                ret[i] = distance;
            }
            
        }
        return ret;
    }
};
```

### 5，哈希表

### 6，二叉树

#### 6.1，Offer 07-重建二叉树

[剑指 Offer 07-重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)

输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

示例1：

```shell
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
```

**解题方法：**

1，递归法
+ 中序遍历的结果可以获取左右子树的元素个数；
+ 前序遍历结果可以获取树的根节点 node 的值。

**C++代码**：
```cpp
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        // 哈希表 dic 存储中序遍历的值与索引的映射
        for(int i=0; i<inorder.size(); ++i){
            index[inorder[i]] = i;
        }
        auto root = recur(preorder, 0, 0, inorder.size());
        return root;
    }

private:
    unordered_map<int,int> index;
    TreeNode* recur(vector<int>& preorder, int root, int left, int right){
        if (left > right) return nullptr;
        int i = index[preorder[left]]; // 获取中序遍历中根节点值的索引
        TreeNode* node = new TreeNode(preorder[left]);
        node->left = recur(preorder, root+1, left, i-1);
        node->right = recur(preorder, root+i-left+1, i+1, right);

        return node;

    }
};
```

#### 6.2，leetcode 104-二叉树的最大深度

[104-二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

给定一个二叉树，找出其最大深度。二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

**说明**: 叶子节点是指没有子节点的节点。

示例：
```shell
给定二叉树 [3,9,20,null,null,15,7]，

    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度 3 。
```

#### 55.2-平衡二叉树

[55.2-平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

给定一个二叉树，判断它是否是高度平衡的二叉树。本题中，一棵高度平衡二叉树定义为：

一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 `1`。

#### leetcode 109-有序链表转换二叉搜索树

[leetcode109. 有序链表转换二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree)

给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。

本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 `1`。

示例:
```shell
给定的有序链表： [-10, -3, 0, 5, 9],

一个可能的答案是：[0, -3, 9, -10, null, 5], 它可以表示下面这个高度平衡二叉搜索树：

      0
     / \
   -3   9
   /   /
 -10  5
```

**解题方法：**

1，将单调递增链表转化为数组，然后分治递归。
2，快慢指针找链表的中间节点，然后递归。

复杂度分析：
- 时间复杂度: O(n)
- 空间复杂度: O(n)

**C++代码**：

```cpp
class Solution {
public:
    // 分治递归
    TreeNode* sortedListToBST(ListNode* head) {
        vector<int> vec;
        for(auto it = head; it!=nullptr ; it=it->next ){
            vec.push_back( it->val );
        }
        return recur(vec, 0, vec.size()-1);
    }

    TreeNode* recur(vector<int> &arr, int left, int right){
        if(left > right) return nullptr;
        int mid = right + (left-right)/2;  // 数组中间位置的索引
        TreeNode* node = new TreeNode(arr[mid]);
        node -> left = recur(arr, left, mid - 1);
        node -> right = recur(arr, mid + 1, right);

        return node;
    }
};
```

### 7，图

## 二，算法

### 1，递归

#### 10-1. 斐波那契数列

[10-1. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

写一个函数，输入 `n`，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：
```shell
F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
```

斐波那契数列由 `0` 和 `1` 开始，之后的斐波那契数就是由之前的两数相加而得出。答案需要取模 `1e9+7（1000000007）`，如计算初始结果为：`1000000008`，请返回 `1`

**解题方法**：

1，记忆化递归
2，迭代法

**C++代码**：
```cpp
// 剑指 offer 10-1. 斐波那契数列
class Solution {
private:
    static const int mod = 1e9 + 7;
    int m = 101;
    vector<int> vec = vector<int>(101, -1);  // c++11 之后，类 private成员初始化方式
public:
    // 1，直接递归会超出时间限制，需要使用记忆化递归
    constexpr int fib(int n) {
        if (n == 0) return 0;
        if (n == 1 || n == 2) return 1;

        if (vec[n] != -1) return vec[n];
        vec[n] = (fib(n - 1) + fib(n - 2)) % mod;

        return vec[n];
    }
    // 2，迭代求解
    int fib(int n) {
        int arr[101];
        arr[0] = 0;
        arr[1] = 1;
        arr[2] = 1;
        for (int i = 2; i < n; i++) {
            arr[i+1] = (arr[i ] + arr[i - 1]) % mod;
        }
        return arr[n];
    }
};
```

### 2，二分查找

### 3，排序

### 4，贪心

#### 63-股票的最大利润

[63-股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)

假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

示例 1:
```shell
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
```

**解题方法：**

1，贪心法：假设每天的股价都是最低价，每天都计算股票卖出去后的利润。一次 for 循环，时间复杂度：O(n)
2，暴力法：两次 for 循环，时间复杂度 O(n^2)

**C++代码**：

```cpp
# include <iostream>
# include <vector>
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
```

### 5，分治

### 6，回溯

### 7，动态规划

#### 10.2-青蛙跳台阶问题

[剑指offer 10.2-青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

**解题方法：**

1，动态规划法：以斐波那契数列性质 $f(n + 1) = f(n) + f(n - 1)$ 为转移方程。

+ 状态定义： 设 $dp$ 为一维数组，其中 $dp[i]$ 的值代表斐波那契数列第 $i$ 个数字 。
+ 转移方程： $dp[i + 1] = dp[i] + dp[i - 1]$ ，即对应数列定义 $f(n + 1) = f(n) + f(n - 1)$；
+ 初始状态： $dp[0] = 1, dp[1] = 1$，即初始化前两个数字；
+ 返回值： $dp[n]$，即斐波那契数列的第 $n$ 个数字。

**C++代码**：

```cpp
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
```

#### 42-连续子数组的最大和

[剑指offer 42-连续子数组的最大和](https://leetcode-cn.com/problems/maximum-subarray/)

给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。 

示例 1：
```shell
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
```

**解题思路**：

动态规划法。

**C++代码**：
```cpp
class Solution {
public:
    //1, 动态规划算法
    int maxSubArray2(vector<int>& nums) {
        int* dp = new int[nums.size()];
        dp[0] = nums[0];
        int maxSum = dp[0];
        for(int i=1; i < nums.size(); i++){
            dp[i] = max(dp[i-1], 0) + nums[i];
            maxSum = max(dp[i], maxSum);
        }
        return maxSum;
    }
    //1, 动态规划，优化空间
    int maxSubArray(vector<int>& nums) {
        int sum = nums[0];
        int maxSum = nums[0];
        for(int i=1; i < nums.size(); i++){
            sum = max(sum, 0) + nums[i];
            maxSum = max(sum, maxSum);
        }
        return maxSum;
    }
};
```

#### 47-礼物的最大价值

[剑指offer 47-礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

在一个 $m\ast n$ 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

示例 1:
```shell
输入: 
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
```

**解题方法：**

动态规划-状态转移方程法。

**C++代码**：

```cpp
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
```

#### 48-最长不含重复字符的子字符串

[剑指offer 42-最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

示例 1:
```shell
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**解题思路**：

参考[这里](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/solution/mian-shi-ti-48-zui-chang-bu-han-zhong-fu-zi-fu-d-9/)

**C++代码**：

```cpp
class Solution {
public:
    // 动态规划+线性遍历
    int lengthOfLongestSubstring(string s) {
        int res=0, tmp = 0, i=0;
        for(int j=0; j < s.size(); j++){
            i = j-1;
            while(i>=0 && s[i] != s[j]) i-= 1;
            if(tmp < j-i) tmp += 1;
            else tmp = j - i;
            res = max(res, tmp);
        }
        return res;
    }
};
```

#### 66-构建乘积数组

[剑指offer 66-构建乘积数组](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/)

给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B[i] 的值是数组 A 中除了下标 i 以外的元素的积, 即 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。
