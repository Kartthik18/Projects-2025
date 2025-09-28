#include <bits/stdc++.h>
using namespace std;

class Solution
{
public:
    // Better approach: use a hashmap(mpp)
    // The brute force soln(not shown here) is use 2 for loops to select el and thent traverse array
    int majorityElement(vector<int> &nums)
    {
        map<int, int> mpp;
        int n = nums.size();
        for (int i = 0; i < n; i++)
        {
            mpp[nums[i]]++;
        }
        for (auto it : mpp)
        {
            if (it.second > n / 2)
            {
                return it.first;
            }
        }
        return -1;
    }
    // TC= O(NlogN)+O(logN)
    // SC=O(N)
};

class Solution
{
public:
    // Optimal soln:Moore's voting algorithm
    int majorityElement(vector<int> &nums)
    {
        int el;
        int ct = 0;
        int n = nums.size();
        for (int i = 0; i < n; i++)
        {
            if (ct == 0)
            {
                el = nums[i];
                ct++;
            }
            else if (nums[i] == el)
            {
                ct++;
            }
            else
            {
                ct--;
            }
        }
        ct = 0;
        for (int i = 0; i < n; i++)
        {
            if (nums[i] == el)
            {
                ct++;
                if (ct > n / 2)
                {
                    return el;
                }
            }
        }
        return -1;
    }
    // TC=O(2N)
    // SC=O(1)
};
