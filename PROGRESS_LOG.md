# DSA Practice & Build Activity Log


## [2026-08-24 16:27:56 UTC] fix(dsa/dp): resolve index out of bounds in Knapsack 0/1 dynamic programming table initialization

**Module:** `dsa/dp`  
**Status:** Verified & Compiled  

### Summary
Fixed table dimensions `dp[N+1][W+1]` allocation to prevent Segmentation Fault when `W == capacity`.

```cpp
vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
for (int i = 1; i <= n; i++) {
    for (int w = 1; w <= W; w++) {
        if (weights[i-1] <= w)
            dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w]);
        else
            dp[i][w] = dp[i-1][w];
    }
}
```

## [2026-08-24 16:28:07 UTC] docs(dsa/readme): update complexity analysis summary for Sorting Algorithms

**Module:** `dsa/readme`  
**Status:** Verified & Compiled  

### Summary
Documented time/space tradeoffs for QuickSort, MergeSort, HeapSort, and Timsort across best, average, and worst cases.

| Algorithm | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| QuickSort | O(N log N) | O(N log N) | O(N^2) | O(log N) |
| MergeSort | O(N log N) | O(N log N) | O(N log N) | O(N) |
| HeapSort | O(N log N) | O(N log N) | O(N log N) | O(1) |
