# Review Flashcards: Core Fundamentals (Review 1)

This module provides high-yield algorithm interview review flashcards: distilled **Problem Definitions**, **Core Mental Models**, **Minimal Core Implementations**, **Complexity Invariants**. Click any card title to expand.

---

### 1. Merge Sort

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Core 01</span>
  <span class="review-card-title">Merge Sort</span>
  <span class="review-card-tag">Divide &amp; Conquer · Stable</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Sort an unsorted array of $n$ integers in non-decreasing order. Worst-case time complexity must be strictly guaranteed to be $O(n \log n)$, preserving the relative order of duplicate elements (stability).

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Canonical Divide & Conquer three-step pipeline:
1. **Divide**: Compute midpoint $mid = \lfloor (l + r) / 2 \rfloor$ to split into equal halves.
2. **Conquer**: Recursively sort left and right halves until subsegments reach base case length $\le 1$.
3. **Combine**: Linearly merge using two pointers; on ties, prefer left elements to guarantee stability.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
def merge_sort(nums: list[int]) -> list[int]:
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left, right = merge_sort(nums[:mid]), merge_sort(nums[mid:])
    
    # Core two-pointer merge (<= ensures stability)
    res, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i]); i += 1
        else:
            res.append(right[j]); j += 1
    return res + left[i:] + right[j:]
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: Best $O(n \log n)$ / Worst $O(n \log n)$ / Average $O(n \log n)$ (Tree height $\log n$, level merge work fixed at $O(n)$)
- **Auxiliary Space**: $O(n)$ (merge buffer) + $O(\log n)$ (call stack frames)
- **Stability**: **Stable** (left-half precedence on ties)

</div>

</div>
</details>

---

### 2. Quick Sort

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Core 02</span>
  <span class="review-card-title">Quick Sort</span>
  <span class="review-card-tag">Partitioning · In-Place · Unstable</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Sort an unsorted array of $n$ integers in non-decreasing order in-place. Average time complexity must achieve $O(n \log n)$, requiring no auxiliary data structures beyond recursive stack frames.

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Key mechanism: **Partitioning before Recursion**:
1. **Randomized Pivot**: Use `random.randint(l, r)` to pick a random element and swap it with the end $r$. This completely breaks adversarial inputs (e.g., sorted or reverse-sorted arrays), eliminating the $O(n^2)$ worst-case skew.
2. **Dedicated Partition Function**: Lomuto partitioning maintains pointer $i$ as the right boundary of elements $\le pivot$. Iterate across $[l, r - 1]$, swapping elements $\le pivot$ into place; finally swap pivot with $nums[i]$ and return split index $p = i$.
3. **Divide & Conquer Recursion**: Recursively sort subarrays around the pivot: `[l, p - 1]` and `[p + 1, r]`.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
import random

def partition(nums: list[int], l: int, r: int) -> int:
    # 1. Random pivot selection to prevent worst-case O(n^2) degeneration
    rand_idx = random.randint(l, r)
    nums[rand_idx], nums[r] = nums[r], nums[rand_idx]

    # 2. Core Lomuto partition: i maintains boundary of elements <= pivot
    pivot, i = nums[r], l
    for j in range(l, r):
        if nums[j] <= pivot:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    nums[i], nums[r] = nums[r], nums[i]
    return i

def quick_sort(nums: list[int], l: int, r: int) -> None:
    if l >= r:
        return
    p = partition(nums, l, r)
    quick_sort(nums, l, p - 1)
    quick_sort(nums, p + 1, r)
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: Best $O(n \log n)$ / Worst $O(n^2)$ (skewed partitions) / Average $O(n \log n)$
- **Auxiliary Space**: $O(\log n)$ (stack frames, degrades to $O(n)$ in worst case)
- **Stability**: **Unstable** (long-distance swaps disrupt relative order)

</div>

</div>
</details>

---

### 3. Dynamic Array Implementation

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Core 03</span>
  <span class="review-card-title">Dynamic Array Implementation</span>
  <span class="review-card-tag">Contiguous Memory · Geometric Doubling · Amortized</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Implement a resizable dynamic array from scratch backed by a fixed-size contiguous buffer (analogous to Python `list` or C++ `std::vector`), supporting $O(1)$ random indexing, tail append `push_back`, tail pop `pop_back`, and automatic doubling expansion.

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Mental model and amortized constant time rationale:
1. **Contiguous Buffer**: Maintain fixed capacity `cap` with active item count `size`.
2. **Geometric Doubling**: When `size == cap`, allocate a new contiguous chunk of $2 \times cap$, copy elements across, and discard old buffer.
3. **Amortized Analysis ($O(1)$)**: A single expansion copies $O(n)$ elements, but occurs exponentially less often. Sum of all copies $1 + 2 + 4 + \dots + n \le 2n$. Amortized over $n$ appends, cost is strictly $O(1)$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
class DynamicArray:
    def __init__(self, capacity: int = 2):
        self.cap, self.size = capacity, 0
        self.arr = [None] * self.cap

    def push_back(self, val: int) -> None:
        # Core: Geometric doubling when full, amortized O(1)
        if self.size == self.cap:
            self.cap *= 2
            new_arr = [None] * self.cap
            for i in range(self.size):
                new_arr[i] = self.arr[i]
            self.arr = new_arr
        self.arr[self.size] = val
        self.size += 1

    def pop_back(self) -> int:
        self.size -= 1
        return self.arr[self.size]

    def get(self, i: int) -> int:
        return self.arr[i]
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Random Indexing `get`/`set`**: $O(1)$ (direct memory address calculation $base + i \times size$)
- **Tail Append `push_back`**: **Amortized $O(1)$** (Worst $O(n)$ during expansion)
- **Tail Pop `pop_back`**: $O(1)$
- **Space Utilization**: $\ge 50\%$

</div>

</div>
</details>

---

### 4. Binary Search Boundary Template

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Core 04</span>
  <span class="review-card-title">Binary Search Boundary Template</span>
  <span class="review-card-tag">Monotonic Search · Interval Invariant</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition &amp; Invariants</div>

Given a non-decreasing sorted integer array, locate the **first occurrence (Lower Bound)** of `target`. If absent, return the index where it should be inserted. Must run in $O(\log n)$ with zero danger of infinite loops.

</div>

<div class="review-block">
<div class="review-block-label">💡 Core Approach &amp; Mental Model</div>

Rigid **Loop Invariant maintenance**:
1. **Closed Interval**: Maintain a **closed range $[l, r]$**, initialized with $l = 0, r = len(nums) - 1$.
2. **Overflow-safe Midpoint**: $mid = l + \lfloor (r - l) / 2 \rfloor$.
3. **Shrinking Decision**:
   - If $nums[mid] \ge target$: Target is at $mid$ or left; shrink right bound: $r = mid - 1$.
   - If $nums[mid] < target$: Target is strictly right; shrink left bound: $l = mid + 1$.
4. **Convergence**: Loop while $l \le r$. Terminates strictly when $l = r + 1$, where $l$ lands on the first item $\ge target$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Core Python Implementation (Minimal)</div>

```python
def search_lower_bound(nums: list[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    # Strictly maintain closed interval [l, r]
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] >= target:
            r = mid - 1  # Seek lower index to the left
        else:
            l = mid + 1
    return l  # Terminates with l as first index >= target
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ Complexity &amp; Key Properties</div>

- **Time Complexity**: $O(\log n)$ (halves search space every iteration)
- **Auxiliary Space**: $O(1)$ (iterative without stack frames)
- **Termination Invariant**: Loop always terminates with $l = r + 1$

</div>

</div>
</details>
