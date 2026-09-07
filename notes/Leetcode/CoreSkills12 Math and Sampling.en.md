# Math: Fast Power & Rejection Sampling

Math and discrete probability problems are not about memorizing formulas, but about decomposing problems into repeatable binary bit-level operations or constructing uniform discrete sample spaces.

This chapter covers two classic high-frequency interview problems:

1. **Binary Exponentiation (Fast Power)**: LeetCode 50: `Pow(x, n)` — compressing multiplication from $O(n)$ to $O(\log n)$ using bit-level decomposition.
2. **Rejection Sampling**: LeetCode 470: `Implement Rand10() Using Rand7()` — converting between random number generators of different bases via uniform space expansion, truncation, and resampling.

---

## Problem 1: Pow(x, n) (Binary Exponentiation)

### Problem Statement

Given a floating-point number `x` and an integer `n`, return `x^n`.

Examples:

```text
Input:  x = 2.00000, n = 10
Output: 1024.00000
```

```text
Input:  x = 2.00000, n = -2
Output: 0.25000
```

Because:

```text
2^-2 = 1 / 2^2 = 1 / 4
```

### Why You Cannot Brute-Force Multiply n Times

The most naive approach is:

```python
ans = 1
for _ in range(n):
    ans *= x
```

This requires `O(n)` multiplications. If `n = 2^31 - 1`, this approach is far too slow and will time out.

The goal of binary exponentiation is to reduce the complexity to:

```text
O(log n)
```

The underlying mathematical reason is:

```text
x^10 = x^(8 + 2)
```

And the binary representation of `10` is:

```text
10 = 1010₂ = 8 + 2
```

Therefore, we do not need to multiply by `x` one by one. We only need to check which bits in the binary representation of `n` are `1`.

### Core Intuition

In each iteration, maintain two variables:

```text
base = the power represented by the current bit
res  = the product of all powers selected so far
```

Read the binary bits of `n` from right to left (from least significant to most significant).

For `n = 10 = 1010₂`:

```text
Bit weight: 8 4 2 1
Bit value:  1 0 1 0
```

Inspecting from the lowest bit:

```text
Bit 1 (value 0): not selected
Bit 2 (value 1): selected
Bit 4 (value 0): not selected
Bit 8 (value 1): selected
```

So:

```text
x^10 = x^2 * x^8
```

This explains the two core actions in the code:

```python
if power & 1:
    res *= x
```

If the lowest bit is `1`, it means the current power contribution of `x` must be multiplied into the cumulative result.

Then:

```python
x *= x
power >>= 1
```

`x *= x` doubles the current power contribution:

```text
x^1 -> x^2 -> x^4 -> x^8 -> ...
```

`power >>= 1` shifts out the lowest bit so that the next bit can be examined.

### Visualization: Why It Is `res *= base`, Then `base *= base`

```pow-demo
```

Expanding `pow(2, 10)`:

```text
10 = 1010₂
```

Reading bits from right to left:

| Current `power` | Lowest bit | Current `base` | Action | `res` |
|---|---:|---:|---|---:|
| 10 | 0 | 2 | Skip multiplication; square base | 1 |
| 5 | 1 | 4 | Multiply into `res` | 4 |
| 2 | 0 | 16 | Skip multiplication; square base | 4 |
| 1 | 1 | 256 | Multiply into `res` | 1024 |

Final result:

```text
res = 4 * 256 = 2^2 * 2^8 = 2^10
```

### Iterative Binary Exponentiation

#### Algorithm
1. If `x == 0`, return `0`.
2. If `n == 0`, return `1`.
3. Set `res = 1`.
4. Set `power = abs(n)`.
5. While `power > 0`:
   - If `power & 1` is true, perform `res *= x`.
   - Perform `x *= x` (square the base).
   - Perform `power >>= 1` (right-shift by 1 bit).
6. If `n < 0`, return `1 / res`; otherwise return `res`.

#### Python Code

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0:
            return 0
        if n == 0:
            return 1

        res = 1
        power = abs(n)

        while power:
            if power & 1:
                res *= x
            x *= x
            power >>= 1

        return res if n >= 0 else 1 / res
```

#### Complexity
- **Time Complexity**: $O(\log |n|)$ — each round divides `power` by 2, yielding exactly $\lfloor \log_2 |n| \rfloor + 1$ iterations.
- **Space Complexity**: $O(1)$ — requires only $O(1)$ scalar state variables.

#### Common Pitfalls
1. **Forgetting negative exponents**: If you initialize `power = n` directly when `n < 0`, the bit-shift and loop condition behave incorrectly. Always take `abs(n)` and invert the result at the end.
2. **Bitwise precedence**: `power & 1` tests whether the lowest bit is set. It is equivalent to `power % 2 == 1`, but bitwise operations reflect the hardware-level intent.
3. **`x == 0` with negative exponents**: Mathematically, $0^{-1}$ is undefined (division by zero). While LeetCode guarantees valid inputs, robust production code should guard against this case.

---

## Problem 2: Implement Rand10() Using Rand7() (Rejection Sampling)

### Interview Objective

The representative problem in this category is LeetCode 470: `Implement Rand10() Using Rand7()`.

Problem Statement:

```text
Given that rand7() returns an integer from 1 to 7 with uniform probability.
Implement rand10(), which returns an integer from 1 to 10 with uniform probability.
```

The core method is **rejection sampling**.

In an interview, you should clearly articulate three insights:

- Multiple independent calls to `randM()` construct a larger uniform discrete Cartesian product space.
- You cannot simply take `% N` unless the size of the constructed space is cleanly divisible by `N`.
- Keep the largest prefix divisible by `N`, and discard the remaining samples to resample (preserving uniform conditional probability).

### Standard Solution: Implement rand10 with rand7

Call `rand7()` twice:

```python
x = (rand7() - 1) * 7 + rand7()
```

This generates `1..49` with uniform probability.

Why is it uniform? The first `rand7()` picks the row index (`0..6`), and the second `rand7()` picks the column offset (`1..7`):

```text
7 x 7 = 49 cells
```

The probability of landing on each cell is:

```text
1/7 * 1/7 = 1/49
```

Thus `x` is strictly uniformly distributed over `1..49`.

However, `49` is not divisible by `10`. Taking `x % 10` directly would result in some outputs appearing 5 times and others appearing only 4 times, destroying uniformity.

Therefore, we only accept `1..40`:

```text
40 is divisible by 10
Within 1..40, each outcome in 1..10 appears exactly 4 times
Values 41..49 are discarded, triggering a resample
```

Code:

```python
class Solution:
    def rand10(self) -> int:
        while True:
            x = (rand7() - 1) * 7 + rand7()  # uniform 1..49
            if x <= 40:
                return (x - 1) % 10 + 1
```

Note that the return value is written as `(x - 1) % 10 + 1`, mapping into `1..10` rather than `0..9`.

### Why You Cannot Take Modulo Directly

Suppose we directly mapped `1..49` to `1..10` using modulo arithmetic:

```text
1, 11, 21, 31, 41 -> 1
2, 12, 22, 32, 42 -> 2
...
9, 19, 29, 39, 49 -> 9
10, 20, 30, 40    -> 10
```

The first 9 outcomes each occur 5 times, whereas outcome `10` occurs only 4 times:

```text
P(1..9) = 5/49 ≈ 0.1020
P(10)   = 4/49 ≈ 0.0816
```

This violates the uniform distribution requirement.

The essence of rejection sampling is: **only sample from regions where probability can be partitioned evenly. The conditional probability $P(A \mid A \in \text{valid}) = \frac{1/49}{40/49} = \frac{1}{40}$ ensures absolute fairness across all accepted outputs.**

### General Template: Implement randN with randM

Given `randM() -> uniform 1..M`, to implement `randN() -> uniform 1..N`:

1. Use `k` calls to `randM()` to construct a sufficiently large uniform space `1..M^k` (where $M^k \ge N$).
2. Determine the maximum divisible threshold `limit = floor(M^k / N) * N`.
3. If the sampled value `x <= limit`, return `(x - 1) % N + 1`.
4. Otherwise reject and resample.

General Code Template:

```python
def randN():
    while True:
        x = 1
        for _ in range(k):
            x = (x - 1) * M + randM()

        limit = (M ** k // N) * N
        if x <= limit:
            return (x - 1) % N + 1
```

### How to Choose k

The simplest rule: **choose the smallest integer $k$ such that $M^k \ge N$.**

For example, `rand7 -> rand10`:
- $7^1 = 7 < 10$
- $7^2 = 49 \ge 10$
So we use 2 calls to `rand7()`.

Acceptance and rejection probabilities:
```text
usable = floor(49 / 10) * 10 = 40
accept = 40 / 49 ≈ 81.63%
reject = 9 / 49 ≈ 18.37%
```

Expected number of rounds per success (geometric distribution expectation):
```text
E[rounds] = 1 / accept = 49 / 40 = 1.225
```
Each round calls `rand7()` twice, giving the expected number of function calls:
```text
2 * (49 / 40) = 2.45 calls
```

### Variation 1: Reusing Rejected Randomness

The standard solution discards `41..49`. However, conditioned on falling into the rejection zone, these 9 numbers are still uniformly distributed over `1..9`! We can reuse this residual randomness and combine it with a new call to `rand7()` to construct a `9 * 7 = 63` uniform space:

```python
class Solution:
    def rand10(self) -> int:
        while True:
            # Step 1: Use 2 calls to rand7 to generate 1..49
            x = (rand7() - 1) * 7 + rand7()
            if x <= 40:
                return (x - 1) % 10 + 1

            # Step 2: x in 41..49 (9 values), remap to 1..9 and combine with rand7 -> 1..63
            x = (x - 40 - 1) * 7 + rand7()  # uniform 1..63
            if x <= 60:
                return (x - 1) % 10 + 1

            # Step 3: x in 61..63 (3 values), remap to 1..3 and combine with rand7 -> 1..21
            x = (x - 60 - 1) * 7 + rand7()  # uniform 1..21
            if x <= 20:
                return (x - 1) % 10 + 1
            # If x == 21, restart the while loop
```

This optimization reduces the expected number of `rand7()` calls from 2.45 down to approximately 2.19. Presenting the standard solution first and then suggesting this optimization demonstrates mastery of randomized algorithms.

### Variation 2: Implement rand7 with rand5

$5^1 = 5 < 7$, $5^2 = 25 \ge 7$.
`limit = floor(25 / 7) * 7 = 21`.

```python
def rand7():
    while True:
        x = (rand5() - 1) * 5 + rand5()  # 1..25
        if x <= 21:
            return (x - 1) % 7 + 1
```

### Variation 3: Implement rand3 with rand2

Two calls to `rand2()` generate `1..4`, with `limit = 3`:

```python
def rand3():
    while True:
        x = (rand2() - 1) * 2 + rand2()  # 1..4
        if x <= 3:
            return x
```

### Variation 4: Implement rand7 with rand10 (Downscaling)

When the source space is already larger than the target space ($M \ge N$), a single call with simple truncation suffices:

```python
def rand7():
    while True:
        x = rand10()
        if x <= 7:
            return x
```

### Variation 5: Generate Arbitrary Range [a, b]

For an interval of size $N = b - a + 1$, generate `randN()` in range `1..N`, then shift:

```python
def randRange(a: int, b: int) -> int:
    return randN() + a - 1
```

### Variation 6: Fair Coin from Biased Coin (Von Neumann Trick)

If the random generator is biased (e.g., a coin lands heads with probability $p \ne 0.5$), Cartesian multiplication cannot be used directly.

John von Neumann's trick is to toss the biased coin twice in succession:
- Result `(Heads, Tails)` has probability $p(1 - p)$.
- Result `(Tails, Heads)` has probability $(1 - p)p = p(1 - p)$.
- These two compound probabilities are identical! Assign `(H, T) -> 0` and `(T, H) -> 1`.
- If `(H, H)` or `(T, T)` occurs, discard and repeat.

This transforms any biased Bernoulli generator of unknown parameter $p$ into an unbiased, fair coin.

### Correctness Proof

1. **Uniform Cartesian Space**: $k$ independent uniform calls to `randM()` yield $M^k$ equally likely elementary outcomes, so $x$ is strictly uniform over $[1, M^k]$.
2. **Conditional Uniformity**: Let $limit = \lfloor M^k / N \rfloor \times N$. Conditioned on $x \le limit$:
   $$P(x = v \mid x \le limit) = \frac{P(x = v)}{P(x \le limit)} = \frac{1 / M^k}{limit / M^k} = \frac{1}{limit}$$
3. **Partition Uniformity**: The mapping $f(x) = ((x - 1) \pmod N) + 1$ partitions the set of size $limit$ into $N$ equal buckets of size $limit / N$. For every target value $y \in [1, N]$:
   $$P(f(x) = y \mid x \le limit) = \frac{limit / N}{limit} = \frac{1}{N}$$
4. **Resampling Preserves Fairness**: Values exceeding $limit$ are rejected without selection. Since trials are independent and identically distributed, discarding rejects does not bias any output.

Thus, the generated values are strictly uniformly distributed over $[1, N]$.

### Complexity Summary

For `rand7 -> rand10`:
- Acceptance probability: $P_{\text{accept}} = 40 / 49$
- Expected calls: $2 \times \frac{49}{40} = 2.45$
- Space complexity: $O(1)$

### Common Pitfalls
- Taking `% N` directly without ensuring the constructed domain size is divisible by $N$.
- Forgetting that `randM()` is 1-indexed, omitting the minus 1 when computing base-$M$ expansions (`(x - 1) * M + randM()`).
- Returning `(x % N) + 1` instead of `(x - 1) % N + 1`, which breaks boundary outputs when $x$ is a multiple of $N$.
- Fearing an infinite loop in rejection sampling: because $P_{\text{accept}} > 0$, the number of trials follows a geometric distribution with finite expectation and terminates with probability 1.

### Interview Answer Template

<details class="solution">
<summary>Expand Interview Template</summary>

We cannot simply take modulo on `rand7()` because 7 or 49 is not divisible by 10; doing so would make some remainders appear more often than others, destroying uniformity.

The standard approach uses Rejection Sampling:
1. Call `rand7()` twice to construct a uniform discrete space of size 49:
   `x = (rand7() - 1) * 7 + rand7()`
2. Find the largest multiple of 10 within 49, which is 40. Accept only samples where $x \le 40$.
3. Map accepted samples to `1..10` using `(x - 1) % 10 + 1`.
4. Discard samples in `41..49` and resample.

Within `1..40`, each integer in `1..10` appears exactly 4 times. By Bayes' theorem on conditional probability, each output occurs with probability exactly $1/10$. The expected number of `rand7()` calls is $2 \times (49/40) = 2.45$, and auxiliary space is $O(1)$.

</details>
