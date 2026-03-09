# Optimizing Memory-Bound Kernels

This lecture builds on the three parallel primitives from the previous two lectures: Reduce, Histogram, and Scan. What they share is extremely low arithmetic intensity, so they are all memory-bound kernels. For kernels in this category, the optimization goal is always to **maximize effective bandwidth utilization**.

This lecture distills the optimization techniques used piecemeal in earlier lectures into a general analytical framework that can be transferred to any memory-bound kernel.

---

## 1. Roofline Analysis: Identifying the Performance Bottleneck

The first step in optimizing any kernel is to establish its performance upper bound. The Roofline model gives the following relationship:

$$P \le \min(\pi,\ \beta \cdot I), \quad I = \frac{\text{FLOPs}}{\text{Bytes}}$$

- $\pi$: peak GPU compute throughput (FLOP/s)
- $\beta$: memory bandwidth (Byte/s)
- $I$: arithmetic intensity (FLOP/Byte)
- $I_{ridge} = \pi / \beta$: ridge point

When $I \ll I_{ridge}$, kernel performance is limited by memory bandwidth rather than compute throughput.

> [!tip] Why Roofline matters
> Roofline turns optimization into a quantifiable analysis problem: how far current performance is from the theoretical upper bound, and what kind of bottleneck is responsible. Without this baseline, there is no solid basis for choosing an optimization direction.

**Example**: The Reduce kernel from the previous lecture reads N floats (4N bytes) and performs N-1 additions, so $I \approx 0.25$ FLOP/Byte. On an A100, $I_{ridge} > 100$, a gap of more than two orders of magnitude, making it a textbook memory-bound kernel.

---

## 2. Five Core Principles for Memory-Bound Optimization

Summarizing optimization experience across kernels such as transpose, stencil, SpMV, histogram, and compaction, we can extract the following five general principles:

### Principle A: Byte Accounting

Before optimizing, you need an accurate estimate of the kernel's total memory traffic. This step determines whether subsequent optimization work is actually targeting the real bottleneck.

Rule of thumb: **sum the bytes of all read and write operations**, and remember that an atomic operation is fundamentally a read-modify-write, so its memory traffic should be counted as 2-3x.

> [!note] A common pitfall
> Cutting FLOPs in half without reducing memory accesses does not improve kernel runtime at all. Worse, reducing computation by introducing extra intermediate arrays can increase memory traffic and actually hurt performance.

**Example: Byte accounting for vector addition**
```
// C[i] = A[i] + B[i], N floats
// Read: A (4N bytes) + B (4N bytes) = 8N bytes
// Write: C (4N bytes)
// Total memory traffic = 12N bytes
// Peak bandwidth 900 GB/s -> theoretical lower bound = 12N / 900G seconds
```

**Example: Byte accounting for Histogram (easy to miscalculate)**
```
// Input: N ints (read 4N bytes)
// Output: bins[] uses atomicAdd
// atomic = read + modify + write -> each update is about 3x4 = 12 bytes
// Total traffic = 4N + 12N = 16N bytes (not the naive 4N + 4N)
```

---

### Principle B: Coalescing

The ideal memory access pattern is: **the 32 threads in a warp access a contiguous 128-byte segment** (for `float`, for example).

More concretely:
- lanes within a warp should map to consecutive memory addresses
- the overall access pattern should be as close to sequential streaming as possible

Coalescing is the prerequisite for all other optimizations. If coalescing is not satisfied, the upper limit of effective bandwidth drops dramatically.

**Example: The coalescing issue in matrix transpose**
```
// Bad case: read by column, warp threads access with stride N
out[j][i] = in[i][j]   // in read by row (coalesced) ✓
                        // out written by column (strided) ✗ -> bandwidth utilization drops sharply

// Good case: use shared memory as a staging buffer
tile[threadIdx.y][threadIdx.x] = in[row][col]   // coalesced read
__syncthreads()
out[col][row] = tile[threadIdx.x][threadIdx.y]   // coalesced write
```

**Example: AoS vs SoA**
```
// AoS (Array of Structs) — when a warp reads x, the stride is sizeof(Point)
struct Point { float x, y, z; };
Point pts[N];            // pts[tid].x → stride=12 bytes ✗

// SoA (Struct of Arrays) — when a warp reads x, accesses are contiguous
float px[N], py[N], pz[N];
px[tid]                  // stride=4 bytes, perfectly coalesced ✓
```

---

### Principle C: Explicit Reuse (Tiling)

When a kernel has neighborhood structure or data reuse (such as stencil, convolution, or some sparse local operators):

- do not rely on implicit hits in the hardware L1/L2 cache
- use tiling (shared memory or registers) to turn reuse into deterministic behavior

The core idea is: **load data from HBM into SRAM and reuse it multiple times to avoid repeated HBM accesses**.

> [!info] What is a Stencil?
> A stencil is a common computational pattern in which each output element is computed as a weighted sum of **the input element itself and input elements in a fixed neighborhood**. A 1D stencil with radius R means that `out[i]` depends on `in[i-R] ... in[i+R]`, for a total of 2R+1 elements. Typical applications include finite differences (CFD/PDE solvers), image blur/sharpening (2D stencil), audio filtering, and more. Because the input windows of neighboring output points overlap heavily, stencil is a classic use case for tiling.

**Example: 1D stencil — without tiling vs with tiling**
```
// Without tiling: each output point reads 2R+1 neighbors from HBM
// Neighboring threads have heavily overlapping reads -> depends on cache hits, not controllable
out[i] = Σ w[k] * in[i-R+k],  k=0..2R

// With tiling: the block cooperatively loads a tile segment (including halo) into shared memory
__shared__ float tile[BLOCK + 2*R];
tile[threadIdx.x + R] = in[blockStart + threadIdx.x];
if (threadIdx.x < R) {               // load left and right halo
    tile[threadIdx.x] = in[blockStart - R + threadIdx.x];
    tile[BLOCK + R + threadIdx.x] = in[blockStart + BLOCK + threadIdx.x];
}
__syncthreads();
out[i] = Σ w[k] * tile[threadIdx.x + k];  // all hits come from SRAM
```

---

### Principle D: Reduce Synchronization and Contention (Sync/Contention)

For memory-bound kernels, the performance bottleneck is often not the bandwidth itself, but rather:
- overly frequent `__syncthreads()` calls, which turn pipeline throughput into serialized waiting
- contention on atomic operations (histogram, scatter), which degrades parallel writes into serialized writes

The "hierarchical privatization" strategy in the previous Histogram lecture is a canonical application of this principle: reduce the scope of contention progressively from global to block to warp, thereby lowering contention overhead.

**Example: Hierarchical privatization for Histogram**
```
// Level 1 — global atomic (maximum contention)
atomicAdd(&global_bins[val], 1);           // all threads contend for the same set of bins

// Level 2 — block-private bins -> final reduction
__shared__ int local_bins[NUM_BINS];       // one copy per block
atomicAdd(&local_bins[val], 1);            // contention shrinks to within the block
__syncthreads();
atomicAdd(&global_bins[tid], local_bins[tid]);  // one-shot reduction

// Level 3 — warp-private bins (registers/shared-memory partitioning)
// contention shrinks further to within 32 threads, with almost no conflicts
```

---

### Principle E: Latency Hiding

When high memory latency is unavoidable (for example, the random accesses in SpMV), latency can be hidden in the following ways:
- **increase occupancy**: raise the number of resident warps so that more warps can be scheduled while others are waiting on memory
- **increase ILP (instruction-level parallelism)**: use unrolling and multiple-elements-per-thread strategies so that each thread can issue more load requests concurrently

The grid-stride loop is a general engineering pattern for realizing this principle.

**Example: Grid-stride loop + ILP unrolling**
```
// Basic version: each thread handles one element; occupancy is the only latency-hiding mechanism
for (int i = tid; i < N; i += gridDim.x * blockDim.x)
    out[i] = f(in[i]);

// ILP version: each thread issues multiple loads at once to hide memory latency
for (int i = tid; i < N; i += stride * 4) {
    float a = in[i];
    float b = in[i + stride];
    float c = in[i + stride*2];
    float d = in[i + stride*3];    // 4 loads in flight at the same time
    out[i]            = f(a);
    out[i + stride]   = f(b);
    out[i + stride*2] = f(c);
    out[i + stride*3] = f(d);
}
```

---

## 3. Pattern Library: Six Classes of Memory-Bound Kernels

Below, memory-bound kernels are grouped into six categories based on their memory access patterns. When facing a new kernel, first determine which category it belongs to, then combine the corresponding principles for optimization.

---

### Pattern 1: Streaming (Linear Read/Write)

**Typical scenarios**: `out[i] = f(in[i])`, elementwise map, vector addition

**Memory access characteristics**: input is read sequentially, output is written sequentially, and there is no data reuse.

**Applicable principles**: B (coalescing) + E (latency hiding)

**Optimization techniques**: vectorized load/store (`float4`), memory alignment, grid-stride loop, loop unrolling

```cpp
// Vectorized elementwise kernel
// Use float4 for vectorized loads and stores; each memory transaction moves 16 bytes
__global__ void vector_add_v4(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n4  // n / 4, i.e. the number of float4 elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n4; i += stride) {
        float4 va = a[i];  // 128-bit load
        float4 vb = b[i];
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        c[i] = vc;  // 128-bit store
    }
}

//// Approximate definition of float4
// struct float4 {
//    float x, y, z, w;
//};

// Launch configuration
// n is the total number of float elements and must be a multiple of 4 (otherwise the tail needs extra handling)
// vector_add_v4<<<(n/4 + 255) / 256, 256>>>(a4, b4, c4, n/4);
```

> [!note] Why `float4` helps
> After switching to `float4`, each load/store instruction moves 16 bytes instead of 4. This reduces the total number of load/store instructions required, allowing the compiler to schedule the instruction pipeline more effectively and improve ILP. When the 32 threads in a warp execute `float4` loads simultaneously, they generate 4 memory transactions of 128B each, transferring 512 bytes in total.

---

### Pattern 2: Reorder / Permutation

**Typical scenario**: Matrix Transpose

**Core conflict**: the read direction and write direction are orthogonal. If the reads are coalesced, the writes are necessarily strided, and vice versa.

**Solution**: use shared memory as an intermediate buffer. Read along the row direction and coalesce into shared memory, then write from shared memory along the column direction with coalescing.

**Applicable principles**: B (both reads and writes must be coalesced) + C (shared memory as a reordering buffer) + D (avoid bank conflicts)

```cpp
// Shared-memory-tiled matrix transpose
// Input [M x N], output [N x M]
#define TILE_DIM 32
#define BLOCK_ROWS 8  // block dimensions: TILE_DIM x BLOCK_ROWS
                      // each block processes a TILE_DIM x TILE_DIM tile

__global__ void transpose_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    // Columns +1 as padding to eliminate bank conflicts (see explanation below)
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Step 1: load from global memory into shared memory along rows (coalesced read)
    // Each thread is responsible for loading TILE_DIM / BLOCK_ROWS = 4 rows
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < M) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * N + x];
        }
    }

    __syncthreads();

    // Step 2: after swapping coordinates, write from shared memory to global memory along rows (coalesced write)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < M && (y + j) < N) {
            output[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Launch configuration
// dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
// dim3 block(TILE_DIM, BLOCK_ROWS);
// transpose_optimized<<<grid, block>>>(input, output, M, N);
```

> [!important] Padding to eliminate bank conflicts
> Shared memory consists of 32 banks, each 4 bytes wide. If the tile has exactly 32 columns, then all elements in the same column map to the same bank, causing a 32-way bank conflict during column-wise accesses. By setting the number of columns to `TILE_DIM + 1 = 33`, elements at the same column position in adjacent rows are shifted by one bank, eliminating the conflict. This trick is widely used in kernels that perform column-wise accesses in shared memory.

---

### Pattern 3: Stencil / Neighborhood

**Typical scenarios**: 1D/2D stencil, image convolution, image-processing filters

**Core characteristic**: each input element is reused by multiple neighboring output points. For a 1D 3-point stencil, for example, each input element is read once by the left, center, and right output points.

**Applicable principles**: C (tile + halo for deterministic reuse) + B (keep halo loading coalesced)

```cpp
// 1D Stencil: out[i] = c0*in[i-R] + c1*in[i-R+1] + ... + c2R*in[i+R]
// R = stencil radius; this example uses R = 4 (9-point stencil)

#define RADIUS 4
#define BLOCK_SIZE 256
// Each block computes BLOCK_SIZE output points
// It needs to load BLOCK_SIZE + 2*RADIUS input points (including halo regions on both sides)

__constant__ float coeff[2 * RADIUS + 1];  // stencil coefficients stored in constant memory

__global__ void stencil_1d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float smem[BLOCK_SIZE + 2 * RADIUS];

    int gidx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int lidx = threadIdx.x + RADIUS;  // offset index inside shared memory

    // Step 1: load the center region
    smem[lidx] = (gidx < n) ? input[gidx] : 0.0f;

    // Step 2: load the left halo (handled by the first RADIUS threads)
    if (threadIdx.x < RADIUS) {
        int halo_idx = gidx - RADIUS;
        smem[threadIdx.x] = (halo_idx >= 0) ? input[halo_idx] : 0.0f;
    }

    // Step 3: load the right halo (handled by the last RADIUS threads)
    if (threadIdx.x >= BLOCK_SIZE - RADIUS) {
        int halo_idx = gidx + RADIUS;
        smem[lidx + RADIUS] = (halo_idx < n) ? input[halo_idx] : 0.0f;
    }

    __syncthreads();

    // Step 4: compute the stencil from shared memory, with no global memory access
    if (gidx < n) {
        float result = 0.0f;
        #pragma unroll
        for (int j = -RADIUS; j <= RADIUS; j++) {
            result += coeff[j + RADIUS] * smem[lidx + j];
        }
        output[gidx] = result;
    }
}
```

**Memory traffic comparison**:

| Version | Global memory read volume | Explanation |
|------|---------------------|------|
| Naive (no shared memory) | $N \times (2R+1)$ | Each output point reads $2R+1$ inputs from HBM |
| Tiled (using shared memory) | $N + 2R \times \text{num\_blocks}$ | Each input element is essentially loaded from HBM only once |

When the stencil radius $R$ becomes larger, data reuse increases and the benefit of tiling becomes more significant.

---

### Pattern 4: Indirection / Gather (Irregular Reads)

**Typical scenarios**: CSR-format SpMV, gather, embedding lookup

**Core difficulty**: indirect indexing `x[col_idx[j]]` makes access addresses depend on the data itself, making ideal coalescing hard to achieve.

**Applicable principles**: A (include index bytes in the accounting) + E (hide latency through warp-level mapping)

Sparse matrix-vector multiplication (SpMV)

```
CSR (Compressed Sparse Row) stores a sparse matrix using three arrays:
═══════════════════════════════════════════════════════════════════════════════

1. values[nnz]:   all nonzero values, stored row by row
2. col_idx[nnz]:  the column index corresponding to each nonzero value
3. row_ptr[M+1]:  the starting position of each row in values/col_idx

For the matrix above:

values[]:   [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
             ↑       ↑   ↑   ↑   ↑           ↑   ↑
            row0    row0 row1 row1 row2      row2 row3

col_idx[]:  [ 0,  2,  4,  1,  5,  0,  1,  2,  3,  5]
             column index for each nonzero element

row_ptr[]:  [ 0,  3,  5,  9, 10]
              ↑   ↑   ↑   ↑   ↑
             row0 row1 row2 row3 end
             start start start start

The indices between row_ptr[i] and row_ptr[i+1] are the nonzero elements of row i
```


warp-per-row strategy
```
Core idea: one warp (32 threads) cooperatively processes one matrix row
═══════════════════════════════════════════════════════════════════════════════

Why not use Thread-per-Row?
─────────────────────────────
If a row has many nonzero elements (for example, 1000), serial processing by a single thread is too slow

Advantages of Warp-per-Row:
─────────────────────────────
1. 32 threads process a row's nonzero elements in parallel
2. Accesses to values[] and col_idx[] are contiguous -> Coalesced Access
3. Uses Warp Shuffle for reduction, with no Shared Memory needed
```

```cpp
// CSR-format SpMV: y = A * x
// CSR storage: row_ptr[M+1], col_idx[nnz], values[nnz]
//
// Mapping strategy:
//   Row lengths are fairly uniform: thread-per-row
//   Row lengths vary widely: warp-per-row (this example uses this strategy)

// Warp-per-row: each warp (32 threads) processes one matrix row
// Advantages:
//   1. Threads within a warp access consecutive col_idx and values entries (coalesced)
//   2. Uses warp shuffle for reduction, with no shared memory or atomic needed
//   3. 32 threads issue loads simultaneously, effectively hiding latency
__global__ void spmv_csr_warp_per_row(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    if (warp_id >= num_rows) return;

    int row_start = row_ptr[warp_id];
    int row_end   = row_ptr[warp_id + 1];

    float sum = 0.0f;

    // The 32 threads in the warp iterate over this row's nonzeros with stride 32
    for (int j = row_start + lane; j < row_end; j += 32) {
        // col_idx[j], values[j]: contiguous accesses, coalesced
        // x[col_idx[j]]: random access, relies on L2 cache and latency hiding
        sum += values[j] * x[col_idx[j]];
    }

    // Warp-level reduction (warp shuffle), no shared memory or barrier needed
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[warp_id] = sum;
    }
}

// Launch configuration
// int threads_per_block = 256;  // each block contains 8 warps
// int num_blocks = (num_rows * 32 + threads_per_block - 1) / threads_per_block;
// spmv_csr_warp_per_row<<<num_blocks, threads_per_block>>>(...);
```

If all lanes need the result, you can broadcast with `__shfl_sync`, or use `__shfl_xor_sync` for a butterfly reduction. But here we only need to write a single `y[warp_id]`, so it is enough for lane 0 alone to hold the result.

> [!note] The bandwidth ceiling for sparse kernels
> For sparse kernels, the achievable bandwidth ceiling is often lower than the theoretical HBM peak. The access pattern of `x[col_idx[j]]` is determined by the sparsity structure of the matrix and cannot be fully controlled at the kernel level. As a result, the optimization goal shifts toward "reducing randomness in accesses," for example by reordering matrix columns to improve locality when accessing vector `x`.

---

### Pattern 5: Scatter / Atomic (Irregular Writes + Contention)

**Typical scenarios**: histogram, scatter-add, some graph algorithms

**Core difficulty**: the write address depends on the data values, and hot destinations (such as high-frequency bins) cause severe serialization of atomic operations.

**Applicable principle**: D (hierarchical privatization)

The optimization method for this class of kernels was already presented in detail in the three Histogram versions from the previous lecture. The core strategy is as follows:

```
Level 0: Global atomic — all threads contend for global memory
  ↓ Privatize
Level 1: Block-level shared memory atomic — contention shrinks to 256 threads
  ↓ Further privatize
Level 2: Warp-level / register-local accumulation — contention shrinks to 32 threads or is fully eliminated
  ↓ Final reduction
Write back to global memory — the number of atomic calls drops from N to num_bins × num_blocks
```

---

### Pattern 6: Filter / Compaction (Conditional Filtering)

**Typical scenarios**: stream compaction, remove-zero, predicate filter

#### What is Stream Compaction?

**Filter out elements that satisfy a condition and store them compactly**

```
Input:      [ 3, -1, 4, 0, -2, 5, 0, 1 ]
Condition:  element > 0
Output:     [ 3, 4, 5, 1 ]
```

#### Why is it hard to parallelize?

Each thread does not know its own output position — **solve it with an Exclusive Prefix Sum**:

```
flag:   [ 1,  0,  1,  0,  0,  1,  0,  1 ]   ← mark elements satisfying the condition
scan:   [ 0,  1,  1,  2,  2,  2,  3,  3 ]   ← exclusive scan

scan[i] = "how many satisfied the condition before me" = my output position
```

#### Three-step workflow

| Step | What it does | Code |
|-----|-------|------|
| **Flag** | Mark elements that satisfy the condition | `flag = (val > 0) ? 1 : 0` |
| **Scan** | Exclusive prefix sum (Blelloch algorithm) | Up-sweep + Down-sweep |
| **Scatter** | Write to the correct position | `output[offset + scan[tid]] = val` |


```cpp
// Stream Compaction: filter elements > 0 from input and write them compactly into output
// Fused implementation: complete flag, block-level scan, and scatter inside a single kernel

#define BLOCK_SIZE 256

__global__ void stream_compaction(
    const int* __restrict__ input,
    int* __restrict__ output,
    int* __restrict__ output_count,  // total number of output elements
    int n
) {
    __shared__ int scan[BLOCK_SIZE];
    __shared__ int block_output_offset;

    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    // Step 1: Flag — mark elements satisfying the condition
    int val = 0;
    int flag = 0;
    if (gid < n) {
        val = input[gid];
        flag = (val > 0) ? 1 : 0;
    }
    scan[tid] = flag;
    __syncthreads();

    // Step 2: Block-level exclusive scan (Blelloch algorithm)

    // Up-sweep
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        int ai = (tid + 1) * offset * 2 - 1;
        if (ai < BLOCK_SIZE) {
            scan[ai] += scan[ai - offset];
        }
        __syncthreads();
    }

    // Extract the total number of qualifying elements in the block and allocate output space with one atomic
    if (tid == 0) {
        int block_total = scan[BLOCK_SIZE - 1];
        block_output_offset = atomicAdd(output_count, block_total);
        scan[BLOCK_SIZE - 1] = 0;
    }
    __syncthreads();

    // Down-sweep
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        int ai = (tid + 1) * offset * 2 - 1;
        if (ai < BLOCK_SIZE) {
            int temp = scan[ai - offset];
            scan[ai - offset] = scan[ai];
            scan[ai] += temp;
        }
        __syncthreads();
    }

    // Step 3: Scatter — write qualifying elements to the correct positions in the output array
    if (gid < n && flag) {
        output[block_output_offset + scan[tid]] = val;
    }
}
```

> [!tip] Key optimization
> `atomicAdd(output_count, block_total)` is called `num_blocks` times rather than N times. This combines Principle D (reduce contention) with Principle A (reduce the memory cost of atomics): each block computes local offsets internally via scan, and only a single atomic operation is needed at the end to allocate output space.

---

## 4. Optimization Checklist

When optimizing a memory-bound kernel, it is recommended to check the following items in order:

| Step | Item | How to observe it |
|------|------|----------|
| 1 | Compute arithmetic intensity and bandwidth upper bound | Roofline model, manual calculation |
| 2 | Measure effective bandwidth | Bytes / kernel_time, compare against the theoretical peak |
| 3 | Check coalescing | `ncu`: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` |
| 4 | Check contention and synchronization overhead | `ncu`: atomic throughput, barrier wait time |
| 5 | Tune occupancy and ILP | grid-stride loop, loop unrolling, multiple elements per thread |

The importance of this order is that if coalescing is not satisfied, then tuning occupancy will yield only limited benefit. You should first ensure that the memory access pattern is correct, and only then move on to higher-level tuning.

---

## 5. Summary

| Principle | Optimization goal | Typical applicable kernels |
|------|----------|----------------|
| A Byte accounting | Accurately quantify memory traffic | All kernels |
| B Coalescing | Consecutive addresses within a warp, fewer memory transactions | streaming, transpose |
| C Explicit reuse | Reuse data multiple times after loading it from HBM into SRAM | stencil, some sparse kernels |
| D Reduce contention | Lower the serialization cost of barriers and atomics | histogram, scatter, compaction |
| E Latency hiding | Mask memory latency with parallelism and ILP | sparse, gather |

The optimizations for Reduce, Histogram, and Scan in the previous two lectures already covered all of these principles (`coalescing`, `first add during load`, `warp shuffle`, `ILP`, `grid-stride loop`). The purpose of this lecture is to abstract those techniques, previously tied to specific examples, into a general analytical framework.

When facing a new memory-bound kernel, first determine which Pattern (1-6) it belongs to, and then formulate an optimization strategy by combining the corresponding principles.
