# Memory-Bound Kernel optimization

This lecture inherits the three parallel primitives of Reduce, Histogram, and Scan from the previous two lectures. The common feature of these three is that the arithmetic intensity is extremely low, and they are all memory-bound kernels. For this type of kernel, the optimization goal is always to maximize effective bandwidth utilization.

This lecture refines the optimization techniques used scatteredly in the previous lectures into a set of general analysis framework, so that they can be migrated to any memory-bound kernel.

---

## 1. Roofline analysis: identify performance bottlenecks

The first step in optimizing any kernel is to establish an upper bound on performance. The Roofline model gives the following relationship:

$$P \le \min(\pi,\ \beta \cdot I), \quad I = \frac{\text{FLOPs}}{\text{Bytes}}$$

- $\pi$: GPU peak computing power (FLOP/s)
- $\beta$: memory bandwidth (Byte/s)
- $I$: Arithmetic intensity (FLOP/Byte)
- $I_{ridge} = \pi / \beta$: ridge point

When $I \ll I_{ridge}$, the performance of the kernel is limited by the memory bandwidth rather than the computing power.

> [!tip]Roofline meaning
> Roofline transforms optimization problems into quantifiable analysis: how far the current performance is from the theoretical bound and what category the bottleneck belongs to. Without this benchmark, there is no basis for choosing the optimization direction.

**Example**: The Reduce kernel in the previous lecture reads N floats (4N bytes), performs N-1 additions, $I \approx 0.25$ FLOP/Byte. A100's $I_{ridge} > 100$, which is two orders of magnitude different, is typical memory-bound.

---

## 2. Five core principles of Memory-Bound optimization

After summarizing the optimization experience of various kernels such as transpose, stencil, SpMV, histogram, and compaction, the following five general principles can be extracted:

### Principle A: Byte Accounting

Before optimization, it is necessary to accurately calculate the total memory traffic of the kernel. This step determines whether subsequent optimization targets the real bottleneck.

Calculation rules: **Add the number of bytes of all read and write operations**. The essence of the atomic operation is read-modify-write, and its memory traffic needs to be calculated as 2-3 times.

> [!note]Common misunderstandings
> Halving FLOPs without reducing memory accesses does not improve the kernel's execution time. What's worse is that the introduction of additional intermediate arrays to reduce calculations actually increases memory traffic and leads to performance degradation.

**Example: Byte ledger for vector addition**
```
// C[i] = A[i] + B[i]，N 个 float
// 读：A (4N bytes) + B (4N bytes) = 8N bytes
// 写：C (4N bytes)
// 总内存流量 = 12N bytes
// 峰值带宽 900 GB/s → 理论下界 = 12N / 900G 秒
```

**Example: Histogram’s byte ledger (easy to miscalculate)**
```
// 输入：N 个 int（读 4N bytes）
// 输出：bins[] 使用 atomicAdd
// atomic = read + modify + write → 每次 ≈ 3×4 = 12 bytes
// 总流量 = 4N + 12N = 16N bytes（而非天真以为的 4N + 4N）
```

---

### Principle B: Coalescing

The ideal memory access form: **32 threads of a warp access continuous 128 bytes** (take float as an example).

Specific requirements:
- Lanes within the warp are mapped to consecutive memory addresses
- The overall access pattern is as close to sequential streaming as possible

Merging memory fetches is a prerequisite for all other optimizations. If coalescing is not met, the upper limit of effective bandwidth will be slashed.

**Example: coalescing problem in matrix transpose**
```
// 反面：按列读取，warp 内线程访问步长为 N
out[j][i] = in[i][j]   // in 按行读（coalesced）✓
                        // out 按列写（strided）✗ → 带宽利用率骤降

// 正面：借助 shared memory 中转
tile[threadIdx.y][threadIdx.x] = in[row][col]   // coalesced 读
__syncthreads()
out[col][row] = tile[threadIdx.x][threadIdx.y]   // coalesced 写
```

**Example: AoS vs SoA**
```
// AoS（Array of Structs）— warp 读 x 时跨步为 sizeof(Point)
struct Point { float x, y, z; };
Point pts[N];            // pts[tid].x → stride=12 bytes ✗

// SoA（Struct of Arrays）— warp 读 x 时连续
float px[N], py[N], pz[N];
px[tid]                  // stride=4 bytes，完美 coalesced ✓
```

---

### Principle C: Explicit reuse (Tiling)

When the kernel has a neighborhood or data reuse structure (such as stencil, convolution, partially sparse local operator):

- Should not rely on implicit hits from hardware L1/L2 cache
- Reuse should be converted into deterministic behavior through tiling (shared memory or register)

Core idea: **Load data from HBM to SRAM and reuse it multiple times to avoid repeated HBM access**.

> [!info]What is Stencil?
> Stencil (template calculation) is a common calculation mode: each output element is obtained by the weighted sum of itself and the input elements in a fixed neighborhood. A 1D stencil of radius R means that `out[i]` depends on `in[i-R] ... in[i+R]` for a total of 2R+1 elements. Typical applications include finite differences (CFD/PDE solving), image blurring/sharpening (2D stencil), audio filtering, etc. Since the input windows of adjacent output points highly overlap, stencil is a classic scenario for tiling optimization.

**Example: 1D Stencil — no tiling vs with tiling**
```
// 无 tiling：每个输出点从 HBM 读 2R+1 个邻居
// 相邻线程的读取大量重叠 → 依赖 cache 命中，不可控
out[i] = Σ w[k] * in[i-R+k],  k=0..2R

// 有 tiling：block 协作加载一段 tile（含 halo）到 shared memory
__shared__ float tile[BLOCK + 2*R];
tile[threadIdx.x + R] = in[blockStart + threadIdx.x];
if (threadIdx.x < R) {               // 加载左右 halo
    tile[threadIdx.x] = in[blockStart - R + threadIdx.x];
    tile[BLOCK + R + threadIdx.x] = in[blockStart + BLOCK + threadIdx.x];
}
__syncthreads();
out[i] = Σ w[k] * tile[threadIdx.x + k];  // 全部命中 SRAM
```

---

### Principle D: Reduce synchronization and contention (Sync/Contention)

The performance bottleneck of Memory-bound kernel is often not the bandwidth itself, but rather:
- `__syncthreads()` is called too frequently, converting pipeline throughput into serial waits
- Atomic operation competition (histogram, scatter), degenerating parallel writing into serial writing

The "hierarchical privatization" in the Histogram in the previous lecture is a typical application of this principle: gradually reducing the competition scope from global to block and then to warp, thereby reducing the contention overhead.

**Example: Histogram Hierarchical Privatization**
```
// 级别 1 — 全局 atomic（最大争用）
atomicAdd(&global_bins[val], 1);           // 所有线程竞争同一组 bins

// 级别 2 — block 私有 bins → 最后归约
__shared__ int local_bins[NUM_BINS];       // 每个 block 一份
atomicAdd(&local_bins[val], 1);            // 争用缩小到 block 内
__syncthreads();
atomicAdd(&global_bins[tid], local_bins[tid]);  // 一次性归约

// 级别 3 — warp 私有 bins（寄存器/shared memory 分区）
// 争用进一步缩小到 32 个线程内，几乎无冲突
```

---

### Principle E: Latency Hiding

When high latency of memory access cannot be avoided (such as random access of SpMV), the latency can be hidden by:
- **Improve occupancy**: Increase the number of warps resident at the same time so that more warps can be scheduled for execution during memory waiting
- **Increase ILP (Instruction Level Parallelism)**: Through unroll and one-thread multi-element strategy, a single thread can initiate more load requests at the same time

Grid-stride loops are a common engineering means of implementing this principle.

**Example: Grid-stride loop + ILP unroll**
```
// 基础版：每线程处理一个元素，occupancy 是唯一延迟隐藏手段
for (int i = tid; i < N; i += gridDim.x * blockDim.x)
    out[i] = f(in[i]);

// ILP 版：每线程同时发起多个 load，隐藏内存延迟
for (int i = tid; i < N; i += stride * 4) {
    float a = in[i];
    float b = in[i + stride];
    float c = in[i + stride*2];
    float d = in[i + stride*3];    // 4 个 load 同时 in-flight
    out[i]            = f(a);
    out[i + stride]   = f(b);
    out[i + stride*2] = f(c);
    out[i + stride*3] = f(d);
}
```

---

## 3. Pattern Library: Six types of Memory-Bound Kernel

The following summarizes memory-bound kernels into six categories according to memory access modes. When faced with a new kernel, first determine its category, and then optimize it according to the corresponding principles.

---

### Pattern 1: Streaming (linear reading and writing)

**Typical scenario**: `out[i] = f(in[i])`, elementwise map, vector addition

**Memory access features**: read input sequentially, write output sequentially, no data reuse.

**Applicable principles**: B (coalescing) + E (delay hiding)

**Optimization methods**: vectorized load/store (float4), memory alignment, grid-stride loop, loop expansion

```cpp
// Vectorized elementwise kernel
// 使用 float4 进行向量化加载和存储，每次内存事务搬运 16 字节
__global__ void vector_add_v4(
    const float4* __restrict__ a,
    const float4* __restrict__ b,
    float4* __restrict__ c,
    int n4  // n / 4，即 float4 元素个数
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

//// float4 的定义（大致）
// struct float4 {
//    float x, y, z, w;
//};

// 启动配置
// n 为 float 元素总数，需为 4 的倍数（否则需额外处理尾部）
// vector_add_v4<<<(n/4 + 255) / 256, 256>>>(a4, b4, c4, n/4);
```

> [!note]Optimization principle of float4
> After using float4, each load/store instruction moves 16 bytes instead of 4 bytes. This reduces the total number of load/store instructions required, allowing the compiler to more efficiently schedule the instruction pipeline and improve instruction-level parallelism (ILP). When 32 threads of a warp execute float4 load at the same time, four 128B memory transactions are generated, and a total of 512 bytes are transferred.

---

### Pattern 2: Reorder/Permutation (reorder class)

**Typical Scenario**: Matrix Transpose

**Core Contradiction**: The reading direction is orthogonal to the writing direction. If the read is coalesced, the write must be strided, and vice versa.

**Solution**: Use shared memory as intermediate buffer. When reading, coalesce is loaded into shared memory in the row direction, and when writing, coalesce is written out from shared memory in the column direction.

**Applicable principles**: B (coalesce is required on both reading and writing ends) + C (shared memory is used as a rearrangement cache) + D (avoiding bank conflict)

```cpp
// Shared memory tiled 矩阵转置
// 输入 [M x N]，输出 [N x M]
#define TILE_DIM 32
#define BLOCK_ROWS 8  // block 尺寸: TILE_DIM x BLOCK_ROWS
                      // 每个 block 处理 TILE_DIM x TILE_DIM 的 tile

__global__ void transpose_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    // 列数 +1 作为 padding，消除 bank conflict（详见下方说明）
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Step 1: 按行方向从 global memory 加载到 shared memory（coalesced read）
    // 每个线程负责加载 TILE_DIM / BLOCK_ROWS = 4 行
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < N && (y + j) < M) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * N + x];
        }
    }

    __syncthreads();

    // Step 2: 坐标互换后，按行方向从 shared memory 写入 global memory（coalesced write）
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < M && (y + j) < N) {
            output[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// 启动配置
// dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
// dim3 block(TILE_DIM, BLOCK_ROWS);
// transpose_optimized<<<grid, block>>>(input, output, M, N);
```

> [!important]Padding eliminates bank conflict
> Shared memory consists of 32 banks, each bank is 4 bytes wide. If the number of columns of a tile is exactly 32, all elements of the same column will be mapped to the same bank, resulting in a 32-way bank conflict when accessing in the column direction. After setting the number of columns to `TILE_DIM + 1 = 33`, elements at the same column position in adjacent rows are staggered by one bank, thereby eliminating conflicts. This technique is widely used in kernels involving shared memory column accesses.

---

### Pattern 3: Stencil / Neighborhood (neighborhood reuse class)

**Typical scenarios**: 1D/2D stencil, image convolution, image processing filter

**Core Feature**: Each input element is multiplexed by multiple adjacent output points. Taking the 1D 3-point stencil as an example, each input element is read once by the three output points of the left, middle and right.

**Applicable principles**: C (tile + halo achieves deterministic reuse) + B (halo maintains coalescing when loaded)

```cpp
// 1D Stencil: out[i] = c0*in[i-R] + c1*in[i-R+1] + ... + c2R*in[i+R]
// R = stencil 半径，本例取 R = 4（9-point stencil）

#define RADIUS 4
#define BLOCK_SIZE 256
// 每个 block 计算 BLOCK_SIZE 个输出点
// 需加载 BLOCK_SIZE + 2*RADIUS 个输入点（含两端 halo 区域）

__constant__ float coeff[2 * RADIUS + 1];  // stencil 系数存入 constant memory

__global__ void stencil_1d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    __shared__ float smem[BLOCK_SIZE + 2 * RADIUS];

    int gidx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int lidx = threadIdx.x + RADIUS;  // shared memory 内的偏移索引

    // Step 1: 加载中间区域
    smem[lidx] = (gidx < n) ? input[gidx] : 0.0f;

    // Step 2: 加载左侧 halo（前 RADIUS 个线程负责）
    if (threadIdx.x < RADIUS) {
        int halo_idx = gidx - RADIUS;
        smem[threadIdx.x] = (halo_idx >= 0) ? input[halo_idx] : 0.0f;
    }

    // Step 3: 加载右侧 halo（后 RADIUS 个线程负责）
    if (threadIdx.x >= BLOCK_SIZE - RADIUS) {
        int halo_idx = gidx + RADIUS;
        smem[lidx + RADIUS] = (halo_idx < n) ? input[halo_idx] : 0.0f;
    }

    __syncthreads();

    // Step 4: 从 shared memory 计算 stencil，无 global memory 访问
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

| Version | Global Memory Read Volume | Description |
|------|---------------------|------|
| Naive (no shared memory) | $N \times (2R+1)$ | Each output point reads $2R+1$ inputs from HBM |
| Tiled (using shared memory) | $N + 2R \times \text{num\_blocks}$ | Each input element is basically loaded from HBM only once |

When the stencil radius $R$ is larger, the degree of data reuse is higher and the benefits of tiling are more significant.

---

### Pattern 4: Indirection / Gather (irregular reading side)

**Typical scenario**: CSR format SpMV, gather, embedding lookup

**Core difficulty**: Indirect addressing `x[col_idx[j]]` causes the access address to depend on the data content, making it difficult to achieve ideal coalescing.

**Applicable principles**: A (Incorporate the byte overhead of index into the calculation) + E (Hide latency through warp-level mapping)

Sparse matrix-vector multiplication (SpMV)

```
CSR (Compressed Sparse Row) 用三个数组存储稀疏矩阵:
═══════════════════════════════════════════════════════════════════════════════

1. values[nnz]:   所有非零值，按行存储
2. col_idx[nnz]:  每个非零值对应的列号
3. row_ptr[M+1]:  每行在 values/col_idx 中的起始位置

对于上面的矩阵:

values[]:   [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
             ↑       ↑   ↑   ↑   ↑           ↑   ↑
            row0    row0 row1 row1 row2      row2 row3

col_idx[]:  [ 0,  2,  4,  1,  5,  0,  1,  2,  3,  5]
             对应每个非零元素的列号

row_ptr[]:  [ 0,  3,  5,  9, 10]
              ↑   ↑   ↑   ↑   ↑
             row0 row1 row2 row3 结束
             起始 起始 起始 起始

row_ptr[i] 到 row_ptr[i+1] 之间的索引就是第 i 行的非零元素
```


warp-per-row strategy
```
核心思想: 一个 Warp (32个线程) 协作处理矩阵的一行
═══════════════════════════════════════════════════════════════════════════════

为什么不用 Thread-per-Row？
─────────────────────────────
如果一行有很多非零元素（比如 1000 个），单线程串行处理太慢

Warp-per-Row 的优势:
─────────────────────────────
1. 32 个线程并行处理一行的非零元素
2. 访问 values[] 和 col_idx[] 时地址连续 → Coalesced Access
3. 使用 Warp Shuffle 归约，不需要 Shared Memory
```

```cpp
// CSR 格式 SpMV: y = A * x
// CSR 存储: row_ptr[M+1], col_idx[nnz], values[nnz]
//
// 映射策略:
//   行长度较均匀: thread-per-row
//   行长度差异大: warp-per-row（本例采用此策略）

// Warp-per-row: 每个 warp（32 线程）处理矩阵的一行
// 优势:
//   1. warp 内线程访问连续的 col_idx 和 values（coalesced）
//   2. 使用 warp shuffle 归约，无需 shared memory 或 atomic
//   3. 32 个线程同时发起 load，有效隐藏延迟
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

    // warp 内 32 个线程以 stride 32 遍历该行的非零元素
    for (int j = row_start + lane; j < row_end; j += 32) {
        // col_idx[j], values[j]: 连续访问，coalesced
        // x[col_idx[j]]: 随机访问，依赖 L2 cache 和延迟隐藏
        sum += values[j] * x[col_idx[j]];
    }

    // Warp-level 归约（warp shuffle），无需 shared memory 或 barrier
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[warp_id] = sum;
    }
}

// 启动配置
// int threads_per_block = 256;  // 每个 block 包含 8 个 warp
// int num_blocks = (num_rows * 32 + threads_per_block - 1) / threads_per_block;
// spmv_csr_warp_per_row<<<num_blocks, threads_per_block>>>(...);
```

If you need all lanes to get the results, you can use __shfl_sync to broadcast, or use __shfl_xor_sync to do butterfly reduction. But here you only need to write a y[warp_id], so it is enough that Lane 0 has the result.

> [!note]Bandwidth upper bound for sparse kernels
> For sparse kernels, the actual achievable bandwidth upper limit is often lower than the theoretical peak value of HBM. The access pattern of `x[col_idx[j]]` is determined by the sparse structure of the matrix and cannot be fully controlled at the kernel level. The optimization direction therefore turns to "reduce the randomness of access", such as reordering matrix columns to improve the access locality of vector x.

---

### Pattern 5: Scatter / Atomic (write-side irregularity + contention)

**Typical scenarios**: histogram, scatter-add, partial graph algorithm

**Core Difficulty**: The write address depends on the data content, and hot targets (such as high-frequency bins) cause atomic operations to be severely serialized.

**Applicable principles**: D (hierarchical privatization)

The optimization method of this type of kernel has been demonstrated in detail in the three versions of Histogram in the previous lecture. The core strategy is as follows:

```
Level 0: Global atomic — 所有线程竞争全局内存
  ↓ 私有化
Level 1: Block 级 shared memory atomic — 竞争范围缩小至 256 线程
  ↓ 进一步私有化
Level 2: Warp 级 / 寄存器本地累积 — 竞争缩小至 32 线程或完全消除
  ↓ 最终归约
写回 global memory — atomic 调用次数从 N 降至 num_bins × num_blocks
```

---

### Pattern 6: Filter / Compaction (conditional filtering class)

**Typical scenarios**: stream compaction, zero removal operation, predicate filter

#### What is Stream Compaction?

**Filter out elements that meet the conditions from the array and store them compactly**

```
输入:  [ 3, -1, 4, 0, -2, 5, 0, 1 ]
条件:  元素 > 0
输出:  [ 3, 4, 5, 1 ]
```

#### Why is it difficult to parallelize?

Each thread does not know its own output location - **solve with Exclusive Prefix Sum**:

```
flag:   [ 1,  0,  1,  0,  0,  1,  0,  1 ]   ← 标记满足条件的
scan:   [ 0,  1,  1,  2,  2,  2,  3,  3 ]   ← exclusive scan

scan[i] = "在我之前有几个满足条件的" = 我的输出位置
```

#### Three-step process

| Steps | What to do | Code |
|-----|-------|------|
| **Flag** | Mark elements that meet the condition | `flag = (val > 0) ? 1 : 0` |
| **Scan** | Exclusive prefix sum (Blelloch algorithm) | Up-sweep + Down-sweep |
| **Scatter** | Write to the correct position | `output[offset + scan[tid]] = val` |


```cpp
// Stream Compaction: 筛选 input 中 > 0 的元素，紧凑写入 output
// Fused 实现: 在单个 kernel 内完成 flag、block-level scan、scatter

#define BLOCK_SIZE 256

__global__ void stream_compaction(
    const int* __restrict__ input,
    int* __restrict__ output,
    int* __restrict__ output_count,  // 输出的元素总数
    int n
) {
    __shared__ int scan[BLOCK_SIZE];
    __shared__ int block_output_offset;

    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    // Step 1: Flag — 标记满足条件的元素
    int val = 0;
    int flag = 0;
    if (gid < n) {
        val = input[gid];
        flag = (val > 0) ? 1 : 0;
    }
    scan[tid] = flag;
    __syncthreads();

    // Step 2: Block-level exclusive scan（Blelloch 算法）

    // Up-sweep
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        int ai = (tid + 1) * offset * 2 - 1;
        if (ai < BLOCK_SIZE) {
            scan[ai] += scan[ai - offset];
        }
        __syncthreads();
    }

    // 提取 block 内满足条件的元素总数，并通过一次 atomic 分配输出空间
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

    // Step 3: Scatter — 将满足条件的元素写入输出数组的正确位置
    if (gid < n && flag) {
        output[block_output_offset + scan[tid]] = val;
    }
}
```

> [!tip]Key optimization
> `atomicAdd(output_count, block_total)` is called num_blocks instead of N. This is a combination of principle D (reduce contention) and principle A (reduce atomic memory overhead): each block internally calculates local offsets through scan, and finally only needs one atomic operation to allocate output space.

---

## 4. Optimize Checklist

When optimizing the memory-bound kernel, it is recommended to check in the following order:

| Steps | Contents | Observation methods |
|------|------|----------|
| 1 | Calculate arithmetic intensity and bandwidth upper limit | Roofline model, manual calculation |
| 2 | Measure effective bandwidth | Bytes / kernel_time, compared with theoretical peak value |
| 3 | Check merged memory fetches | ncu: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` |
| 4 | Check contention and synchronization overhead | ncu: atomic throughput, barrier wait time |
| 5 | Adjust occupancy and ILP | grid-stride loop, loop expansion, multiple elements in one thread |

The importance of this order is that if coalescing is not satisfied and occupancy is adjusted, the benefits obtained will be very limited. You should first ensure that the memory access mode is correct before performing higher-level tuning.

---

## 5. Summary

| Principles | Optimization goals | Typical applicable kernel |
|------|----------|----------------|
| A-byte ledger | Accurately quantify memory traffic | All kernels |
| B merged memory access | Continuous address access within warp, reducing the number of memory transactions | streaming, transpose |
| C explicit multiplexing | Data is loaded from HBM to SRAM and multiplexed multiple times | stencil, partial sparse |
| D Reduce contention | Reduce the serialization overhead of barrier and atomic | histogram, scatter, compaction |
| E latency hiding | Masking memory latency through parallelism and ILP | sparse, gather |

The optimization of Reduce, Histogram, and Scan in the first two lectures has covered all the above principles (coalescing, first add during load, warp shuffle, ILP, grid-stride loop). The purpose of this lecture is to abstract these techniques scattered in specific examples into a general analysis framework.

When facing a new memory-bound kernel, first determine the Pattern (1-6) it belongs to, and then formulate an optimization strategy based on the corresponding principles.
