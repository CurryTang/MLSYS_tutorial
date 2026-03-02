# CuTe — CUDA Tensors 完全教程

> **基于 Cris Cecka (NVIDIA Principal Research Scientist) 的 GTC 演讲资料**
> 本文是对 NVIDIA CUTLASS 库中 CuTe 子系统的深度解析，涵盖基础概念、布局代数、算法实现与 MMA 原子操作。

---

## 目录

- [[#0. 前言：CUDA 的痛点与 CUTLASS 的诞生]]
- [[#1. CuTe 是什么？]]
- [[#2. 历史背景与动机]]
- [[#3. CUTLASS 4.0 —— Python 时代的到来]]
- [[#4. 核心概念：Layout = 表示 + 代数]]
- [[#5. 循环哲学：从 for 循环到 Layout]]
- [[#6. Layout 表示：从坐标到索引]]
- [[#7. 层次化 Layout 与张量折叠]]
- [[#8. Layout 兼容性：Shape 的偏序关系]]
- [[#9. 为什么需要 CuTe？]]
- [[#10. 类型系统与概念]]
- [[#11. 算法：Copy 与 GEMM]]
- [[#12. Layout 组合 (Composition)]]
- [[#13. 组合的力量：线程-值分区]]
- [[#14. MMA 原子操作]]
- [[#15. TiledMMA 与 ThrMMA]]
- [[#16. 总结与资源]]

---

## 0. 前言：CUDA 的痛点与 CUTLASS 的诞生

### 0.1 原始 CUDA 编程的困境

GPU 的计算能力极强，但高效地利用这些算力并不容易。原始 CUDA C++ 编程面临几个根本性挑战：

**性能陷阱无处不在**

GPU 性能高度依赖内存访问模式。以最常见的矩阵乘法（GEMM）为例：

```
朴素实现（naive GEMM）:
  每个线程独立从全局内存（HBM）读取数据
  → 大量重复读取，带宽利用率极低
  → 实际性能可能只有理论峰值的 1%~5%

高性能实现需要:
  1. Tiling：将数据分块加载到共享内存（SRAM）
  2. Double Buffering：计算与数据加载重叠
  3. Warp-level / Instruction-level 并行
  4. 向量化内存访问（128-bit load/store）
  5. 张量核心（Tensor Core / MMA 指令）
```

每一层优化都要求程序员**手动管理坐标变换**——从线程 ID 到数据索引的映射极其繁琐，且极易出错。

**索引地狱**

一个典型的高性能 GEMM 内核中，光是计算"当前线程负责哪块数据"就需要数十行索引算术：

```cpp
// 一段真实的 CUTLASS 2.x 风格索引代码（简化）
int warp_id = threadIdx.x / 32;
int lane_id = threadIdx.x % 32;
int warp_m  = warp_id / WarpCount_N;
int warp_n  = warp_id % WarpCount_N;
int lane_m  = lane_id / ThreadsPerWarp_N;
int lane_n  = lane_id % ThreadsPerWarp_N;

int global_m = blockIdx.x * BlockTile_M + warp_m * WarpTile_M + lane_m * ThreadTile_M;
int global_n = blockIdx.y * BlockTile_N + warp_n * WarpTile_N + lane_n * ThreadTile_N;
// ... 还没算 K 维度，还没处理边界，还没考虑 swizzle ...
```

这类代码：**可读性差、不可复用、换个 GPU 架构就要重写**。

---

### 0.2 CUTLASS 是什么？

**CUTLASS**（CUDA Templates for Linear Algebra Subroutines and Solvers）是 NVIDIA 开源的高性能线性代数库，其定位是：

> **介于 cuBLAS（黑盒）和原始 CUDA（白盒）之间的透明高性能实现。**

| | cuBLAS | CUTLASS | 原始 CUDA |
|---|---|---|---|
| 性能 | 接近峰值 | 接近峰值 | 取决于实现 |
| 可定制性 | 几乎没有 | 高度可定制 | 完全自由 |
| 代码可读性 | 不开源 | 中等（C++模板） | 自己决定 |
| 适合场景 | 标准 GEMM | 融合算子、非标准布局 | 全新算法原型 |

CUTLASS 的核心价值在于**可定制性**：当你需要把 GEMM 和 epilogue（如 bias add、activation）融合到一个 kernel 时，cuBLAS 无能为力，而 CUTLASS 提供了清晰的分层抽象来支持这类需求。

**CUTLASS 解决的核心问题**：

1. **布局多样性**：支持任意步长（stride）的矩阵、填充（padding）、转置、批量（batched）操作
2. **精度灵活性**：FP32、FP16、BF16、INT8、FP8 等混合精度
3. **架构适配**：自动利用 Volta/Ampere/Hopper/Blackwell 的张量核心指令
4. **算子融合**：在一个 kernel 内完成 GEMM + 任意 epilogue

---

### 0.3 CUTLASS 的分层架构

CUTLASS 采用严格的分层设计，从高层到底层：

```
┌──────────────────────────────────────────────────────┐
│  Device Layer                                        │
│  GemmUniversal, Conv2d, ...                          │
│  用户调用的接口，处理 grid/block 配置、workspace 分配  │
├──────────────────────────────────────────────────────┤
│  Kernel Layer                                        │
│  GemmUniversal_Kernel, ...                           │
│  单个 CUDA kernel 的实现骨架                          │
├──────────────────────────────────────────────────────┤
│  Collective Layer (Mainloop + Epilogue)               │
│  CollectiveMma, CollectiveEpilogue                   │
│  协作性的 warp/threadblock 级别操作                   │
├──────────────────────────────────────────────────────┤
│  CuTe Layer                                          │
│  Layout, Tensor, TiledCopy, TiledMMA                 │
│  ← 本教程的主题                                       │
└──────────────────────────────────────────────────────┘
```

---

### 0.4 CUTLASS 与 CuTe 的关系

**CuTe 是 CUTLASS 的底层语言，CUTLASS 是 CuTe 的上层应用。**

```
CuTe                          CUTLASS
────────────────────          ──────────────────────────────
提供：                         使用：
  Layout 代数                    用 Layout 描述矩阵分块方式
  Tensor 抽象                    用 Tensor 表示共享内存/寄存器
  TiledCopy                     用 TiledCopy 实现 gmem→smem 搬运
  TiledMMA                      用 TiledMMA 实现张量核心计算
  分区原语（partition）           用分区原语分配线程工作
```

换句话说：

- **没有 CuTe**：CUTLASS 的每个新功能都需要手写复杂的索引代码（CUTLASS 2.x 的现状）
- **有了 CuTe**：通过 Layout 代数自动推导所有索引，代码量减少 90%，且正确性有保证（CUTLASS 3.x 的方式）

**一个类比**：CuTe 之于 CUTLASS，类似于 NumPy 之于 PyTorch——NumPy 提供了张量操作的基础语义，PyTorch 在其上构建深度学习框架。但 CuTe 更底层：它在 CUDA 线程级别工作，并且同时用来描述**数据布局**和**线程布局**，这是其独特之处。

---

### 0.5 为什么 GPU 编程需要 Layout 抽象？

GPU 编程中有一个核心挑战：**同一份逻辑数据，需要以不同的物理布局在不同层次的内存之间流动**。

```
HBM（全局内存）中:  行主序，128字节对齐
      ↓ TMA/cp.async 搬运
SMEM（共享内存）中: 列主序 + swizzle（避免 bank conflict）
      ↓ ldmatrix 指令加载
寄存器中:          碎片化布局（由 MMA 指令决定，非连续）
      ↓ mma.sync 计算
寄存器中:          同上
      ↓ stmatrix / store 写回
HBM（全局内存）中:  行主序，128字节对齐
```

每次数据格式转换都需要正确的索引映射。没有 CuTe 的 Layout 抽象，程序员需要为每种情况手写映射函数；有了 CuTe，所有映射都通过 Layout 组合代数自动完成。

> [!note] 本文的范围
> 接下来的内容将深入 CuTe 的设计细节。理解 CuTe 不仅能让你读懂 CUTLASS 3.x 的源码，还能让你具备从零开始编写高性能 CUDA kernel 的能力。

---

## 1. CuTe 是什么？

**CuTe** 是 **CU**DA **Te**nsors 的缩写，是 NVIDIA CUTLASS 库的核心底层抽象。它提供了一套**统一的张量编程模型**，用于在 GPU 上高效地表示和操作多维数据。

CuTe 的核心思想可以用一句话概括：

> **Layout 既是一种表示（Representation），也是一种代数（Algebra）。**

作为**表示**：
- 超越行主序和列主序的通用数据布局描述
- 支持 copy、gemm、reduce 等通用算法
- 封装元数据（metadata）

作为**代数**：
- 通过组合 Layout 创建新的 Layout
- 支持通用的分块（tiling）与分区（partitioning）操作

---

## 2. 历史背景与动机

### 2.1 CuTe 的演化之路

| 阶段 | 关键事件 |
|------|---------|
| FFT 研究 | BLAS 扩展用于张量收缩 |
| 低秩 ML 研究 | "张量收缩就是 GEMM，为什么这么难？" |
| cuTensor 合作 | 与 Paul Springer 合作，解决 CUTLASS 2.x 痛点 |
| "这不应该很难" | 支持 Volta、Ampere、Hopper 架构 |
| CUTLASS 3.x | 由 Vijay Thakkar 主导，支持 Hopper/Blackwell + TMA |

### 2.2 C++ 模板的痛点

CuTe 诞生的直接动机来自 C++ 模板编程的诸多问题：

1. **模板不便利**：编译期逻辑增加心智负担，错误信息冗长如小说
2. **编译时间慢**：前端过于通用，阻碍快速迭代；禁止大规模 JIT 和暴力自动调优
3. **生态割裂**：深度学习领域全面拥抱 Python；所有人都讨厌写绑定代码
4. **nvcc 依赖**：LLM 更擅长生成 Python 程序

---

## 3. CUTLASS 4.0 —— Python 时代的到来

### 3.1 架构层次

CUTLASS 4.0 引入了 Python 前端，初始版本开放了 CuTe 层：

```
┌─────────────────────────────────────────┐
│     Device Layer (Gemm Universal, Conv)  │  ← 更多预调优配方
├─────────────────────────────────────────┤
│         CUTLASS Kernel Layer             │
├─────────────────────────────────────────┤
│          Collective Layer                │
│  SM90 TMA SS | SM100 TMA | SM100 ATMEM  │
├─────────────────────────────────────────┤
│     CuTe: Tiled Copy & Tiled MMA        │  ← Python 首发层
├─────────────────────────────────────────┤
│            CuTe Atoms                    │  ← 更多控制
│  TMEM load | TMEM store | TMA | MMA     │
└─────────────────────────────────────────┘
```

### 3.2 性能与编译速度

Python 版 CuTe 的两大优势：

- **峰值性能**：8k×8k×8k GEMM 达到与 C++ 相同的 %SOL（接近 100%）
- **编译速度**：比 C++ 快 **100 倍以上**（C++: ~28000ms → Python: ~241ms）

### 3.3 Hello World

```python
pip install nvidia-cutlass-dsl

import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel():
    tidx, _, _ = cutlass.arch.thread_idx()
    if tidx == 0:
        cute.print_("Hello world")

@cute.jit
def host():
    cutlass.cuda.initialize_cuda_context()
    kernel().launch(
        grid=(1,1,1),
        block=(32,1,1))

host()
```

CuTe Python 提供的核心 API 包括：
- `make_shape`, `make_layout`, `make_identity_tensor`
- `zipped_divide`, `local_tile`
- `tiled_mma.get_slice`, `thr_mma.partition_A`, `thr_copy.partition_S`

---

## 4. 核心概念：Layout = 表示 + 代数

### 4.1 Layout 的本质

CuTe 中的 **Layout** 是一个从**逻辑坐标（coordinate）**到**物理索引（index）**的函数：

$$\text{idx} = \text{inner\_product}(\text{coord}, \text{stride})$$

一个 Layout 由两部分组成：
- **Shape**：定义坐标空间（每个维度的大小）
- **Stride**：定义索引映射（每个维度相邻元素的间隔）

### 4.2 经典布局示例

考虑一个 2×3 的逻辑矩阵 `[[a,b,c],[d,e,f]]`：

**列主序 (Column-major)**：
```
Shape:  (2, 3)
Stride: (1, 2)
idx = i*1 + j*2

逻辑:  a b c    物理: a d b e c f
       d e f
```

**行主序 (Row-major)**：
```
Shape:  (2, 3)
Stride: (3, 1)
idx = i*3 + j*1

逻辑:  a b c    物理: a b c d e f
       d e f
```

**填充的列主序 (Padded Column-major)**：
```
Shape:  (2, 3)
Stride: (1, 4)
idx = i*1 + j*4

逻辑:  a b c    物理: a d _ _ b e _ _ c f
       d e f
```

**张量布局 (Tensor layout)**：
```
Shape:  (2, 2, 2)
Stride: (4, 1, 2)
idx = inner_product(coord, stride)

逻辑:              物理:
(_,_,0): a b       a b e f c d g h
         c d
(_,_,1): e f
         g h
```

### 4.3 Tensor 是视图，不是容器

```cpp
// Tensor 是一个 (指针, Layout) 的组合
Tensor matrix = make_tensor(storage.data(), layout);

// 关键属性：
// rank:   模式（mode）的数量
// size:   某个模式的大小
// stride: 某个模式中相邻元素的间隔
```

---

## 5. 循环哲学：从 for 循环到 Layout

CuTe 的一个核心洞察是：**任何线性仿射循环都可以用 (base-ptr, shape, stride) 表示**。

### 5.1 一维循环

```cpp
// 原始循环
for (int i = 2; i <= 50; i += 3) {
    A[7*i + 5] = 0;
}

// 规范化后
for (int i = 0; i < 17; ++i) {
    (A+19)[21*i] = 0;
}
```

提取参数：
```
Base-ptr: A+19
Size:     17
Stride:   21
```

### 5.2 二维循环

```cpp
// 原始循环
for (int j = 3; j < 43; j += 2) {
    for (int i = 4; i <= 20; i += 5) {
        A[10*i - j + 1] = 0;
    }
}

// 规范化后
for (int j = 0; j < 20; ++j) {
    for (int i = 0; i < 4; ++i) {
        (A+38)[50*i - 2*j] = 0;
    }
}
```

提取参数：
```
Base-ptr: A+38
Shape:    (4,  20)
Stride:   (50, -2)
```

> [!tip] 关键洞察
> 循环的起始值被吸收到基指针（base pointer）中，步长被吸收到 stride 中，迭代次数成为 shape。这正是 Layout 的抽象核心。

---

## 6. Layout 表示：从坐标到索引

### 6.1 三层映射

CuTe 中的数据访问涉及三层映射：

```
逻辑 1-D 坐标  →  逻辑 n-D 坐标  →  逻辑 h-D 坐标  →  物理存储索引
   A(I)              A(i,j)          A(i,(j₀,j₁))         A[k]
                  ←坐标映射→        ←坐标映射→         ←索引映射→
```

- **Shape** 定义坐标映射：$I \Leftrightarrow (i,j) \Leftrightarrow (i,(j_0,j_1))$
- **Stride** 定义索引映射：$(i,(j_0,j_1)) \rightarrow [k]$

### 6.2 具体示例

```
Shape:  (4, (2,2))
Stride: (2, (1,8))
```

| 1-D 坐标 | n-D 坐标 | h-D 坐标 | 物理索引 |
|----------|----------|----------|---------|
| 0 | (0,0) | (0,(0,0)) | 0 |
| 1 | (0,1) | (0,(1,0)) | 1 |
| 2 | (0,2) | (0,(0,1)) | 8 |
| 3 | (0,3) | (0,(1,1)) | 9 |
| 4 | (1,0) | (1,(0,0)) | 2 |
| 5 | (1,1) | (1,(1,0)) | 3 |
| ... | ... | ... | ... |

### 6.3 基础代码示例

**基本创建**：
```cpp
int n_rows = 22;
int n_cols = 19;
thrust::host_vector<float> storage(n_rows * n_cols);

Layout layout = make_layout(make_shape(n_rows, n_cols));
Tensor matrix = make_tensor(storage.data(), layout);

static_assert(rank(matrix) == 2);

// print(matrix) 输出:
// ptr[32b](0x...) o (22,19):(_1,22)
```

**静态整数**：
```cpp
auto n_rows = Int<22>{};  // 编译期已知
int  n_cols = 19;          // 运行时

Layout layout = make_layout(make_shape(n_rows, n_cols));
Tensor matrix = make_tensor(storage.data(), layout);

static_assert(size<0>(matrix) == 22);  // 编译期断言成功！

// print(matrix) 输出:
// ptr[32b](0x...) o (_22,19):(_1,_22)
//                    ^^^       ^^^
//                    带下划线表示编译期已知
```

**自定义步长**：
```cpp
auto n_rows = Int<22>{};
int  n_cols = 19;
int  d_rows = 47;
auto d_cols = Int<2>{};

Layout layout = make_layout(make_shape (n_rows, n_cols),
                            make_stride(d_rows, d_cols));

// print(matrix) 输出:
// ptr[32b](0x...) o (_22,19):(47,_2)
```

步长的便捷设置方式：
| 函数 | 含义 |
|------|------|
| `make_stride(...)` | 自定义步长 |
| `LayoutLeft{}` | 列主序（默认） |
| `LayoutRight{}` | 行主序 |

---

## 7. 层次化 Layout 与张量折叠

### 7.1 张量折叠：多种视图

考虑一个 2×2×2 的张量：

```
Shape:  (2,2,2)
Stride: (4,1,2)

逻辑排列:
(_,_,0):  a b     (_,_,1):  e f
          c d               g h

物理存储: a b e f c d g h
```

**2×4 视图**（自然展平）：
```
Shape:  (2, 4)
Stride: (4, 1)

a b e f
c d g h  →  a b e f c d g h
```

**4×2 视图**（需要嵌套 Layout）：
```
a b
c d      不能用简单的 Shape: (4,2), Stride: (?,1) 表示！
e f
g h
```

解决方案——**嵌套 Layout**：
```
Shape:  ((2,2), 2)
Stride: ((4,2), 1)

行坐标映射:
0 → (0,0)
1 → (1,0)
2 → (0,1)
3 → (1,1)

idx = inner_product(coord, stride)
```

> [!important] 核心洞察
> 嵌套 Shape 允许表达非连续的维度折叠，这是 CuTe 处理复杂张量布局（如张量核心要求的布局）的关键能力。

### 7.2 通用张量折叠：所有 GETT 都是 GEMM

所有张量收缩（tensor contractions）都可以映射到一个规范形式——**批量 GEMM**：

$$C_{mn\ell p} = A_{mnkpr} B_{\ell nrk}$$

通过按类型分组模式并定义多模式（multi-modes）：

| 模式类型 | 名称 | 出现在 |
|---------|------|--------|
| m-modes | "行模式" | C & A & !B |
| n-modes | "列模式" | C & !A & B |
| k-modes | "归约模式" | !C & A & B |
| p-modes | "批次模式" | C & A & B |

重写为：$C_{(\hat{m})(\hat{n})(\hat{p})} = A_{(\hat{m})(\hat{k})(\hat{p})} B_{(\hat{k})(\hat{n})(\hat{p})}$

---

## 8. Layout 兼容性：Shape 的偏序关系

### 8.1 定义

Shape $A$ 与 Shape $B$ **兼容** 当且仅当 $A$ 的所有坐标列表也是 $B$ 的坐标列表。记作 $A \leq B$。

### 8.2 性质

- **自反性**：$A \leq A$
- **反对称性**：若 $A \leq B$ 且 $B \leq A$，则 $A = B$
- **传递性**：若 $A \leq B$ 且 $B \leq C$，则 $A \leq C$
- **最小元素**：对于整数 $n$ 和 Shape $S$（$n = |S|$），有 $n \leq S$
- **最大元素**：如果 Shape $P$ 的所有元素都是 1，则不存在 $A \neq P$ 使得 $P \leq A$

### 8.3 36 的 Shape 分解

以 36 为例，展示兼容的 Shape 层次：

```
(36)
├── (1,36) (2,18) (3,12) (4,9) (6,6) (9,4) ...
│   ├── ((1,6),6) ((2,3),6) ((3,2),6) ...
│   │   ├── ((2,3),(2,3)) ((3,2),(2,3)) ...
│   │   │   └── ((2,(3,1,1)),(2,3)) ...
```

---

## 9. 为什么需要 CuTe？

### 9.1 CUTLASS 2.x 的问题

CUTLASS 2.x 使用大量命名类型表示不同布局：

```
RowMajor, ColMajor, RowMajorInterleaved, ColumnMajorInterleaved,
PitchLinear, TensorNCHW,
VoltaTensorOpMultiplicandCongruous,
ColumnMajorVoltaTensorOpMultiplicandCongruous,
RowMajorVoltaTensorOpMultiplicandBCongruous,
VoltaTensorOpMultiplicandCrosswise,
... 还有更多 ...
```

### 9.2 CuTe 的解决方案

```
Layout<Shape, Stride>   // 一个类型搞定一切
```

CuTe 的优势：
1. **Layout 涵盖所有迭代器**：单一实现与词汇类型
2. **形式化代数**：concatenation、composition、complement、inverse、product、divide
3. **统一线程和数据布局**：用同一套 Layout 机制描述线程映射和数据映射

### 9.3 Wild Indexing 对比

CUTLASS 2.x 的 Swizzle 索引代码（左）vs CuTe（右）：

```cpp
// CUTLASS 2.x: ~40行复杂的索引计算代码
int vec_contiguous_idx = coord.contiguous() / kElementsPerAccess;
int vec_strided_idx = coord.strided() / kFactor;
// ... 大量中间变量和取模/除法运算 ...

// CuTe: 3行
auto swizzle_atom = composition(
    Swizzle<3,3,3>{},
    Layout<Shape <Shape <_2, _4, _2>, Shape <_8, _2>>,
           Stride<Stride<_8,_64,_32>, Stride<_1,_16>>>{});
auto swizzle_layout = tile_to_shape(swizzle_atom, Shape<_128,_64>{});
```

---

## 10. 类型系统与概念

### 10.1 Shape

```
概念 HTuple<Int>: Int 或 HTuple<Int> 的 Tuple

示例:
N                           // 标量
(N, M)                      // 2-D
(N, M, P)                   // 3-D
((N0, N1), M)               // 嵌套
((N0, N1), (M0, (M10, M11, M12)))  // 深度嵌套
```

### 10.2 Stride

```
概念 HTuple<D>: D 或 HTuple<D> 的 Tuple
D: 支持与整数做内积的任意类型
必须与 Shape 同构（congruent）
```

### 10.3 Layout 与 Tensor

```cpp
Layout<Shape, Stride>
// 逻辑坐标与 D 之间的映射

Tensor<Ptr, Layout>
// Layout 与底层存储的组合
// Ptr 可以是: T[], T*, smem_ptr<T>, gmem_ptr<T>,
//            counting_iterator, transform_iterator
```

### 10.4 Layout 代数操作

| 操作 | 签名 | 用途 |
|------|------|------|
| `concatenation` | `(Layouts...) → Layout` | 拼接多个 Layout |
| `composition` | `(LayoutA, LayoutB) → Layout` | 函数组合 |
| `right_inverse` | `(Layout) → Layout` | 右逆 |
| `left_inverse` | `(Layout) → Layout` | 左逆 |
| `complement` | `(Layout, Shape) → Layout` | 补集 |
| `logical_product` | `(LayoutA, LayoutB) → Layout` | 逻辑积 |
| `logical_divide` | `(LayoutA, LayoutB) → Layout` | 逻辑除 |

**逻辑积** $f_A \otimes g_B$：
> "生成一个 Layout，其中 A 的每个元素都是一个 Layout B。"

**逻辑除** $f_A \oslash g_B$：
> "从 Layout A 中生成一个 B 的 Layout。"

---

## 11. 算法：Copy 与 GEMM

### 11.1 Copy：Rank-1 算法

```cpp
template <class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE void
copy(Tensor<SrcEngine,SrcLayout> const& src,
     Tensor<DstEngine,DstLayout>       & dst)
{
    for (int i = 0; i < size(dst); ++i) {
        dst(i) = src(i);
    }
}
```

> [!note] Copy 是 Rank-1 算法
> 无论参数张量的秩是多少，copy 总是沿着 1-D 坐标遍历。对于静态 shape，这是最优的：循环被展开，坐标变换在编译期计算，内积求物理偏移通常也在编译期完成。

### 11.2 Copy 的通用性

同一个 `copy` 接口涵盖所有操作模式：

| 操作 | 源布局 | 目标布局 |
|------|--------|---------|
| 1D 数组 | `8:1` | `8:1` |
| ND 数组 | `(8,2,3):(1,8,16)` | `(8,2,3):(1,8,16)` |
| GATHER | `(2,2,2):(42,1,128)` | `8:1` |
| SCATTER | `8:1` | `(2,2,2):(42,1,128)` |
| BROADCAST | `8:0` | `8:1` |
| CONSTANT | `8:0` | `8:0` |
| TRANSPOSE | `(8,3):(1,8)` | `(8,3):(3,1)` |

### 11.3 分块的多样性

对于一个 M×N 的张量，有多种方式分成 4×8 的子张量：

```
<4:1, 8:1>              // 连续 4×8 块
<(2,2):(1,4), 8:1>      // 行方向交错
<(2,2):(1,4), 8:2>      // 两个方向都交错
<4:2, 8:2>              // 两个方向都跨步
```

但无论如何分块，接口始终是：
```cpp
cute::copy(src_4x8_tensor, dst_4x8_tensor);
```

### 11.4 GEMM：Rank-3 算法

```cpp
template <class AEngine, class ALayout,
          class BEngine, class BLayout,
          class CEngine, class CLayout>
CUTE_HOST_DEVICE constexpr void
gemm(Tensor<AEngine,ALayout> const& A,  // (M, K)
     Tensor<BEngine,BLayout> const& B,  // (N, K)
     Tensor<CEngine,CLayout>       & C) // (M, N)
{
    for (int k = 0; k < size<1>(A); ++k) {
        for (int m = 0; m < size<0>(A); ++m) {
            for (int n = 0; n < size<0>(B); ++n) {
                C(m,n) += A(m,k) * B(n,k);
            }
        }
    }
}
```

同一个 `gemm` 接口涵盖所有变体：

| 变体 | A Layout | B Layout | C Layout |
|------|----------|----------|----------|
| NT/TN/NN/TT GEMM | `(M,K):(1,lda)` | `(N,K):(1,ldb)` | `(M,N):(1,ldc)` |
| NTT GEMM | `(M,K):(1,lda)` | `(N,K):(1,ldb)` | `(M,N):(ldc,1)` |
| BLIS GEMM | `(M,K):(dma,dka)` | `(N,K):(dnb,dkb)` | `(M,N):(dmc,dnc)` |
| GETT | `((M₁,M₂),K)` | `(N,K)` | `((M₁,M₂),N)` |
| GETT | `(M,(K₁,K₂))` | `(N,(K₁,K₂))` | `(M,N)` |
| CONV | `(K,(C,T,R,S))` | `((N,Z,P,Q),(C,T,R,S))` | `(K,(N,Z,P,Q))` |

---

## 12. Layout 组合 (Composition)

### 12.1 定义

Layout A 和 B 的组合产生 Layout R：

$$A \circ B \rightarrow R$$

其中 $B \preceq R$（B 与 R 兼容），使得：

$$\forall c \in \mathbb{Z}(B), \quad R(c) = A(B(c))$$

### 12.2 代数性质

| 性质 | 公式 |
|------|------|
| 左单位元 | $I \circ B = B$ |
| 右单位元 | $A \circ I_A = A$ |
| 结合律 | $(A \circ B) \circ C = A \circ (B \circ C)$ |
| 左分配律 (B-满射) | $A \circ B = A \circ (B_1, B_2) = (A \circ B_1, A \circ B_2)$ |

### 12.3 直觉理解

Layout Composition 的本质是**函数组合**：

```
逻辑 1-D 坐标 ──Layout B──→ 中间索引 ──Layout A──→ 最终索引
     A(I)                                           A[k]
```

当我们有一个数据布局 A 和一个线程-值映射 B 时，$A \circ B$ 给出了每个线程需要访问的物理地址。

---

## 13. 组合的力量：线程-值分区

这是 CuTe 最强大的应用之一。让我们通过一个完整的例子来理解。

### 13.1 问题设置

假设我们有一个 24 元素的 1-D 数组，想要用 4 个线程处理，每个线程处理 6 个值。

**值布局 (Values)**：
```
Shape:  (2, 3)
Stride: (1, 4)
产生索引: 0 1 4 5 8 9
```

**线程布局 (Threads)**：
```
Shape:  (2, 2)
Stride: (2, 12)
产生索引: 0 2 12 14
```

### 13.2 组合成线程-值 (Thr-Val) 布局

将线程和值布局组合：

```
        Thr         Val
Shape:  ((2, 2),   (2, 3))
Stride: ((2, 12),  (1, 4))
```

这个布局定义了一个函数：`(thread_idx, value_idx) → 1D coord of array`

```
         值 →
线      0  1  4  5  8  9
程      2  3  6  7  10 11
↓       12 13 16 17 20 21
        14 15 18 19 22 23
```

### 13.3 分区操作 = 组合 + 切片

```cpp
Tensor input    = make_tensor(ptr, input_layout);
Tensor tv_input = composition(input, thr_val);   // 组合
Tensor thr_input = tv_input(tid, _);              // 切片
```

> [!important] 核心公式
> **分区 = 函数组合 + 切片**
> 
> 给定一个数据 Layout 和一个线程-值 Layout，通过 composition 将它们组合，然后用线程 ID 切片，就得到了每个线程的数据视图。

### 13.4 完整示例：4×8 数据分区

```
4×8 数据 (任意布局)
    0 1 2 3 4 5 6 7
0 | a e i m q u y γ
1 | b f j n r v z δ
2 | c g k o s w α ε
3 | d h l p t x β η

                    ↓ composition

8×4 thr-val → 4×8 1d coord        8×4 thr-val → 4×8 data
Shape:  ((2,4), (2,2))             (直接得到每个线程的数据)
Stride: ((8,1), (4,16))
```

---

## 14. MMA 原子操作

### 14.1 通用 FMA

最基本的 MMA 是标量 FMA（Fused Multiply-Add）：

```cpp
template <class D, class A = D, class B = A, class C = D>
struct UniversalFMA {
    using DRegisters = D[1];
    using ARegisters = A[1];
    using BRegisters = B[1];
    using CRegisters = C[1];

    CUTE_HOST_DEVICE static constexpr void
    fma(D& d, A const& a, B const& b, C const& c) {
        using cute::fma;
        fma(d, a, b, c);
    }
};
```

### 14.2 MMA_Traits 结构

每个 MMA 操作都有一个 Traits 结构，定义了：

```cpp
template <>
struct MMA_Traits<SomeOp> {
    using ElementDVal = ...;  // D 矩阵元素类型
    using ElementAVal = ...;  // A 矩阵元素类型
    using ElementBVal = ...;  // B 矩阵元素类型
    using ElementCVal = ...;  // C 矩阵元素类型

    using Shape_MNK = Shape<_M, _N, _K>;  // MMA 的逻辑形状

    using ThrID   = Layout<...>;  // 线程 ID 映射
    using ALayout = Layout<...>;  // (Thr, Val) → (M, K) 映射
    using BLayout = Layout<...>;  // (Thr, Val) → (N, K) 映射
    using CLayout = Layout<...>;  // (Thr, Val) → (M, N) 映射
};
```

> [!note] 布局映射方向
> 这些布局将 thread/value ID（定义域）映射到 MNK 逻辑坐标。实际使用时通常考虑其逆映射：MNK → thr,val

### 14.3 SM61 DP4A（整数点积）

```cpp
struct SM61_DP4A {
    using DRegisters = int32_t[1];
    using ARegisters = uint32_t[1];
    using BRegisters = uint32_t[1];
    using CRegisters = int32_t[1];

    CUTE_HOST_DEVICE static void
    fma(int32_t& d, uint32_t const& a,
        uint32_t const& b, int32_t const& c) {
        asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
            : "=r"(d)
            : "r"(a), "r"(b), "r"(c));
    }
};

template <>
struct MMA_Traits<SM61_DP4A> {
    using ElementDVal = int32_t;
    using ElementAVal = int8_t;   // 注意：逻辑类型 vs 架构寄存器类型
    using ElementBVal = int8_t;
    using ElementCVal = int32_t;

    using ThrID   = Layout<Shape<_1>>;
    using ALayout = Layout<Shape<_1, _4>>;
    using BLayout = Layout<Shape<_1, _4>>;
    using CLayout = Layout<Shape<_1, _1>>;
};
```

### 14.4 不同架构的 MMA 对比

| 架构 | 操作 | Shape MNK | 线程数 | 特点 |
|------|------|-----------|--------|------|
| SM61 | DP4A | 1×1×4 | 1 | 标量 int8 点积 |
| SM70 (Volta) | FP16 MMA | 8×8×4 | 8 | 首代张量核心 |
| SM80 (Ampere) | FP64 MMA | 8×8×4 | 32 | 双精度张量核心 |
| SM80 (Ampere) | FP16 MMA | 16×8×8 | 32 | 增大的指令形状 |
| SM90 (Hopper) | FP16 SS MMA | 64×16×16 | 128 | GMMA + 共享内存描述符 |

### 14.5 Volta FP16 MMA 详解

```cpp
template <>
struct MMA_Traits<SM70_8x8x4_F32F16F16F32_NT> {
    using Shape_MNK = Shape<_8, _8, _4>;

    using ThrID = Layout<Shape <_4,  _2>,
                         Stride<_1, _16>>;

    // (T8, V4) → (M8, K4)
    using ALayout = Layout<Shape <Shape <_4, _2>, _4>,
                           Stride<Stride<_8, _4>, _1>>;

    // (T8, V8) → (M8, N8)
    using CLayout = Layout<
        Shape <Shape <_2, _2,  _2>, Shape <_2, _2,  _2>>,
        Stride<Stride<_1, _16, _4>, Stride<_8, _2, _32>>>;
};
```

### 14.6 Hopper GMMA（SM90）

```cpp
template <GMMA::Major tnspA, GMMA::Major tnspB>
struct MMA_Traits<SM90_64x16x16_F32F16F16_SS<tnspA, tnspB>> {
    using Shape_MNK = Shape<_64, _16, _16>;
    using ThrID = Layout<_128>;

    using ElementAFrg = GMMA::smem_desc<tnspA>;  // 共享内存描述符
    using ElementBFrg = GMMA::smem_desc<tnspB>;

    using ALayout = GMMA::ABLayout<64, 16>;
    using BLayout = GMMA::ABLayout<16, 16>;
    using CLayout = Layout<
        Shape <Shape <_4, _8,   _4>, Shape <_2, _2,   _2>>,
        Stride<Stride<_128, _1, _16>, Stride<_64, _8, _512>>>;
};
```

---

## 15. TiledMMA 与 ThrMMA

### 15.1 层次结构

CuTe 的 MMA 系统分为四层：

```
MMA_Op          ← 原始 PTX 指令
    ↓
MMA_Traits      ← PTX 元信息（类型、布局）
    ↓
MMA_Atom        ← 检查调用接口 + Fragment 生成
    ↓
TiledMMA        ← MMA_Atom 的布局 + 分区工具
    ↓
ThrMMA          ← 单个线程的视图 + 分区/切片工具
```

### 15.2 MMA_Atom

```cpp
MMA_Atom mma = MMA_Atom<SM90_16x8x4_F64F64F64F64_TN>{};
print_latex(mma);
```

MMA_Atom 提供：
- 类型检查的调用接口
- Fragment（寄存器片段）生成

### 15.3 TiledMMA

通过布局多个 MMA_Atom 构建更大的操作：

```cpp
// 基本：单个 atom
Tiled_MMA mma = make_tiled_mma(SM90_16x8x4_F64F64F64F64_TN{});

// 2×2 warp 布局
Tiled_MMA mma = make_tiled_mma(
    SM90_16x8x4_F64F64F64F64_TN{},
    Layout<Shape<_2, _2>>{}       // 2×2 warps
);

// 2×2 warp + M 维度排列
Tiled_MMA mma = make_tiled_mma(
    SM90_16x8x4_F64F64F64F64_TN{},
    Layout<Shape<_2, _2>>{},      // 2×2 warps
    Tile<Layout<_8, _2>>{}        // Permute M
);
```

### 15.4 ThrMMA：完整的 GEMM 内核

```cpp
// 创建共享内存和全局内存张量
Tensor sA = make_tensor(ptrA, Shape<_64, _16>{});  // (64, 16)
Tensor sB = make_tensor(ptrB, Shape<_32, _16>{});  // (32, 16)
Tensor gC = make_tensor(ptrC, Shape<_64, _32>{});  // (64, 32)

// 获取当前线程的 MMA 视图
ThrMMA thr_mma = mma.get_slice(threadIdx.x);

// 分区：将数据按 MMA 分配给线程
Tensor tCsA = thr_mma.partition_A(sA);   // (ValA, M', K')
Tensor tCsB = thr_mma.partition_B(sB);   // (ValB, N', K')
Tensor tCgC = thr_mma.partition_C(gC);   // (ValC, M', N')

// 创建寄存器 Fragment
Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (ValA, M', K')
Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (ValB, N', K')
Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (ValC, M', N')

// 从共享内存加载到寄存器
copy(tCsA, tCrA);
copy(tCsB, tCrB);
clear(tCrC);

// 执行 MMA
for (int m = 0; m < size<1>(tCrC); ++m) {
    for (int n = 0; n < size<2>(tCrC); ++n) {
        for (int k = 0; k < size<2>(tCrA); ++k) {
            mma.call(tCrA(_, m, k),
                     tCrB(_, n, k),
                     tCrC(_, m, n));
        }
    }
}

// 写回结果
copy(tCrC, tCgC);
```

> [!tip] 分区张量的形状含义
> 分区后张量的形状为 `(Val, M', N')` 或 `(Val, M', K')`：
> - **第 0 维 (Val)**：该线程在一个 MMA_Atom 中负责的值
> - **第 1 维 (M'/N')**：该线程跨越的 MMA_Atom 数量（M/N 方向）
> - **第 2 维 (K'/N')**：该线程跨越的 MMA_Atom 数量（K/N 方向）

---

## 16. 总结与资源

### 16.1 CuTe 的核心思想总结

```
1. Layout = (Shape, Stride)     → 统一的数据表示
2. Layout 是一种代数            → 组合、逆、积、除
3. 分区 = 组合 + 切片           → 线程-数据映射的核心机制
4. 一切都是 Layout              → 线程映射、数据布局、MMA 描述
5. Rank-1 遍历                  → 编译期优化的通用算法
```

### 16.2 关键 API 速查

| API | 用途 |
|-----|------|
| `make_shape(...)` | 创建 Shape |
| `make_stride(...)` | 创建 Stride |
| `make_layout(shape, stride)` | 创建 Layout |
| `make_tensor(ptr, layout)` | 创建 Tensor（视图） |
| `composition(A, B)` | Layout 组合 |
| `logical_product(A, B)` | 逻辑积 |
| `logical_divide(A, B)` | 逻辑除 |
| `right_inverse(L)` | 右逆 |
| `left_inverse(L)` | 左逆 |
| `complement(L, S)` | 补集 |
| `tile_to_shape(atom, shape)` | 将原子扩展到目标形状 |
| `make_tiled_mma(op, layout)` | 创建 TiledMMA |
| `thr_mma.partition_A/B/C(tensor)` | 线程分区 |
| `thr_mma.make_fragment_A/B/C(...)` | 创建寄存器 Fragment |

### 16.3 资源链接

- **CUTLASS GitHub**: https://github.com/NVIDIA/cutlass
- **安装 Python 版**: `pip install nvidia-cutlass-dsl`
- **GTC 演讲**: GTC'18 到 GTC'25 历年演讲
- **统计数据**（截至演讲时）: 7500 GitHub stars, 180 贡献者, 420 万月下载量

---

> [!quote] 最后的思考
> CuTe 的设计哲学告诉我们：好的抽象不仅仅是封装复杂性，而是发现问题的**数学本质**。Layout 作为一个从坐标到索引的函数，配合组合、逆、积、除等代数运算，构成了一套完备的张量编程语言。这不是偶然的——它反映了 GPU 编程中数据移动和计算模式的内在结构。