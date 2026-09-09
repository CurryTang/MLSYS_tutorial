# ML Coding 01B · Transformer Architecture Variants: MHA Tensor Shapes, FLOPs Breakdown & KV Cache Hardware Optimizations

Rigorous analysis of Transformer operator foundations: architectural classification and masking matrices, MHA tensor pipeline and mathematical derivations, FLOPs decomposition and regime shifts, autoregressive KV cache memory modeling, and hardware-aware long-context optimizations (FlashAttention, three efficiency trajectories, head reductions, and an executive engineering reference card).

---

## Module 1: Architectural Taxonomies, Masking Patterns & Cross-Attention

```text
Comparison of Attention Masking Patterns:
Encoder-Only (BERT):           Decoder-Only (GPT / LLaMA):     Encoder-Decoder (T5 / BART):
┌───┬───┬───┬───┐             ┌───┬───┬───┬───┐               ┌───┬───┬───┬───┐
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ -∞│ -∞│ -∞│               │ 0 │ 0 │ 0 │ 0 │  (Encoder: Fully Bidirectional)
├───┼───┼───┼───┤             ├───┼───┼───┼───┤               ├───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ 0 │ -∞│ -∞│               │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┤             ├───┼───┼───┼───┤               └───┴───┴───┴───┘
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ 0 │ 0 │ -∞│               ┌───┬───┬───┬───┐
├───┼───┼───┼───┤             ├───┼───┼───┼───┤               │ 0 │ -∞│ -∞│ -∞│  (Decoder: Causal Lower-Triangular)
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ 0 │ 0 │ 0 │               └───┴───┴───┴───┘
└───┴───┴───┴───┘             └───┴───┴───┴───┘               + Cross-Attention: Q_dec × K_enc^T
[Bidirectional M_ij = 0]      [Causal Lower-Triangular]       [Bidirectional Enc + Causal Dec + Cross-Attn]
```

### High-Density Comparison of Transformer Archetypes

| Archetype | Mask Matrix $M_{ij}$ | Processing Paradigm | KV Cache State | Canonical Models & Primary Use Cases |
|---|---|---|---|---|
| **Encoder-Only** | Fully bidirectional ($M_{ij} = 0$) | Non-autoregressive; parallel execution across all $S$ tokens | **None** (Single forward pass) | BERT, RoBERTa (Classification, NER, dense embeddings) |
| **Decoder-Only** | Causal lower-triangular ($j > i \implies -\infty$) | Autoregressive; sequential token generation | **Mandatory** (Caches past Key/Value to prevent recomputation) | GPT-4, LLaMA-3, Qwen, DeepSeek (Foundation LLMs, code, reasoning) |
| **Encoder-Decoder** | Encoder bidirectional + Decoder causal + **Cross-Attention** | Bidirectional source encoding; autoregressive target decoding | **Dual Cache** (Static encoder cache + Dynamic decoder cache) | T5, BART, Whisper (Translation, summarization, ASR) |

#### Cross-Attention Mathematical Essence
- **Queries ($Q$)**: Generated from the decoder hidden state $Q_{\text{dec}} = X_{\text{dec}} W_Q \in \mathbb{R}^{B \times S_{\text{dec}} \times D}$;
- **Keys ($K$) and Values ($V$)**: Generated from final encoder representations $K_{\text{enc}} = X_{\text{enc}} W_K, \ V_{\text{enc}} = X_{\text{enc}} W_V \in \mathbb{R}^{B \times S_{\text{enc}} \times D}$;
- **Execution Invariant**: Encoder $K_{\text{enc}}, V_{\text{enc}}$ are computed once during Prefill and reused across all subsequent autoregressive decoding steps.

---

## Module 2: Multi-Head Attention (MHA) Math, Tensor Shapes & Execution Pipeline

Let batch size be $B$, sequence length $S$, hidden dimension $D$, number of attention heads $H$, and per-head dimension $d_k = D / H$.

```text
MHA Tensor Flow Lifecycle:
Input X (B, S, D)
  ├──> W_Q (D, D) ──> Q (B, S, D) ──> Reshape & Transpose ──> (B, H, S, d_k) ┐
  ├──> W_K (D, D) ──> K (B, S, D) ──> Reshape & Transpose ──> (B, H, S, d_k) ┼──> Scaled Dot-Product & Softmax
  └──> W_V (D, D) ──> V (B, S, D) ──> Reshape & Transpose ──> (B, H, S, d_k) ┘     │
                                                                                    ▼
                                                                           Score A (B, H, S, S)
                                                                                    │ × V (B, H, S, d_k)
                                                                                    ▼
                                                                           Context (B, H, S, d_k)
                                                                                    │
                                                                           Transpose & Concat (B, S, D)
                                                                                    │ × W_O (D, D)
                                                                                    ▼
                                                                           Output (B, S, D)
```

### Detailed 6-Step Tensor Transformation

1. **Linear Projections**:
   Input tensor $\mathbf{X} \in \mathbb{R}^{B \times S \times D}$ with projection weights $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$:

$$Q = \mathbf{X}W_Q, \quad K = \mathbf{X}W_K, \quad V = \mathbf{X}W_V \quad \in \mathbb{R}^{B \times S \times D}$$

2. **Head Reshaping and Transposition**:

$$\text{Reshape: } (B, S, D) \to (B, S, H, d_k) \xrightarrow{\text{Transpose (1, 2)}} (B, H, S, d_k)$$

3. **Scaled Dot-Product Attention Scores**:

$$A = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times S \times S}$$

   > **Why divide by $\sqrt{d_k}$?**  
   > Assuming $Q$ and $K$ components are independent random variables with zero mean and unit variance, the dot product $\sum_{i=1}^{d_k} q_i k_i$ has mean 0 and **variance $d_k$**. Scaling by $\frac{1}{\sqrt{d_k}}$ preserves unit variance, preventing dot products from exploding into extreme values that saturate softmax gradients.

4. **Masking & Softmax**:

$$\tilde{A} = \text{softmax}(A + M), \quad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

5. **Value Aggregation & Concatenation**:

$$\text{Head}_h = \tilde{A}_h V_h \in \mathbb{R}^{B \times H \times S \times d_k} \xrightarrow{\text{Transpose \& Reshape}} \text{MultiHead} \in \mathbb{R}^{B \times S \times D}$$

6. **Output Projection**:

$$\text{Output} = \text{MultiHead} \cdot W_O \in \mathbb{R}^{B \times S \times D}, \quad W_O \in \mathbb{R}^{D \times D}$$

---

## Module 3: MHA Computational Complexity & Regime Shift Analysis

Each multiply-accumulate operation corresponds to 2 FLOPs.

### 1. FLOPs Breakdown (for batch size $B=1$)

1. **Linear Projections ($Q, K, V, W_O$)**:
   Four matrix multiplications of shape $(S \times D) \times (D \times D)$:
   $$\text{FLOPs}_{\text{proj}} = 4 \times (2 \times S \times D \times D) = \mathbf{8 S D^2} \implies \mathcal{O}(S D^2)$$
2. **Attention Score Matrix ($Q K^T$)**:
   $H$ heads performing $(S \times d_k) \times (d_k \times S)$ multiplication:
   $$\text{FLOPs}_{QK^T} = H \times (2 \times S \times d_k \times S) = \mathbf{2 S^2 D} \implies \mathcal{O}(S^2 D)$$
3. **Value Aggregation ($\tilde{A} V$)**:
   $H$ heads performing $(S \times S) \times (S \times d_k)$ multiplication:
   $$\text{FLOPs}_{AV} = H \times (2 \times S \times S \times d_k) = \mathbf{2 S^2 D} \implies \mathcal{O}(S^2 D)$$
4. **Total MHA Layer FLOPs**:

$$\text{Total FLOPs}_{\text{MHA}} = 8 S D^2 + 4 S^2 D$$

---

### 2. Context Length Regime Shifts

```text
MHA FLOPs Component Crossover:
FLOPs
  ▲
  │                                    /  O(S² D) Attention MatMul
  │                                   /   (Dominates in long-context)
  │                                  /
  │            O(S D²) Projections  /
  │           (Dominates in short) /
  │         ─────────────────────/
  │                             /
  └────────────────────────────┴─────────────► Sequence Length S
                             S ≈ 2D (Crossover Point)
```

- **Short-Context Regime ($S < 2D$, e.g., $S=2048, D=4096$)**:
  $8 S D^2 > 4 S^2 D$. Projection matrix multiplications dominate compute ($>80\%$). GEMM throughput on Tensor Cores is the primary optimization objective.
- **Long-Context Regime ($S \gg D$, e.g., $S=32K \sim 128K, D=4096$)**:
  $4 S^2 D \gg 8 S D^2$. The quadratic attention matrix calculation explodes, dominating runtime and memory. Hardware-aware tiling (FlashAttention) and sparse/linear variants become essential.

---

## Module 4: Autoregressive Inference & KV Cache Memory Growth

### 1. Prefill Phase vs. Decode Phase

```text
Inference Phase Characteristics:
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Phase                   │ Hardware Dynamics & Bottlenecks                                        │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. Prefill Phase        │ • Parallel processing of entire prompt sequence ($S_{\text{prompt}}$)  │
│    (Prompt Processing)  │ • Computes initial KV cache                                            │
│                         │ • High arithmetic intensity $\implies$ **Compute-Bound**               │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. Decode Phase         │ • Step-by-step single token input ($x_t \in \mathbb{R}^{1 \times D}$)  │
│    (Token Generation)   │ • Appends new $k_t, v_t$ to cache; computes $q_t K_{\le t}^T V_{\le t}$│
│                         │ • Must stream full model weights & KV cache per generated token        │
│                         │ • Arithmetic intensity $\approx 1 \text{ FLOP/Byte} \implies$ **Memory-Bound**│
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

#### Why Is Query ($Q$) Never Cached?
- Query vector $q_t \in \mathbb{R}^{1 \times D}$ is only needed to compute attention against past keys $K_{\le t}$ for the current token;
- At step $t+1$, the newly generated token produces its own query $q_{t+1}$;
- **Past queries $q_1, \dots, q_t$ are never reused**, giving $Q$ an ephemeral lifespan.

---

### 2. KV Cache Memory Footprint Formula

For batch size $B$, sequence length $S$, number of layers $L$, number of key-value heads $H_{KV}$, per-head dimension $d_k$, and byte precision $b$ (e.g., $b=2$ for FP16/BF16):

$$\text{Memory}_{\text{KVCache}} = 2 \times B \times S \times L \times H_{KV} \times d_k \times b \quad \text{Bytes}$$

> *The factor of 2 accounts for Key and Value tensors.*

#### Case Study: LLaMA-3-70B
- Specifications: $L=80, D=8192, H_Q=64, H_{KV}=8 \text{ (GQA)}, d_k=128, b=2$
- Per-token memory footprint:
  $$\text{Per-Token Memory} = 2 \times 80 \times 8 \times 128 \times 2 = 327,680 \text{ Bytes} \approx \mathbf{320 \text{ KB / Token}}$$
- For batch $B=64$ and context $S=8192$:
  $$\text{Total Memory} = 64 \times 8192 \times 320 \text{ KB} \approx \mathbf{167.77 \text{ GB}}$$
  *(Exceeds the 140 GB static model weights footprint!)*

---

## Module 5: Long-Context & Hardware-Aware Attention Optimizations

### 1. Core Contradiction of Long-Context Attention & The Three Physical Walls

When the sequence length enters the long-context regime ($S \gg 2D$), quadratic attention matrix FLOPs and activation footprints overtake linear projections. Models collide simultaneously with three physical boundaries during training and serving:

```text
The Three Physical Bottlenecks of Long Context:
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│ 1. Quadratic Compute    │ 2. Training Activation  │ 3. Inference KV Cache   │
│    (Compute Wall)       │    Memory (Memory Wall) │    (IO Bandwidth Wall)  │
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ • S scales 4K -> 128K:  │ • Materializing S×S     │ • Memory capacity       │
│   Length expands 32x    │   Logits & Attn scores  │   O(B·S·L·d) grows      │
│ • Attn FLOPs surge      │ • Naive O(S²) causes    │   monotonically linear  │
│   1024x                 │   GPU OOM crashes       │ • Decoding 1 token must │
│ • Prefill latency       │ • Backward pass retains │   read full history KV  │
│   explodes quadratically│   dense gradient graphs │ • Low arithmetic        │
│                         │                         │   intensity, mem-bound  │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

To resolve these contradictions, the system stack follows an evolutionary roadmap: **Systems Patch $\to$ Three Algorithmic & Partitioning Trajectories $\to$ Hybrid Convergence**:

![[assets/attention-efficiency-landscape.png|Long-Context Attention Efficiency Landscape]]

---

### 2. Systems/Operator Patch: FlashAttention (Exact Attention, Invariant Asymptotics)

FlashAttention (Dao et al.) is foundational infrastructure for modern LLM training and serving, with the definitive trait of **exact mathematical equivalence (zero precision loss)**.

#### (1) Three Core Mechanisms
- **Tiling (On-Chip SRAM Chunking)**: Partitions input tensors $Q, K, V$ into blocks sized to fit inside ultra-fast GPU on-chip SRAM (100 KB–228 KB per SM). Matrix multiplications and normalizations execute within SRAM;
- **Online Softmax (Incremental Dynamic Normalization)**: Maintains running scaling factors $m_i$ and $l_i$, dynamically updating intermediate attention blocks as chunks stream through SRAM. **This completely eliminates reading and writing the $S \times S$ intermediate matrix to slow High Bandwidth Memory (HBM)**;
- **Recomputation in Backward**: The backward pass discards intermediate forward attention matrices and recomputes them on-the-fly inside SRAM, shrinking training activation memory from $\mathcal{O}(S^2)$ to $\mathcal{O}(S D)$.

#### (2) Physical Boundaries & Trade-Offs
- **Problems Solved**: Eliminates training-time $\mathcal{O}(S^2)$ activation memory OOM crashes; reduces HBM memory traffic from $\mathcal{O}(S^2)$ to $\mathcal{O}(S)$, boosting MFU by 2–4×;
- **Problems Unsolved**: **Total FLOPs remain strictly $\mathcal{O}(S^2 D)$**; autoregressive decoding still requires loading full historical KV Cache; quadratic Prefill latency persists at 1M+ context.

---

### 3. Three Algorithmic & Systems Efficiency Trajectories

#### Trajectory A: Sparse Attention (Pruning the Graph)
- **Core Idea**: Prune non-essential edges in the attention bipartite graph, lowering complexity from $\mathcal{O}(S^2)$ to $\mathcal{O}(S \cdot k)$ or $\mathcal{O}(S\sqrt{S})$.
- **Static Heuristics (Longformer / BigBird)**: Manually combines local sliding windows + strided/dilated windows + global tokens. Fails to capture dynamic, irregular semantic dependencies.
- **Hardware-Aligned Native Sparse Attention (DeepSeek NSA)**:
  1. **Compressed Tokens (Coarse-Grained View)**: Aggregates consecutive token blocks into pooled vectors; Queries scan at coarse granularity to locate relevant regions;
  2. **Selected Tokens (Top-$k$ Block Interaction)**: Only top-$k$ critical blocks are scheduled into fast on-chip memory for exact fine-grained attention;
  3. **Sliding Window (Fine Local Context)**: Preserves full attention over adjacent local tokens.
- **Trade-Offs**: Retains Softmax contrastive sharpness; requires block-level hardware alignment to avoid gather/scatter memory latency penalties.

#### Trajectory B: Linear & Kernelized Attention (Rewriting Associativity & Delta Rule)
- **Core Idea**: Decomposes Softmax via feature maps $\phi(\cdot)$ such that $\text{Sim}(Q, K) = \phi(Q)\phi(K)^T$. Rewriting evaluation order via associativity:
  $$\text{Standard: } (Q K^T) V \in \mathcal{O}(S^2 D) \implies \text{Linear: } \phi(Q) \left(\phi(K)^T V\right) \in \mathcal{O}(S \cdot D^2)$$
- **Constant-State Autoregressive Decoding**:
  $$S_t = S_{t-1} + \phi(k_t) v_t^T \in \mathbb{R}^{d \times d}, \quad o_t = \phi(q_t) S_t$$
  Each decoding step updates a constant-sized hidden state $S_t$. **Inference memory and time complexity per step are strictly $\mathcal{O}(1)$**, completely eliminating the linearly expanding KV Cache!
- **Overcoming Saturation: The Delta Rule (DeltaNet / RetNet)**:
  Pure accumulation ($S_t = S_{t-1} + k_t v_t^T$) suffers from memory saturation (attention dilution). Introducing associative memory erasure:
  $$W_t = W_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$
  Subtracts old projections before writing new values, achieving associative retrieval recall approaching standard Softmax.
- **Industrial Deployment**: **Hybrid Architectures** (e.g., Jamba, Nemotron-4), interleaving standard causal attention periodically among SSM/linear layers.

#### Trajectory C: Chunking & System-Level Parallelism (Altering System Partitioning)
- **Core Idea**: Fix local attention blocks $B \ll S$ algorithmically, or partition long sequences across multiple GPUs at the systems level.
- **RingAttention (Liu et al.)**:
  - Partitions sequences into $P$ chunks across $P$ GPUs, each holding local $Q$;
  - **Ring Topology**: $K, V$ blocks circulate peer-to-peer across the interconnect;
  - **Compute-Communication Overlap**: Chunk attention computation completely overlaps with asynchronous P2P transfer of the next $K, V$ block;
  - **Chunked Prefill**: Slices ultra-long prompts into scheduled chunks, eliminating Head-of-Line Blocking for concurrent Decode steps.

---

### 4. Architectural Head Reduction & Serving Memory Optimizations

```text
MHA vs MQA vs GQA Comparison:
MHA (Multi-Head Attention):        MQA (Multi-Query Attention):       GQA (Grouped-Query Attention):
Q Heads:   [1] [2] [3] [4] [5] [6] [7] [8]  Q Heads:   [1] [2] [3] [4] [5] [6] [7] [8]  Q Heads:   [1][2] [3][4] [5][6] [7][8]
K/V Heads: [1] [2] [3] [4] [5] [6] [7] [8]  K/V Heads: [         1 (Shared)        ]  K/V Heads:  [ 1 ]  [ 2 ]  [ 3 ]  [ 4 ]
(KV Cache 100%, highest VRAM)               (KV Cache 1/H, capacity loss)               (LLaMA-3 Standard: Balanced)
```

- **MHA**: $H_Q = H_{KV}$. Every Query head pairs with a distinct Key/Value head. Highest expressive capacity, largest KV cache footprint;
- **MQA**: $H_Q = H, H_{KV} = 1$. All Query heads share a single Key/Value head. Compresses KV cache by $H\times$, but impairs complex multi-turn reasoning;
- **GQA**: $H_Q = H, H_{KV} = G$ ($1 < G < H$). Partitions Query heads into $G$ groups (e.g., 64:8 in LLaMA-3-70B), preserving $\approx 99\%$ of MHA quality while approaching MQA throughput;
- **PagedAttention (vLLM)**: Virtual memory paging for KV tensors (e.g., 16 tokens/block), slashing memory fragmentation from $60\% \sim 80\%$ to $<4\%$;
- **KV Cache Quantization (FP8 / INT4)**: Quantizes cached Key/Value vectors to 8-bit or 4-bit precision, halving or quartering memory capacity demands and elevating decode concurrency.

---

### 5. Multi-Paradigm Comparison Matrix & Industrial Convergence

| Paradigm / Mechanism | Compute Complexity | Training Memory | Inference KV State Memory | Core Strengths | Core Limitations & Engineering Overhead |
|---|---|---|---|---|---|
| **Standard (+FlashAttention)** | $\mathcal{O}(S^2 D)$ | $\mathcal{O}(S D)$ | $\mathcal{O}(B \cdot S \cdot L \cdot d_k)$ Linear | Exact Softmax, zero quality loss, sharp associative recall | Prefill & FLOPs remain quadratic; low throughput at long sequences |
| **Sparse / NSA (DeepSeek)** | $\mathcal{O}(S \cdot k \cdot D)$ | $\mathcal{O}(S \cdot k)$ | $\mathcal{O}(B \cdot k \cdot L \cdot d_k)$ Sparse | Retains Softmax contrastive sharpness; high NIAH retrieval | Requires hardware-aligned block kernels; non-contiguous memory access risks |
| **Linear / DeltaNet** | $\mathcal{O}(S \cdot D^2)$ | $\mathcal{O}(S D)$ | $\mathcal{O}(D^2)$, Step $\mathcal{O}(1)$ | Extreme decoding throughput; zero KV Cache expansion | Pure accumulation suffers from memory saturation; weaker ICL than Softmax |
| **Ring / Chunked Parallel** | Distributed $\mathcal{O}(S^2 D / P)$ | Per-GPU $\mathcal{O}(B_{\text{chunk}} D)$ | Distributed slices | Breaks single-device memory limits; scales to 1M+ context | Heavy reliance on high-speed interconnects (NVLink/RoCE); network can bottleneck |

#### Industrial Convergence Stack
1. **Hardware & Interconnect Foundation**: FlashAttention manages single-GPU SRAM-HBM IO optimization, while RingAttention orchestrates cross-node sequence slicing;
2. **Architecture & Algorithm Co-Design**: GQA shrinks inference head footprints, paired with NSA native dynamic sparsity or SSM/DeltaNet hybrid interleaving.

---

## Module 6: Core Engineering Formulas & Technical Reference

| Metric / Parameter | Exact Formula | Production Benchmark (LLaMA-3-70B, $S=4096$) |
|---|---|---|
| **Layer Projection FLOPs** | $\text{FLOPs}_{\text{proj}} = 8 S D^2$ | $8 \times 4096 \times 8192^2 \approx \mathbf{2.20 \text{ TFLOPs}}$ |
| **Layer Attention FLOPs** | $\text{FLOPs}_{\text{attn}} = 4 S^2 D$ | $4 \times 4096^2 \times 8192 \approx \mathbf{0.55 \text{ TFLOPs}}$ |
| **KV Cache per Token** | $\text{Memory}_{\text{token}} = 2 L H_{KV} d_k b$ | $2 \times 80 \times 8 \times 128 \times 2 = \mathbf{320 \text{ KB / Token}}$ |
| **Decode Arithmetic Intensity** | $\text{Operational Intensity} \approx \frac{2 \times \text{Params}}{\text{Params} \times b + \text{KVCache}} \approx 1$ | Strictly Memory-Bound; throughput bounded by HBM bandwidth |
| **FlashAttention IO Speedup** | HBM transfers reduced from $\mathcal{O}(S^2)$ to $\mathcal{O}(S)$ | Activation memory drops from $\mathcal{O}(S^2)$ to $\mathcal{O}(S D)$; 2–4× MFU gain |
| **Linear Attention Decoding** | Recurrent update $S_t = S_{t-1} + k_t v_t^T \in \mathbb{R}^{d \times d}$ | $\mathcal{O}(1)$ step time; $\mathcal{O}(D^2)$ hidden state; zero expanding cache |

### Four Core Architectural Engineering Rules
1. **FlashAttention does not alter FLOPs**: It is an IO-aware memory access optimization solving activation OOM and memory bandwidth stalls; quadratic computational limits require algorithmic shifts (Sparse or Linear);
2. **Sparse attention requires memory alignment**: Dynamic sparsity must be block-aligned (e.g., DeepSeek NSA) to avoid irregular scatter/gather memory access penalties;
3. **Linear attention requires active erasure**: Pure additive states saturate over long sequences; Delta-rule updates are mandatory to approach Softmax associative recall;
4. **RingAttention is exact distributed partitioning**: It hides communication latency behind compute to preserve exact mathematical outputs across clusters.
