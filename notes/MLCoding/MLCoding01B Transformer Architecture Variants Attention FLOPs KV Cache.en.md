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

#### (2) Rigorous Dissection of Complexities: Memory vs IO vs Compute (FLOPs)
A frequent misconception among practitioners is conflating "reduced activation memory" with "reduced computational complexity." FlashAttention embodies the quintessential systems engineering philosophy of **"Compute for Memory & IO"**:

| Complexity Dimension | Standard Attention | FlashAttention (Exact) | Physical Mechanism & Engineering Essence |
| :--- | :--- | :--- | :--- |
| **Activation Memory Footprint** | $\mathcal{O}(S^2)$<br>Materializes and saves dense $S \times S$ score and probability maps | $\mathcal{O}(S \cdot D)$<br>Stores only output tensor $O$ and normalization vector $L_i$ | **Dimensional Reduction (1000× lower)**: Completely eliminates training-time activation OOM crashes on long sequences. |
| **HBM Access / IO Complexity** | $\mathcal{O}(S^2 + S D)$<br>Repeated round-trips writing/reading $S \times S$ matrices across slow HBM | $\mathcal{O}\left(\frac{S^2 D^2}{M}\right) \approx \mathcal{O}(S D)$<br>$M$ is SRAM capacity; all intermediates reside in on-chip registers | **5–10× IO Traffic Drop**: Flips the operator from heavily **Memory-Bound** to **Compute-Bound**, doubling hardware MFU. |
| **Forward FLOPs** | $\approx 4 S^2 D$<br>($2 S^2 D$ under causal masking) | $\approx 4 S^2 D$<br>($2 S^2 D$ under causal masking) | **Strictly Invariant**: Underlying fused GEMMs retain identical tensor multiply-accumulate operations. |
| **Backward FLOPs** | $\approx 8 S^2 D$<br>($4 S^2 D$ under causal masking) | $\approx 10 S^2 D$<br>($5 S^2 D$ under causal masking) | **~25% FLOP Increase**: Since intermediate $S \times S$ attention maps were discarded, backward recomputes $Q K^T$ and softmax on-the-fly in SRAM. |
| **Total Computational FLOPs** | $\mathcal{O}(S^2 D)$ | $\mathcal{O}(S^2 D)$<br>(Total arithmetic operations slightly increase by ~16.7%) | **Does NOT reduce asymptotic compute!** FlashAttention speedups stem 100% from cutting slow HBM traffic, not reducing mathematical FLOPs. |

#### (3) Physical Boundaries & Trade-Offs
- **Problems Solved**: Eliminates training-time $\mathcal{O}(S^2)$ activation memory OOM crashes; reduces HBM memory traffic from $\mathcal{O}(S^2)$ to $\mathcal{O}(S)$, boosting MFU by 2–4×;
- **Problems Unsolved**:
  1. **Total FLOPs remain strictly $\mathcal{O}(S^2 D)$**: Prefill latency at 1M+ context still explodes quadratically;
  2. **Autoregressive Decoding Memory Wall Persists**: Generating each token still requires reading the full historical KV Cache;
  3. **Cannot Reduce Inference KV Cache Footprint**: KV Cache capacity still scales linearly as $\mathcal{O}(B \cdot S \cdot L \cdot D)$.

> [!TIP]
> **Hands-On Coding Exercise**:
> Want to implement FlashAttention's core tiling and GPU kernel from scratch? Check out **[[MLCoding03 Attention Variants GQA Sliding Window KV Cache.en.md#Exercise 7 · Flash Attention (tiling + online softmax)|ML Coding 03 · Exercise 7: Flash Attention from PyTorch Online Softmax to Production Triton GPU Kernel]]** for complete implementation and numerical validation.

### 3. Three Algorithmic & Systems Efficiency Trajectories

To fundamentally transcend the "quadratic FLOPs wall" and "autoregressive decoding KV cache bandwidth wall" left unresolved by FlashAttention, the ecosystem has developed three primary efficiency trajectories. The table below presents a unified, multi-dimensional complexity comparison benchmarking these three trajectories alongside Standard Attention and FlashAttention under the same physical hardware metrics:

#### (0) Unified Complexity & Hardware Bottleneck Benchmark Matrix

| Mechanism / Efficiency Trajectory | Training Activation Memory | HBM Access / IO Traffic | Forward FLOPs | Backward FLOPs | Autoregressive Decode Step Cost | Hardware Regime | Physical Bottleneck & Engineering Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Attention<br>(Standard / Eager)** | $\mathcal{O}(S^2)$<br>Materializes dense $S \times S$ matrix | $\mathcal{O}(S^2 + S D)$<br>Frequent round-trip HBM spills | $4 S^2 D$<br>($2 S^2 D$ causal) | $8 S^2 D$<br>($4 S^2 D$ causal) | Memory: $\mathcal{O}(S D)$ linear growth<br>Compute: $2 S D$ scans full history | **Severe Memory-Bound**<br>Intensity $< 1$, extreme bandwidth starvation | Instant activation OOM on long sequences; zero on-chip data reuse. |
| **FlashAttention<br>(Dao et al. Systems Patch)** | $\mathcal{O}(S \cdot D)$<br>Discards intermediates; stores only $O$ and $L_i$ | $\mathcal{O}\left(\frac{S^2 D^2}{M}\right) \approx \mathcal{O}(S D)$<br>SRAM-tiled pipelined fusion | $4 S^2 D$<br>($2 S^2 D$ causal) | $10 S^2 D$<br>($5 S^2 D$ causal)<br>Recomputation adds ~25% | Memory: $\mathcal{O}(S D)$ linear growth<br>Compute: $2 S D$ scans full history | **Compute-Bound**<br>Fully saturates Tensor Core systolic arrays | **Does NOT reduce quadratic FLOPs**; Prefill latency explodes at 1M+ context; cannot break decoding KV cache wall. |
| **Trajectory A: Native Sparse Attention<br>(DeepSeek NSA / Sparse)** | $\mathcal{O}(S \cdot D)$<br>Stores only coarse indices and Top-$k$ active blocks | $\mathcal{O}(S \cdot k_{\text{eff}} \cdot D)$<br>TMA block-aligned coalesced transfers | $\approx 4 S \cdot k_{\text{eff}} \cdot D$<br>(Scales near-linearly with length) | $\approx 8 S \cdot k_{\text{eff}} \cdot D$<br>(Compute drops by multiples to orders of magnitude) | Memory: $\mathcal{O}(k_{\text{eff}} \cdot D)$<br>Compute: $2 k_{\text{eff}} D$ scans active blocks only | **Compute-Bound**<br>(Requires 64-token block alignment) | Must strictly enforce hardware block alignment (64 tokens); discrete token pruning causes catastrophic uncoalesced memory stalls. |
| **Trajectory B: Linear Attention & SSM<br>(DeltaNet / RetNet / SSM)** | $\mathcal{O}(S \cdot D)$<br>Propagates $D \times D$ hidden states chunkwise | $\mathcal{O}(S \cdot D)$<br>Single streaming linear scan, minimal IO | $\approx 4 S D^2$<br>(At $S \gg D$, compute drops 1000×) | $\approx 8 S D^2$<br>(Strictly linear complexity) | Memory: $\mathcal{O}(D^2)$ **Strictly $\mathcal{O}(1)$ constant**<br>Compute: $\mathcal{O}(D^2)$ **Strictly $\mathcal{O}(1)$ constant** | **Compute-Bound** (Training)<br>**Throughput-Flat** (Decoding) | Plain kernelization suffers capacity saturation & attention dilution; requires Delta projection erasure; slightly trails Softmax on complex retrieval. |
| **Trajectory C: Distributed Context Parallelism<br>(RingAttention / CP)** | Per-GPU $\mathcal{O}\left(\frac{S}{P} \cdot D\right)$<br>Scales down linearly with GPU count $P$ | Per-GPU retains SRAM tiling; cross-node over NVLink/RDMA ring | Per-GPU $\approx \frac{2 S^2 D}{P}$<br>Cluster total strictly invariant | Per-GPU $\approx \frac{5 S^2 D}{P}$<br>Cluster total strictly invariant | Per-GPU holds slice $\mathcal{O}\left(\frac{S}{P} \cdot D\right)$<br>Asynchronous ring flow | **Overlap Compute-Bound**<br>(Double-buffering hides comm latency) | Highly dependent on cluster interconnect bandwidth; too small chunks fail to hide communication (falls back to Comm-Bound). |

---

#### Trajectory A: Sparse Attention (Pruning the Graph)
- **Core Idea**: Prune non-essential edges in the attention bipartite graph, lowering complexity from $\mathcal{O}(S^2)$ to $\mathcal{O}(S \cdot k)$ or $\mathcal{O}(S\sqrt{S})$.
- **Static Heuristics (Longformer / BigBird)**: Manually combines local sliding windows + strided/dilated windows + global tokens. Fails to capture dynamic, irregular semantic dependencies.
- **Hardware-Aligned Native Sparse Attention (DeepSeek NSA)**:
  1. **Compressed Tokens (Coarse-Grained View)**: Aggregates consecutive token blocks into pooled vectors; Queries scan at coarse granularity to locate relevant regions;
  2. **Selected Tokens (Top-$k$ Block Interaction)**: Only top-$k$ critical blocks are scheduled into fast on-chip memory for exact fine-grained attention;
  3. **Sliding Window (Fine Local Context)**: Preserves full attention over adjacent local tokens.
- **Trade-Offs**: Retains Softmax contrastive sharpness; requires block-level hardware alignment to avoid gather/scatter memory latency penalties.

<details class="technical-deep-dive">
<summary><span class="deep-dive-badge">Kernel Deep-Dive</span><span class="deep-dive-title">DeepSeek NSA Native Sparse Attention: Hardware Tile Alignment & Triton Kernel Implementation</span></summary>
<div class="deep-dive-content">

##### 1. Why Fine-Grained Token-Level Sparsity Fails on Modern GPUs
Early dynamic sparsity approaches (e.g., token-level Top-$k$ pruning) reduce theoretical FLOPs, but frequently result in **severe throughput degradation** on modern hardware (e.g., Hopper H100, Blackwell B200):
- **Memory Coalescing Breakdown**: GPU HBM3/HBM3e peak bandwidth requires 32-thread warps to request contiguous 128-byte cache lines. Dynamic per-token gather/scatter addressing breaks memory coalescing, degrading effective DRAM bandwidth to below $10\%$ of peak;
- **Tensor Core Systolic Array Misalignment**: Tensor Cores (such as Hopper `wgmma`) process dense $64 \times 64$ or $128 \times 128$ tiles directly within on-chip Shared Memory (SRAM). Unstructured token selections cannot populate these dense hardware pipelines, forcing fallback to low-throughput generic CUDA cores;
- **Warp Divergence**: Divergent sparse index patterns across threads within the same warp cause execution serialization and pipeline stalls.

##### 2. DeepSeek NSA 3-Branch Hardware-Aligned Architecture
DeepSeek NSA enforces **Block Alignment ($L_b = 64$ contiguous tokens)** as the fundamental hardware dataflow primitive:
1. **Compressed Coarse Branch**:
   - Compresses Key/Value tokens with block-level AvgPool: $K^{\text{cmp}} \in \mathbb{R}^{\frac{S}{L_c} \times D}$;
   - Queries interact with compressed keys to yield coarse block-importance scores via fast, contiguous Tensor Core GEMMs.
2. **Selected Fine-Grained Block Branch**:
   - Based on coarse scores, selects the Top-$k$ **entire contiguous blocks**;
   - Blocks are loaded using Hopper TMA (Tensor Memory Accelerator) asynchronous copy engines directly from HBM to SRAM with 100% coalescing.
3. **Sliding Window Branch**:
   - Maintains dense causal attention over the most recent $W$ tokens (e.g., 512 tokens = 8 contiguous blocks) to ensure syntactic and grammatical integrity.
4. **Unified Streaming Online Softmax**:
   - All three branches update a single shared Online Softmax state $(m_i, l_i, O_i)$ inside SRAM registers within one fused kernel pass, avoiding numerical drift or intermediate activation spills.

##### 3. NSA Forward Triton Kernel Skeleton

```python
import triton
import triton.language as tl

@triton.jit
def _nsa_fwd_kernel(
    Q, K, V, SelectedIndices, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_sb, stride_sh, stride_sm, stride_sk,
    sm_scale,
    Q_LEN: tl.constexpr,
    NUM_SELECTED_BLOCKS: tl.constexpr,  # e.g., Top-4 selected blocks
    BLOCK_SIZE: tl.constexpr,           # Hardware-aligned block size (64 tokens)
    HEAD_DIM: tl.constexpr,             # Head hidden dimension (64 or 128)
    BLOCK_M: tl.constexpr,              # Query tile size (64)
):
    # Grid: (cdiv(Q_LEN, BLOCK_M), NUM_HEADS, BATCH)
    pid_m = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    # 1. Load Query Tile into SRAM [BLOCK_M, HEAD_DIM]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN, other=0.0)

    # 2. Initialize Online Softmax accumulators in registers
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Iterate over hardware-coalesced selected block indices
    sel_base = SelectedIndices + batch_idx * stride_sb + head_idx * stride_sh + pid_m * stride_sm
    
    for k_idx in range(NUM_SELECTED_BLOCKS):
        block_id = tl.load(sel_base + k_idx * stride_sk)
        
        # Contiguous block offset calculation
        offs_n = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        
        # Coalesced block transfer into SRAM
        k_block = tl.load(k_ptrs) # [HEAD_DIM, BLOCK_SIZE]
        v_block = tl.load(v_ptrs) # [BLOCK_SIZE, HEAD_DIM]

        # Compute block dot product via Tensor Core [BLOCK_M, BLOCK_SIZE]
        s_ij = tl.dot(q, k_block) * sm_scale

        # Online Softmax running statistics update
        m_curr = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.exp(m_i - m_curr)
        p = tl.exp(s_ij - m_curr[:, None])

        # Accumulate attention output and normalization factor
        acc_o = acc_o * alpha[:, None] + tl.dot(p.to(v_block.dtype), v_block)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_curr

    # 4. Sliding window blocks process similarly in SRAM...

    # 5. Normalize and write back to global HBM
    out_ptrs = Out + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, (acc_o / l_i[:, None]).to(Out.dtype.element_ty), mask=offs_m[:, None] < Q_LEN)
```

</div>
</details>

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

<details class="technical-deep-dive">
<summary><span class="deep-dive-badge">Kernel Deep-Dive</span><span class="deep-dive-title">DeltaNet Associative Memory Update, Chunkwise Parallel Scan & Triton Kernel Implementation</span></summary>
<div class="deep-dive-content">

##### 1. Capacity Saturation in Linear Attention & The Delta Erasure Mechanism
In naive linear attention, the recurrent update is purely additive: $S_t = S_{t-1} + k_t v_t^T$.
- **Capacity Saturation**: The recurrent state $S_t \in \mathbb{R}^{d \times d}$ is an unweighted sum of outer products. For sequence lengths $t \gg d$, the matrix rank saturates and early signals cannot be forgotten, precipitating severe "Attention Dilution" in associative retrieval and multi-needle tasks;
- **Delta Rule (Online Associative Gradient Descent)**:
  DeltaNet casts each memory write as error-driven online correction against key $k_t$:
  $$\mathcal{L}_t = \frac{1}{2} \| S_{t-1} k_t - v_t \|_2^2$$
  Updating the state with learning rate $\beta_t \in [0, 1]$:
  $$S_t = S_{t-1} - \beta_t \nabla_S \mathcal{L}_t = S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$
  The factor $(I - \beta_t k_t k_t^T)$ forms a rank-1 Householder-like projection that **geometrically projects out and erases old memory stored along the direction of $k_t$** before writing the new value $v_t$.

##### 2. The Training Parallelization Paradox & Chunkwise Parallel Formulation
- **The Parallelization Paradox**: At inference time, updating $S_t$ takes $\mathcal{O}(1)$ time. At training time, however, sequential dependence across $S = 4096$ tokens would serialize the GPU and underutilize SMs;
- **Chunkwise Parallel Decoupling**: Decomposes the sequence into blocks of size $C = 64$ (aligned with Tensor Core tiles):
  1. **Intra-Chunk Fast Solve**:
     Internal causal interactions within a chunk are solved via a lower-triangular matrix equation in SRAM:
     $$V_{\text{new}} = (I + \text{tril}(\beta K K^T, -1))^{-1} (\beta \odot V)$$
     Since $C = 64$ is small, the matrix inverse $(I + L)^{-1}$ is resolved directly inside SRAM registers using a 1st-order Neumann series expansion ($I - L + L^2$) or forward substitution;
     Intra-chunk attention is evaluated as: $O_{\text{intra}} = \text{tril}(Q K^T) V_{\text{new}}$;
  2. **Inter-Chunk State Transfer**:
     Inter-chunk state transmission operates as a macro-RNN:
     $$S_c = S_{c-1} A_c + B_c$$
     where $A_c = \prod_{t \in c} (I - \beta_t k_t k_t^T) \in \mathbb{R}^{d \times d}$ is the cumulative chunk decay matrix and $B_c = K_c^T V_{\text{new}}$.
     The state is updated once per chunk, shrinking the sequential step count to $S / C$ ($4096 / 64 = 64$ steps), achieving high Tensor Core utilization.

##### 3. DeltaNet Chunkwise Triton Kernel Skeleton

```python
import triton
import triton.language as tl

@triton.jit
def _deltanet_chunk_fwd_kernel(
    Q, K, V, Beta, Out,
    stride_b, stride_h, stride_s, stride_d,
    CHUNK_SIZE: tl.constexpr, # Tile size (64)
    DIM: tl.constexpr,        # Hidden state dimension (64 or 128)
    NUM_CHUNKS: tl.constexpr  # S // CHUNK_SIZE
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    base_ptr = batch_idx * stride_b + head_idx * stride_h

    # 1. Initialize recurrent state S [DIM, DIM] in SRAM registers
    offs_d1 = tl.arange(0, DIM)
    offs_d2 = tl.arange(0, DIM)
    s_state = tl.zeros([DIM, DIM], dtype=tl.float32)

    # 2. Iterate across chunks (S // CHUNK_SIZE iterations)
    for c_idx in range(NUM_CHUNKS):
        offs_c = c_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
        
        # Load chunk Q, K, V, Beta into SRAM
        q = tl.load(Q + base_ptr + offs_c[:, None] * stride_s + offs_d1[None, :] * stride_d)
        k = tl.load(K + base_ptr + offs_c[:, None] * stride_s + offs_d1[None, :] * stride_d)
        v = tl.load(V + base_ptr + offs_c[:, None] * stride_s + offs_d1[None, :] * stride_d)
        beta = tl.load(Beta + base_ptr + offs_c[:, None] * stride_s) # [CHUNK_SIZE, 1]

        # 3. Inter-chunk historical memory contribution: O_inter = Q @ S_{c-1}
        o_inter = tl.dot(q, s_state) # [CHUNK_SIZE, DIM]

        # 4. Intra-chunk causal solve
        gram = tl.dot(k, tl.trans(k)) * beta
        mask_tril = offs_c[:, None] > offs_c[None, :]
        gram_tril = tl.where(mask_tril, gram, 0.0)

        # First-order Neumann expansion solve in SRAM
        v_eff = beta * v
        v_eff = v_eff - tl.dot(gram_tril, v_eff)

        # Intra-chunk attention interaction: O_intra = tril(Q @ K.T) @ v_eff
        qk = tl.dot(q, tl.trans(k))
        qk_causal = tl.where(mask_tril, qk, 0.0)
        o_intra = tl.dot(qk_causal, v_eff)

        # 5. Write back combined chunk output: Out = O_inter + O_intra
        tl.store(Out + base_ptr + offs_c[:, None] * stride_s + offs_d1[None, :] * stride_d, o_inter + o_intra)

        # 6. Update global recurrent state S_c = S_{c-1} (I - beta K^T K) + K^T v_eff
        decay = tl.dot(tl.trans(k), tl.dot(k, s_state)) * tl.mean(beta)
        s_state = s_state - decay + tl.dot(tl.trans(k), v_eff)
```

</div>
</details>

#### Trajectory C: Chunking & System-Level Parallelism (Altering System Partitioning)
- **Core Idea**: Fix local attention blocks $B \ll S$ algorithmically, or partition long sequences across multiple GPUs at the systems level.
- **RingAttention (Liu et al.)**:
  - Partitions sequences into $P$ chunks across $P$ GPUs, each holding local $Q$;
  - **Ring Topology**: $K, V$ blocks circulate peer-to-peer across the interconnect;
  - **Compute-Communication Overlap**: Chunk attention computation completely overlaps with asynchronous P2P transfer of the next $K, V$ block;
  - **Chunked Prefill**: Slices ultra-long prompts into scheduled chunks, eliminating Head-of-Line Blocking for concurrent Decode steps.

<details class="technical-deep-dive">
<summary><span class="deep-dive-badge">Distributed Systems Deep-Dive</span><span class="deep-dive-title">RingAttention: Asynchronous P2P Double-Buffering, Causal Block Skipping & Streaming Softmax Fusion</span></summary>
<div class="deep-dive-content">

##### 1. The Context Parallelism Bottleneck & Ring Topology Solution
For context lengths spanning millions of tokens ($S = 1\text{M} \sim 10\text{M}$), a single GPU's 80GB HBM cannot hold activations and KV Caches.
- **Memory Explosion of AllGather**: Using naive distributed AllGather to broadcast all $K, V$ blocks restores the per-GPU memory footprint to $\mathcal{O}(S)$, negating distributed memory advantages;
- **Ring P2P Circulation**: Arranges $P$ GPUs in a 1D logical ring ($0 \to 1 \to \dots \to P-1 \to 0$):
  - Each GPU persistently holds its local query slice $Q_{\text{local}} \in \mathbb{R}^{\frac{S}{P} \times D}$;
  - $K, V$ slices stream peer-to-peer across the ring in chunks. Each GPU computes against the incoming block and forwards it, keeping **per-GPU memory bounded strictly at $\mathcal{O}(S / P)$**.

##### 2. Exact Condition for 100% Compute-Communication Overlap
By allocating ping-pong double buffers:
- **Pipeline Step $s$**:
  1. **Asynchronous Communication**: In the background, non-blocking NCCL P2P primitives (`isend` / `irecv`) transmit $K^{(s)}, V^{(s)}$ to $(r+1)\%P$ and receive $K^{(s+1)}, V^{(s+1)}$ from $(r-1)\%P$;
  2. **Compute Core**: On the GPU foreground, Tensor Cores evaluate FlashAttention between local $Q$ and incoming $K^{(s)}, V^{(s)}$;
- **Zero-Overhead Invariant**:
  $$T_{\text{comm}} = \frac{4 \times (S/P) \times D \times b}{\text{Bandwidth}_{\text{ring}}}, \quad T_{\text{comp}} = \frac{4 \times (S/P)^2 \times D}{\text{TFLOPS}_{\text{GPU}}}$$
  As long as local chunk size satisfies $S/P \ge \frac{\text{TFLOPS}_{\text{GPU}}}{\text{Bandwidth}_{\text{ring}}} \cdot b$, compute time exceeds network transfer, rendering communication **100% hidden (Compute-Bound)**.

##### 3. Causal Block Pruning & Inter-Step Streaming Softmax Update
- **Causal Block Pruning**:
  For GPU rank $i$, attention only requires evaluating source keys $j \le i$:
  - If $j > i$ (future blocks): entirely skipped, saving $50\%$ of global FLOPs;
  - If $j == i$ (diagonal block): applies standard causal triangular mask;
  - If $j < i$ (historical blocks): full unmasked attention.
- **Inter-Step Online Softmax Formulation**:
  Each GPU maintains local state $(m_{\text{run}}, l_{\text{run}}, O_{\text{run}})$ updated smoothly after each ring hop:
  $$m_{\text{new}} = \max(m_{\text{run}}, m_{\text{block}})$$
  $$\alpha = \exp(m_{\text{run}} - m_{\text{new}}), \quad \beta = \exp(m_{\text{block}} - m_{\text{new}})$$
  $$l_{\text{run}} = \alpha \cdot l_{\text{run}} + \beta \cdot l_{\text{block}}$$
  $$O_{\text{run}} = \alpha \cdot O_{\text{run}} + \beta \cdot O_{\text{block}}$$
  After $P$ steps, normalizing $O_{\text{final}} = O_{\text{run}} / l_{\text{run}}$ produces a result **bit-exact to full single-GPU attention**.

##### 4. PyTorch Distributed Ring Attention Core Implementation

```python
import torch
import torch.distributed as dist

def ring_flash_attention_forward(q_local, k_local, v_local, group=None):
    """
    q_local, k_local, v_local: [Batch, S_local, Heads, Dim], with S_local = Total_Seq / P
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    # 1. Allocate ping-pong buffers to prevent memory overwrites
    k_curr, v_curr = k_local.clone(), v_local.clone()
    k_next = torch.empty_like(k_local)
    v_next = torch.empty_like(v_local)

    # 2. Initialize streaming Online Softmax state
    m_running = torch.full((q_local.shape[0], q_local.shape[2], q_local.shape[1]), -float('inf'), device=q_local.device)
    l_running = torch.zeros_like(m_running)
    o_running = torch.zeros_like(q_local)

    # 3. Step through the logical ring
    for step in range(world_size):
        work_handles = []
        if step < world_size - 1:
            reqs = [
                dist.P2POp(dist.isend, k_curr, next_rank, group),
                dist.P2POp(dist.isend, v_curr, next_rank, group),
                dist.P2POp(dist.irecv, k_next, prev_rank, group),
                dist.P2POp(dist.irecv, v_next, prev_rank, group),
            ]
            work_handles = dist.batch_isend_irecv(reqs)

        source_rank = (rank - step + world_size) % world_size

        # Causal block pruning
        if source_rank <= rank:
            is_causal = (source_rank == rank)
            out_block, m_block, l_block = flash_attn_chunk(q_local, k_curr, v_curr, causal=is_causal)
            
            # Dynamic Online Softmax rescaling
            m_new = torch.maximum(m_running, m_block)
            alpha = torch.exp(m_running - m_new)
            beta = torch.exp(m_block - m_new)
            
            l_running = alpha * l_running + beta * l_block
            o_running = alpha.unsqueeze(-1) * o_running + beta.unsqueeze(-1) * out_block
            m_running = m_new

        if step < world_size - 1:
            for req in work_handles:
                req.wait()
            k_curr, k_next = k_next, k_curr
            v_curr, v_next = v_next, v_curr

    return o_running / l_running.unsqueeze(-1)
```

</div>
</details>

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
