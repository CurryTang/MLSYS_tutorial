# ML Coding 01B · Transformer 架构变体：MHA 张量维度推导、FLOPs 分解与 KV Cache 硬件优化

系统解析 Transformer 算子底层的五大核心模块：架构分类与掩码矩阵、MHA 张量流动与数学推导、FLOPs 严密分解与计算瓶颈体制转移（Regime Shift）、自回归推理与 KV Cache 显存模型、以及长序列硬件感知优化全景（FlashAttention、三大效率路线、头数压缩与工程速查清单）。

---

## 模块一：Transformer 架构分类学与注意力掩码模式

```text
三大架构注意力掩码模式对比：
Encoder-Only (BERT):           Decoder-Only (GPT / LLaMA):     Encoder-Decoder (T5 / BART):
┌───┬───┬───┬───┐             ┌───┬───┬───┬───┐               ┌───┬───┬───┬───┐
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ -∞│ -∞│ -∞│               │ 0 │ 0 │ 0 │ 0 │  (Encoder: 全双向)
├───┼───┼───┼───┤             ├───┼───┼───┼───┤               ├───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ 0 │ -∞│ -∞│               │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┤             ├───┼───┼───┼───┤               └───┴───┴───┴───┘
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ 0 │ 0 │ -∞│               ┌───┬───┬───┬───┐
├───┼───┼───┼───┤             ├───┼───┼───┼───┤               │ 0 │ -∞│ -∞│ -∞│  (Decoder: 因果掩码)
│ 0 │ 0 │ 0 │ 0 │             │ 0 │ 0 │ 0 │ 0 │               └───┴───┴───┴───┘
└───┴───┴───┴───┘             └───┴───┴───┴───┘               + Cross-Attention: Q_dec × K_enc^T
[全双向无掩码 M_ij = 0]       [因果下三角掩码 j > i 时 -∞]   [双向编码 + 因果解码 + 跨注意力]
```

### 三大架构形态高密度对比

| 架构形态 | 注意力掩码矩阵 $M_{ij}$ | 输入与解码范式 | KV Cache 状态 | 典型代表与适用场景 |
|---|---|---|---|---|
| **Encoder-Only** | 全双向（$M_{ij} = 0$） | 非自回归，单次前向并行处理全部 $S$ 个 Token | **无**（单次前向直接输出） | BERT, RoBERTa（文本分类、实体识别、向量表征） |
| **Decoder-Only** | 因果下三角（$j > i$ 时 $M_{ij} = -\infty$） | 自回归生成，逐 Token 依赖历史上下文 | **必须维护**（缓存历史 Key/Value 避免重复计算） | GPT-4, LLaMA-3, Qwen, DeepSeek（通用大模型、代码生成、推理） |
| **Encoder-Decoder** | Encoder 双向 + Decoder 因果 + **Cross-Attention** | 双向编码源端，自回归生成目标端 | **双份缓存**（Encoder 静态缓存 + Decoder 动态缓存） | T5, BART, Whisper（机器翻译、文本摘要、ASR） |

#### Cross-Attention（跨注意力）计算本质
- **Query（$Q$）**：来源于 Decoder 上一层的隐层状态 $Q_{\text{dec}} = X_{\text{dec}} W_Q \in \mathbb{R}^{B \times S_{\text{dec}} \times D}$；
- **Key（$K$）与 Value（$V$）**：来源于 Encoder 顶层输出 $K_{\text{enc}} = X_{\text{enc}} W_K, \ V_{\text{enc}} = X_{\text{enc}} W_V \in \mathbb{R}^{B \times S_{\text{enc}} \times D}$；
- **执行特性**：Encoder 的 $K_{\text{enc}}, V_{\text{enc}}$ 在 Prefill 阶段仅计算一次并缓存，解码全程被所有解码步反复共享读取。

---

## 模块二：多头注意力（MHA）数学推导、张量形状演化与执行流水线

设批量大小为 $B$，序列长度为 $S$，模型隐藏维度为 $D$，注意力头数为 $H$，每个头的维度为 $d_k = D / H$。

```text
多头注意力 (MHA) 张量流动全景图：
输入 X (B, S, D)
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

### 详细六步张量变换流程

1. **输入与线性投影（Linear Projections）**：
   输入张量 $\mathbf{X} \in \mathbb{R}^{B \times S \times D}$，权重矩阵 $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$：

$$Q = \mathbf{X}W_Q, \quad K = \mathbf{X}W_K, \quad V = \mathbf{X}W_V \quad \in \mathbb{R}^{B \times S \times D}$$

2. **多头拆分与轴转置（Head Reshape & Transposition）**：
   将隐藏维度 $D$ 拆分为 $H$ 个头，每个头维度为 $d_k$：

$$\text{Reshape: } (B, S, D) \to (B, S, H, d_k) \xrightarrow{\text{Transpose (1, 2)}} (B, H, S, d_k)$$

3. **缩放点积注意力得分（Scaled Dot-Product Attention Scores）**：

$$A = \frac{Q K^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times S \times S}$$

   > **为什么必须除以 $\sqrt{d_k}$？**  
   > 假设 $Q$ 和 $K$ 的各个分量是均值为 0、方差为 1 的独立随机变量，则点积 $\sum_{i=1}^{d_k} q_i k_i$ 的均值为 0，**方差为 $d_k$**。如果不进行缩放，在高维情况下点积数值会变得极大，导致 Softmax 函数进入**梯度饱和区（极度平坦）**，反向传播时梯度几乎消失。除以 $\sqrt{d_k}$ 将方差重新拉回 1，保持 Softmax 的灵敏度。

4. **因果掩码与归一化（Causal Masking & Softmax）**：

$$\tilde{A} = \text{softmax}(A + M), \quad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

5. **Value 聚合与头拼接（Value Aggregation & Concatenation）**：

$$\text{Head}_h = \tilde{A}_h V_h \in \mathbb{R}^{B \times H \times S \times d_k} \xrightarrow{\text{Transpose \& Reshape}} \text{MultiHead} \in \mathbb{R}^{B \times S \times D}$$

6. **输出投影（Output Projection）**：

$$\text{Output} = \text{MultiHead} \cdot W_O \in \mathbb{R}^{B \times S \times D}, \quad W_O \in \mathbb{R}^{D \times D}$$

---

## 模块三：MHA 计算复杂度 FLOPs 严密分解与体制转移

在算法面试与系统设计中，精确估算单层 Attention 的浮点运算次数（FLOPs，乘加各算 1 次，一次乘加 = 2 FLOPs）至关重要。

### 1. FLOPs 严密分解（以单样本 $B=1$ 为例）

1. **四次线性投影（$Q, K, V, W_O$）**：
   每个投影为 $(S \times D) \times (D \times D)$ 的矩阵乘法：
   $$\text{FLOPs}_{\text{proj}} = 4 \times (2 \times S \times D \times D) = \mathbf{8 S D^2} \implies \mathcal{O}(S D^2)$$
2. **计算注意力得分矩阵（$Q K^T$）**：
   $H$ 个头，每个头做 $(S \times d_k) \times (d_k \times S)$ 的矩阵乘法：
   $$\text{FLOPs}_{QK^T} = H \times (2 \times S \times d_k \times S) = 2 S^2 (H \cdot d_k) = \mathbf{2 S^2 D} \implies \mathcal{O}(S^2 D)$$
3. **Value 加权聚合（$\tilde{A} V$）**：
   $H$ 个头，每个头做 $(S \times S) \times (S \times d_k)$ 的矩阵乘法：
   $$\text{FLOPs}_{AV} = H \times (2 \times S \times S \times d_k) = \mathbf{2 S^2 D} \implies \mathcal{O}(S^2 D)$$
4. **单层 MHA 总计算量**：

$$\text{Total FLOPs}_{\text{MHA}} = 8 S D^2 + 4 S^2 D$$

---

### 2. 计算瓶颈体制转移（Regime Shift Analysis）

```text
MHA 计算量主导项随序列长度 S 的变化：
FLOPs
  ▲
  │                                    /  O(S² D) Attention 矩阵乘法
  │                                   /   (长文本场景，二次方爆炸)
  │                                  /
  │            O(S D²) 线性投影     /
  │           (短文本场景，占主导) /
  │         ─────────────────────/
  │                             /
  └────────────────────────────┴─────────────► 序列长度 S
                             S ≈ 2D (临界交叉点)
```

- **短序列常规体制（$S < 2D$，如 $S=2048, D=4096$）**：
  $8 S D^2 > 4 S^2 D$，**线性投影 $O(S D^2)$ 占据绝大部分计算量**（占比 $>80\%$）。此时优化重点是 GEMM 矩阵乘法效率。
- **长序列长文本体制（$S \gg D$，如 $S=32K \sim 128K, D=4096$）**：
  $4 S^2 D \gg 8 S D^2$，**注意力矩阵计算 $O(S^2 D)$ 呈二次方爆炸并成为绝对算力瓶颈**。此时必须依赖 FlashAttention、稀疏注意力或线性注意力进行优化。

---

## 模块四：自回归推理机制与 KV Cache 显存模型

### 1. Prefill 阶段 vs. Decode 阶段

大模型推理在计算特征上分为两个截然不同的阶段：

```text
推理双阶段特征对比：
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ 推理阶段                │ 硬件行为与瓶颈特征                                                     │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. Prefill 阶段         │ • 输入所有 Prompt Token（长序列），全并行计算 Q, K, V                  │
│    (Prompt 预填充)      │ • 填充并生成初始 KV Cache                                              │
│                         │ • 算术强度高，属于**算力受限（Compute-Bound）**                        │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. Decode 阶段          │ • 每次只输入上一步生成的 1 个 Token ($x_t \in \mathbb{R}^{1 \times D}$)│
│    (Token 逐字生成)     │ • 生成当前步的 $q_t, k_t, v_t$，将 $k_t, v_t$ 追加到 KV Cache 末尾     │
│                         │ • 每次生成 1 个 Token 都需从显存搬运整个历史 KV Cache 与全部权重       │
│                         │ • 算术强度极低（≈ 1 FLOP/Byte），属于**显存带宽受限（Memory-Bound）** │
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

#### 为什么不需要 Cache Query（$Q$）？
- 当前时间步 $t$ 生成的查询向量 $q_t \in \mathbb{R}^{1 \times D}$，只需要与历史所有 Key 向量 $K_{\le t}$ 计算注意力得分；
- 在下一个时间步 $t+1$，新生成的 Token 会产生全新的查询向量 $q_{t+1}$；
- **历史上的旧查询向量 $q_1, q_2, \dots, q_t$ 永远不会再被未来任何步骤使用**，因此 $Q$ 的生命周期仅在当前时间步内，随用随弃，无需占用显存缓存。

---

### 2. KV Cache 显存占用精确数学公式

对于批量大小 $B$，当前上下文长度 $S$，模型总层数 $L$，KV 头数 $H_{KV}$，每个头的维度 $d_k$，每个参数占用字节数 $b$（例如 FP16/BF16 占用 $b=2$ 字节）：

$$\text{Memory}_{\text{KVCache}} = 2 \times B \times S \times L \times H_{KV} \times d_k \times b \quad \text{Bytes}$$

> *公式前面的系数 $2$ 代表 Key 和 Value 两个张量。*

#### 工业级实例测算（LLaMA-3-70B）
- 参数配置：$L=80, D=8192, H_Q=64, H_{KV}=8 \text{ (GQA)}, d_k=128, b=2 \text{ (BF16)}$
- 单 Token 的 KV Cache 显存消耗：
  $$\text{Per-Token Memory} = 2 \times 80 \times 8 \times 128 \times 2 = 327,680 \text{ Bytes} \approx \mathbf{320 \text{ KB / Token}}$$
- 当并发批次 $B=64$，上下文长度 $S=8192$ 时：
  $$\text{Total KV Cache} = 64 \times 8192 \times 320 \text{ KB} \approx \mathbf{167.77 \text{ GB}}$$
  **KV Cache 显存甚至直接超过了 70B 模型本身的权重显存（140 GB）！**

---

## 模块五：长序列与硬件感知注意力优化体系

### 1. 长序列 Attention 核心矛盾与三大物理墙

当序列长度进入长文本体制（$S \gg 2D$）时，注意力矩阵交互的二次方计算与显存占用超越线性投影，模型在训练与 Serving 阶段同时撞上三大物理墙：

```text
长序列三大物理瓶颈特征：
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│ 1. 算力二次方 (Compute) │ 2. 训练激活显存 (Memory)│ 3. 推理 KV Cache (IO带宽│
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ • S 从 4K 拓展到 128K:  │ • 物化 S×S 的 Logits 与 │ • 显存容量 O(B·S·L·d)   │
│   长度扩大 32 倍         │   Attention Score 矩阵   │   呈线性持续膨胀        │
│ • Attn FLOPs 暴增 1024倍│ • 经典实现 O(S²) 导致   │ • Decode 每生成 1 Token │
│ • Prefill 耗时呈二次方  │   GPU 发生显存溢出(OOM) │   需读取全量历史 KV     │
│   剧烈爆炸              │ • 反向传播需保留得分梯度│ • 算术强度极低，带宽受限│
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

为了从根本上化解这一矛盾，业界形成了**系统补丁 $\to$ 三大算法路线（改计算复杂度 / 改系统切分）$\to$ 混合收敛形态（Hybrid）**的演进脉络：

![[assets/attention-efficiency-landscape.png|长序列 Attention 效率演进脉络]]

---

### 2. 系统/算子层补丁：FlashAttention（Exact，不改变渐近复杂度）

FlashAttention（Dao et al.）是现代大模型基础设施级系统优化，核心特征为**数学结果完全等价（Exact Attention，零精度损失）**。

#### (1) 三大核心机制
- **Tiling（SRAM 分块平铺）**：将输入 $Q, K, V$ 划分为适合 GPU 片上高速 SRAM（通常为 100KB~228KB/SM）大小的子块，矩阵乘法与归一化计算均在 SRAM 内部完成；
- **Online Softmax（增量动态归一化）**：维护流式局部最大值 $m_i$ 与归一化因子 $l_i$，在流式加载子块时动态更新局部 Attention 输出，**彻底消除在慢速高带宽显存（HBM）中显式读写 $S \times S$ 激活值矩阵的过程**；
- **Recomputation in Backward（反向重算）**：反向传播时不保存前向的 $S \times S$ 激活图，而在 SRAM 中极速重算，将训练激活显存从 $\mathcal{O}(S^2)$ 压低至 $\mathcal{O}(S D)$。

#### (2) 物理边界与权衡
- **解决的问题**：消除训练期 $\mathcal{O}(S^2)$ 激活值显存 OOM 危机；将 HBM 访存复杂度从 $\mathcal{O}(S^2)$ 降为 $\mathcal{O}(S)$，MFU 提升 2~4 倍；
- **无法解决的问题**：**总计算量依然严格为 $\mathcal{O}(S^2 D)$**；自回归 Decode 仍需遍历加载全量历史 KV；1M 超长序列的 Prefill 二次方耗时依然存在。

---

### 3. 三大效率路线：改算法复杂度与改系统切分

#### 路线 A：稀疏注意力（Sparse Attention，剪枝图）
- **核心思想**：切断注意力全连接图中的绝大部分边，复杂度从 $\mathcal{O}(S^2)$ 降至 $\mathcal{O}(S \cdot k)$ 或 $\mathcal{O}(S\sqrt{S})$。
- **静态规则稀疏（Longformer / BigBird）**：人工组合滑动局部窗口（Local Window）+ 跨步空洞窗口（Dilated Window）+ 全局标记（Global Tokens）。局限在于硬编码规则无法自适应非规则长程依赖。
- **硬件对齐原生稀疏（DeepSeek NSA: Native Sparse Attention）**：
  1. **Compressed Tokens（粗筛视野）**：将相邻 Token 块通过轻量池化压缩为单向量，Query 先在粗粒度扫描定位潜在相关区域；
  2. **Selected Tokens（Top-$k$ 块交互）**：仅将粗筛得分最高的块载入高速缓存进行精确细粒度交互；
  3. **Sliding Window（局部精细上下文）**：对邻近若干 Token 保持全注意力。
- **优缺点**：保留 Softmax 指数放大与注意力锐度；但必须做块级硬件对齐，否则离散访存开销会抵消算力节约。

#### 路线 B：低秩与核化线性注意力（Linear Attention & Delta Rule，改写结合律）
- **核心思想**：利用非线性映射 $\phi(\cdot)$ 解耦 Softmax 为内积 $\phi(Q)\phi(K)^T$，借由乘法结合律调整计算顺序：
  $$\text{Standard: } (Q K^T) V \in \mathcal{O}(S^2 D) \implies \text{Linear: } \phi(Q) \left(\phi(K)^T V\right) \in \mathcal{O}(S \cdot D^2)$$
- **自回归推理 RNN 恒定态**：
  $$S_t = S_{t-1} + \phi(k_t) v_t^T \in \mathbb{R}^{d \times d}, \quad o_t = \phi(q_t) S_t$$
  单步推理维护固定维度状态 $S_t$，**显存与计算复杂度均为 $\mathcal{O}(1)$**，无需线性膨胀的 KV Cache。
- **突破“容量饱和”：Delta 学习规则（DeltaNet / RetNet）**：
  纯累加缺乏擦除机制会导致历史无关信息填满状态（Attention Dilution）。引入经典关联记忆擦除机制：
  $$W_t = W_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$
  新 Key 写入前先从记忆矩阵中扣除旧值投影，辅以 Chunkwise 并行扫描算子，召回率逼近标准 Softmax。
- **工业落地形态**：**Hybrid 混合架构**（如 Jamba、Nemotron-4），周期性交替堆叠 SSM/线性层与因果 Attention 层，兼顾恒定吞吐与复杂检索精度。

#### 路线 C：分块与系统级并行（Chunking & RingAttention，改系统切分）
- **核心思想**：算法上固定局部注意力块 $B \ll S$，或系统架构上将长序列切分分散至多张 GPU 流转。
- **RingAttention（Liu et al.）**：
  - 将序列切分为 $P$ 块分布于 $P$ 张 GPU，每卡持有本地 $Q$；
  - 核心流转：$K, V$ 块通过 GPU 环形拓扑（Ring P2P）跨卡流动；
  - **计算通信完全重叠（Compute-Comm Overlap）**：计算当前块注意力时，底层异步流水线同步发送与接收下一块 $K, V$，网络传输耗时被计算隐藏；
  - **Chunked Prefill**：将超长 Prompt 切片打散分批调度，防止单次大 Prefill 独占计算资源导致 Decode 任务出现排队毛刺（Head-of-Line Blocking）。

---

### 4. 架构级头数压缩与 Serving 显存优化

```text
MHA vs MQA vs GQA 架构对比：
MHA (Multi-Head Attention):        MQA (Multi-Query Attention):       GQA (Grouped-Query Attention):
Q Heads:   [1] [2] [3] [4] [5] [6] [7] [8]  Q Heads:   [1] [2] [3] [4] [5] [6] [7] [8]  Q Heads:   [1][2] [3][4] [5][6] [7][8]
K/V Heads: [1] [2] [3] [4] [5] [6] [7] [8]  K/V Heads: [         1 (共享)          ]  K/V Heads:  [ 1 ]  [ 2 ]  [ 3 ]  [ 4 ]
(KV 缓存 100%, 显存占用最大)                (KV 缓存压缩至 1/H, 表达力略损)             (LLaMA-3 标配: 兼顾容量与吞吐)
```

- **MHA**：$H_Q = H_{KV}$。每个 Query 独享一组 Key/Value，表达力最强，但 KV Cache 显存开销最大；
- **MQA**：$H_Q = H, H_{KV} = 1$。所有 Query 头共享 1 组 Key/Value，KV Cache 骤降 $H$ 倍，但长程复杂推理表征容量有所损耗；
- **GQA**：$H_Q = H, H_{KV} = G$（$1 < G < H$）。Query 头分为 $G$ 组（如 LLaMA-3 的 64:8），实证表明 GQA 能以接近 MHA 99% 的性能达成接近 MQA 的吞吐与带宽压缩比；
- **PagedAttention（vLLM）**：借鉴虚拟内存分页，将逻辑连续 KV 张量映射到离散物理内存页（如 16 Tokens/页），显存碎片从 $60\% \sim 80\%$ 压至 $<4\%$；
- **KV Cache 量化（FP8 / INT4）**：将缓存数值从 16-bit 压缩至 8-bit 或 4-bit，带宽与容量需求减半至四分之一。

---

### 5. 四大机制横向对比与工业收敛形态

| 机制 / 范式 | 计算时间复杂度 | 训练显存复杂度 | 推理 KV 状态显存 | 核心优势 | 核心局限与工程代价 |
|---|---|---|---|---|---|
| **Standard (+FlashAttention)** | $\mathcal{O}(S^2 D)$ | $\mathcal{O}(S D)$ | $\mathcal{O}(B \cdot S \cdot L \cdot d_k)$ 线性膨胀 | 精确 Softmax、零质量损失、长程关联极强 | 算力与 Prefill 仍呈二次方，超长序列吞吐低 |
| **Sparse / NSA (DeepSeek)** | $\mathcal{O}(S \cdot k \cdot D)$ | $\mathcal{O}(S \cdot k)$ | $\mathcal{O}(B \cdot k \cdot L \cdot d_k)$ 稀疏受限 | 保持 Softmax 尖锐度，NIAH 大海捞针检索极高 | 依赖块级硬件对齐定制算子，非连续访存 |
| **Linear / DeltaNet** | $\mathcal{O}(S \cdot D^2)$ | $\mathcal{O}(S D)$ | $\mathcal{O}(D^2)$，单步 $\mathcal{O}(1)$ | 吞吐极高，显存不随序列膨胀 | 纯核化易容量饱和；少样本 ICL 弱于标准注意力 |
| **Ring / Chunked Parallel** | 多卡 $\mathcal{O}(S^2 D / P)$ | 单卡 $\mathcal{O}(B_{\text{chunk}} D)$ | 分布式切片持有 | 打破单卡显存上限，支持百万级长文本 | 强依赖高速互联（NVLink/RoCE），网络易成瓶颈 |

#### 工业收敛体系
1. **底层访存与网络**：FlashAttention 负责单卡 SRAM-HBM 极速访存，RingAttention 负责多卡序列并行扩展；
2. **架构与算法协同**：GQA 压缩推理头数，结合 NSA 原生动态稀疏或 SSM/DeltaNet 周期混合堆叠（Hybrid）。

---

## 模块六：核心工程公式与推导速查清单

| 核心工程指标 | 精确计算式 | 典型工业规模基准 (LLaMA-3-70B, $S=4096$) |
|---|---|---|
| **单层 MHA 投影 FLOPs** | $\text{FLOPs}_{\text{proj}} = 8 S D^2$ | $8 \times 4096 \times 8192^2 \approx \mathbf{2.20 \text{ TFLOPs}}$ |
| **单层 MHA 注意力 FLOPs** | $\text{FLOPs}_{\text{attn}} = 4 S^2 D$ | $4 \times 4096^2 \times 8192 \approx \mathbf{0.55 \text{ TFLOPs}}$ |
| **单 Token KV Cache 容量** | $\text{Memory}_{\text{token}} = 2 L H_{KV} d_k b$ | $2 \times 80 \times 8 \times 128 \times 2 = \mathbf{320 \text{ KB / Token}}$ |
| **自回归单步算术强度** | $\text{Operational Intensity} \approx \frac{2 \times \text{Params}}{\text{Params} \times b + \text{KVCache}} \approx 1$ | 处于绝对 Memory-Bound 状态，吞吐上限受限于显存带宽 |
| **FlashAttention 访存优化比** | HBM 读写从 $\mathcal{O}(S^2)$ 降低到 $\mathcal{O}(S)$ | 激活显存开销从 $\mathcal{O}(S^2)$ 压至 $\mathcal{O}(S D)$，MFU 提升 2~4 倍 |
| **线性注意力单步推理复杂度** | 状态更新 $S_t = S_{t-1} + k_t v_t^T \in \mathbb{R}^{d \times d}$ | 时间 $\mathcal{O}(1)$，隐状态显存 $\mathcal{O}(D^2)$，无线性膨胀 Cache |

### 四条架构与系统工程法则
1. **FlashAttention 不改 FLOPs**：其本质是硬件层访存优化，解决了激活显存 OOM 与 Memory-Bound 算子等待问题，长文本的二次方计算瓶颈必须依靠稀疏或线性注意力化解；
2. **稀疏注意力的成败在于内存对齐**：必须采用如 DeepSeek NSA 的块级对齐结构，离散非连续寻址会抵消算法 FLOPs 节约；
3. **线性注意力必须具备记忆擦除**：单纯累加会导致信息弥散饱和，引入 Delta 规则（DeltaNet）动态擦除旧记忆是逼近 Softmax 表达力的关键；
4. **RingAttention 是系统切分而非近似**：通过环形通信隐藏传输时延，在集群上保持数学结果完全等价（Exact Attention）。
