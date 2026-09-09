# ML Coding 01B · Transformer 架构变体：MHA 张量维度推导、FLOPs 分解与 KV Cache 硬件优化

在大语言模型（LLM）与 Transformer 架构工程中，深入理解多头注意力（MHA）的精确张量流动、计算复杂度（FLOPs）的算力瓶颈转移、以及自回归推理阶段 KV Cache 的显存瓶颈与硬件感知加速（FlashAttention、GQA、PagedAttention），是模型架构设计、性能调优与大规模 Serving 部署的核心基本功。

本篇系统梳理 Transformer 算子底层的 5 大核心体系：
1. **Transformer 架构分类学（Encoder-Only vs Decoder-Only vs Encoder-Decoder）**
2. **多头注意力（MHA）数学推导、张量形状演化与执行流水线**
3. **MHA 计算复杂度 FLOPs 严密分解与长短序列体制转移（Regime Shift）**
4. **自回归推理动态与 KV Cache 显存容量精确数学模型**
5. **长序列与硬件感知注意力优化体系（效率演进全景、MQA / GQA、FlashAttention SRAM 分块平铺、PagedAttention 与 KV 量化）**

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

### 三大架构形态全景对比

| 架构形态 | 注意力机制与掩码模式 | 输入与生成范式 | KV Cache 需求 | 工业界典型代表 | 最佳适用场景 |
|---|---|---|---|---|---|
| **Encoder-Only** | 全双向自注意力（$M_{ij} = 0$） | 非自回归，单次前向并行处理全部 $S$ 个 Token | **无需 KV Cache**（单次前向完成） | BERT, RoBERTa, DeBERTa | 文本分类、命名实体识别（NER）、向量表征（Embedding） |
| **Decoder-Only** | 因果下三角自注意力（$j > i$ 时 $M_{ij} = -\infty$） | 自回归生成，逐 Token 依次依赖历史上下文 | **必须维护 KV Cache**（避免重复计算历史 Key/Value） | GPT-4, LLaMA-3, Mistral, Qwen, DeepSeek | 通用大语言模型、指令遵循、代码生成、多步推理 |
| **Encoder-Decoder** | Encoder 双向 + Decoder 因果自注意力 + **Cross-Attention（跨注意力）** | 双向编码输入上下文，自回归生成目标序列 | **需双份 KV Cache**（Encoder 静态缓存 + Decoder 动态缓存） | T5, BART, Whisper, 原始 Transformer | 机器翻译、文本摘要、语音识别（ASR） |

#### Cross-Attention（跨注意力机制）工作原理
在 Encoder-Decoder 架构中：
- **Query（$Q$）**：来源于 Decoder 上一层的隐层状态（表示“当前解码器需要关注什么”）；
- **Key（$K$）与 Value（$V$）**：来源于 Encoder 顶层的输出表征（表示“输入上下文提供了哪些全局信息”）；
- **执行逻辑**：Encoder 的 $K, V$ 在 Prefill 阶段只需计算一次，随后在整个自回归解码过程中被所有解码步骤反复共享读取。

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

在标准 Scaled Dot-Product Attention 中：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
整个计算由两部分构成：
- **线性投影**：$Q, K, V, W_O$ 四次矩阵乘法，时间复杂度 $\mathcal{O}(S D^2)$；
- **注意力交互**：$Q K^T$ 得分计算与 $\tilde{A} V$ 聚合，时间复杂度 $\mathcal{O}(S^2 D)$，中间矩阵空间复杂度 $\mathcal{O}(S^2)$。

当序列长度 $S \gg D$ 时，计算与显存的瓶颈发生根本性转移，长上下文在 Prefill 与 Decode 阶段同时撞上**三大物理墙**：

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

为了从根本上化解这一矛盾，学术界与工业界形成了**系统补丁 $\to$ 三大算法路线（改计算复杂度 / 改系统切分）$\to$ 混合收敛形态（Hybrid）**的演进脉络：

![长序列 Attention 效率演进脉络](./assets/attention-efficiency-landscape.png)

---

### 2. 系统/算子层补丁：FlashAttention（Exact，不改变渐近复杂度）

FlashAttention（Dao et al.）是现代大模型训练与推理的基础设施级系统优化。它的核心特质是**数学完全等价（Exact Attention，零精度损失）**。

#### (1) 核心机制
- **Tiling（SRAM 分块平铺）**：将输入 $Q, K, V$ 划分为适合 GPU 片上高速 SRAM（通常为 100KB~228KB/SM）大小的子块，所有矩阵乘法与归一化计算均在 SRAM 内部完成；
- **Online Softmax（增量归一化）**：传统 Softmax 需要物化整个序列求最大值与指数和；Online Softmax 通过维护局部缩放因子 $m_i$ 与 $l_i$，在流式加载子块时动态更新局部 Attention 输出，**彻底消除了在慢速高带宽显存（HBM）中显式写入与读取 $S \times S$ 激活值矩阵的过程**；
- **Recomputation in Backward（反向重算）**：反向传播时不保存前向的 $S \times S$ 激活图，而是在 SRAM 中直接快速重算，将训练激活显存从 $\mathcal{O}(S^2)$ 压缩到 $\mathcal{O}(S D)$。

#### (2) 权衡分析：解决了什么 vs. 解决不了什么
- **$\checkmark$ 解决了什么**：
  - 消除了训练期间的 $\mathcal{O}(S^2)$ 激活值显存 OOM 危机；
  - 减少了大量低效的 HBM 访存往返，将算子从 Memory-Bound 状态推向高算力利用率（MFU 提升 2~4 倍）；
  - 输出与标准 Attention 严格一致，无需修改模型权重或重训。
- **$\times$ 解决不了什么**：
  - **总计算量（FLOPs）仍然是严格的 $\mathcal{O}(S^2 D)$**，算法渐近复杂度并未降低；
  - 自回归 Decode 阶段仍然需要逐步加载全量历史 KV Cache；
  - 当上下文长度向 1M 推进时，Prefill 阶段的二次方计算耗时依然无法承受。
- **核心工程结论**：**FlashAttention 是硬件层/IO 访存优化，长序列算法与吞吐瓶颈必须依靠三大效率路线（Sparse / Linear / Chunking）彻底解决。**

---

### 3. 三大效率路线：改算法复杂度与改系统切分

#### 路线 A：稀疏注意力（Sparse Attention，剪枝图）
* **核心思想**：切断全连接注意力图中的大部分边，每个 Query 只与特定的 Key 集合子集计算相关性，将复杂度从 $\mathcal{O}(S^2)$ 降为 $\mathcal{O}(S \cdot k)$ 或 $\mathcal{O}(S\sqrt{S})$。
* **早期静态规则稀疏（Static Heuristics）**：
  * **Longformer / BigBird**：手工设计三类注意力模式的组合——局部滑动窗口（Local Window，捕捉相邻短语语法结构）+ 跨步空洞窗口（Strided / Dilated，捕捉中程语境）+ 全局固定标记（Global Tokens，如 `[CLS]` 负责汇聚全篇语义）。
  * **局限**：硬编码规则无法自适应复杂的动态语义跳跃，对不规则依赖的建模能力受限。
* **现代前沿：端到端硬件对齐原生稀疏（DeepSeek NSA: Native Sparse Attention）**：
  * 摒弃传统的细粒度非连续索引，改为**块级对齐（Block-Aligned）的原生动态稀疏**：
    1. **Compressed Tokens（压缩块粗视野）**：将相邻 Token 块通过轻量池化压缩为单向量，Query 先在粗粒度级别扫描定位潜在相关的上下文区域；
    2. **Selected Tokens（Top-$k$ 细粒度检索）**：基于粗筛结果，只将得分最高的几个关键块调入高速缓存进行精确细粒度注意力交互；
    3. **Sliding Window（局部精细上下文）**：对邻近的最近若干 Token 保持全注意力。
  * **工程收益**：在 64K+ 长文本下实现了数倍的预训练与推理加速，且在“大海捞针”（Needle In A Haystack, NIAH）基准上保持极高锐度。
* **优缺点权衡**：
  * $\checkmark$ 保留 Softmax 指数放大特性，注意力聚焦度（Sharpness）高，复杂长程检索能力强；
  * $\times$ 动态稀疏索引若未做块对齐，非连续离散访存开销会抵消算力节约。

#### 路线 B：低秩与核化线性注意力（Linear Attention & Delta Rule，改写结合律）
* **核心思想**：通过非线性映射 $\phi(\cdot)$ 将 Softmax 解耦为内积 $\text{Sim}(Q, K) = \phi(Q) \phi(K)^T$。利用矩阵乘法的结合律改写计算顺序：
  $$\text{Standard: } (Q K^T) V \in \mathcal{O}(S^2 D) \implies \text{Linear: } \phi(Q) \left(\phi(K)^T V\right) \in \mathcal{O}(S \cdot D^2)$$
* **流派与演进**：
  * **Linformer**：将 $K, V$ 的序列维度通过投影矩阵压缩为固定的 $k \ll S$；
  * **Performer / Linear Transformer**：利用正随机特征（Positive Random Features）近似高斯核函数。
* **自回归推理的 RNN 恒定态**：
  在自回归生成中，Key 与 Value 的聚合可以写成流式累加状态：
  $$S_t = S_{t-1} + \phi(k_t) v_t^T \in \mathbb{R}^{d \times d}, \quad o_t = \phi(q_t) S_t$$
  此时单步推理仅需维护固定维度的隐状态 $S_t$，**推理显存与计算复杂度均为 $\mathcal{O}(1)$**，无需维护线性膨胀的 KV Cache！
* **致命缺陷与 Delta 规则破局（DeltaNet / RetNet / Mamba-2）**：
  * **纯线性累加的“容量饱和与弥散”**：由于 $S_t = S_{t-1} + k_t v_t^T$ 只有写入没有擦除，随着序列增长，历史无关信息迅速填满隐状态矩阵，导致对新信息的辨识力彻底丧失（Attention Dilution）。
  * **Delta 学习规则（DeltaNet / RetNet）**：引入关联记忆的经典 Delta 擦除更新机制：
    $$W_t = W_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$
    若新 Key 与历史 Key 冲突，则先从记忆矩阵中扣除旧值的投影，再写入新 Value。
  * **并行训练突破**：Chunkwise 算法在块内将 Rank-1 更新转化为下三角可逆系统，块间使用高速 Associative Scan 算子结合 SRAM 融合，训练复杂度达 $\mathcal{O}(S D^2)$ 全并行，召回率逼近标准 Softmax。
* **优缺点权衡**：
  * $\checkmark$ 推理吞吐极高，无 KV Cache 显存暴涨，无限生成长度友好；
  * $\times$ 纯线性注意力的联想记忆（Associative Recall）与少样本上下文学习（In-Context Learning）能力仍弱于标准注意力；
  * **工业落地形态**：**混合架构（Hybrid）**。如 **Jamba** 与 **Nemotron-4**，每隔若干线性/SSM 层插入一层标准因果 Attention，以极小显存代价换取全量检索精度。

#### 路线 C：分块与系统级并行（Chunking & RingAttention，改系统切分）
* **核心思想**：算法上固定局部注意力块 $B \ll S$，或者在系统架构层面将长序列跨 GPU 进行切片通信，单卡峰值显存控制在 $\mathcal{O}(B \cdot D)$。
* **算法端分块**：
  * **Transformer-XL**：维护固定的向前段落 Memory Cache，跨块截断反向传播梯度，仅保留前向隐藏状态流。
* **分布式系统前沿：RingAttention（Liu et al.）**：
  * 将长序列切分为 $P$ 个分块分散在 $P$ 张 GPU 上；
  * 核心流转：每张 GPU 持有本地的 $Q$，通过环形通信拓扑（Ring P2P）在 GPU 之间流动传递 $K, V$ 块；
  * **计算与通信重叠（Compute-Communication Overlap）**：当 GPU 计算第 $i$ 块的 Attention 时，底层异步流水线同步向邻居节点发送与接收下一块的 $K, V$，计算耗时有效掩盖网络传输开销；
  * **Chunked Prefill（vLLM / SGLang）**：将超长 Prompt 切片分批进入调度队列，防止单次大 Prefill 垄断计算单元导致排队任务的 Decode 延迟发生毛刺（Head-of-Line Blocking）。
* **优缺点权衡**：
  * $\checkmark$ 彻底打破单卡显存上限，单卡即可处理百万上下文；
  * $\times$ 算法分块会损失跨块直接关联；RingAttention 则对跨卡网络互联（NVLink / RoCE）带宽要求极高。

---

### 4. 架构级头数压缩（MHA vs MQA vs GQA）与 Serving 显存优化

除了算法层面的改动，现代大模型在架构层面通过精简 Key/Value 头数，直接在物理上压缩自回归推理的显存开销：

```text
MHA vs MQA vs GQA 架构对比：
MHA (Multi-Head Attention):        MQA (Multi-Query Attention):       GQA (Grouped-Query Attention):
Q Heads:   [1] [2] [3] [4] [5] [6] [7] [8]  Q Heads:   [1] [2] [3] [4] [5] [6] [7] [8]  Q Heads:   [1][2] [3][4] [5][6] [7][8]
K/V Heads: [1] [2] [3] [4] [5] [6] [7] [8]  K/V Heads: [         1 (共享)          ]  K/V Heads:  [ 1 ]  [ 2 ]  [ 3 ]  [ 4 ]
(KV 缓存 100%, 显存占用最大)                (KV 缓存压缩至 1/H, 表达力略损)             (LLaMA-3 标配: 兼顾容量与吞吐)
```

- **MHA**：$H_Q = H_{KV}$。每个 Query 头对应独立的 Key/Value 头，表达能力最强，但 KV Cache 显存最大；
- **MQA**：$H_Q = H, H_{KV} = 1$。所有 Query 头共享单一组 Key/Value 头，KV Cache 显存骤降 $H$ 倍，但大幅降低了模型在长文本和复杂多轮推理下的多头表达容量；
- **GQA**：$H_Q = H, H_{KV} = G$（$1 < G < H$）。将 Query 头划分为 $G$ 组，每组共享一组 Key/Value 头（如 LLaMA-3 的 64:8 分组）。实证表明，**GQA 能够以接近 MHA 99% 的模型效果，获得接近 MQA 的推理显存带宽与吞吐提升**。

#### PagedAttention 与 KV Cache 量化
- **PagedAttention（vLLM 核心引擎）**：借鉴操作系统分页机制，将连续的 KV 张量映射到离散物理内存页（如 16 Tokens/Block），将显存碎片从 $60\% \sim 80\%$ 压缩到 $<4\%$；
- **KV Cache 量化（FP8 / INT4）**：将缓存值量化存储，显存需求直接减半甚至减少 $75\%$，在显存带宽受限的 Decode 阶段显著倍增推理并发。

---

### 5. 四大机制横向对比与工业收敛形态

| 机制 / 范式 | 计算时间复杂度 | 训练显存复杂度 | 推理 KV 状态显存 | 核心优势 | 核心局限与工程代价 |
|---|---|---|---|---|---|
| **Standard (+FlashAttention)** | $\mathcal{O}(S^2 D)$ | $\mathcal{O}(S D)$ | $\mathcal{O}(B \cdot S \cdot L \cdot d_k)$ 线性膨胀 | 精确 Softmax、无损质量、长程关联强 | 算力与 Prefill 仍二次方，长序列吞吐低 |
| **Sparse / NSA (DeepSeek)** | $\mathcal{O}(S \cdot k \cdot D)$ | $\mathcal{O}(S \cdot k)$ | $\mathcal{O}(B \cdot k \cdot L \cdot d_k)$ 稀疏受限 | 保持 Softmax 尖锐度，NIAH 检索极强，64K+ 解码加速 | 需深度定制硬件对齐算子，非连续访存 |
| **Linear / DeltaNet** | $\mathcal{O}(S \cdot D^2)$ | $\mathcal{O}(S D)$ | $\mathcal{O}(D^2)$，单步 $\mathcal{O}(1)$ | 吞吐极高，显存不随序列线性增长 | 纯核化缺乏擦除易弥散，ICL 联想记忆较弱 |
| **Ring / Chunked Parallel** | 多卡 $\mathcal{O}(S^2 D / P)$ | 单卡 $\mathcal{O}(B_{\text{chunk}} D)$ | 分布式切片持有 | 打破单卡显存墙，支持超百万上下文 | 强依赖高速互联（NVLink/RoCE），通信易成瓶颈 |

#### 工业收敛形态：Hybrid + 算法硬件协同
在千万级、亿级 Token 的前沿生产系统中，效率优化已收敛为多层级的协同体系：
- **底层硬件与通信地基**：使用 **FlashAttention** 实现单卡 SRAM-HBM 访存优化，结合 **RingAttention** 进行跨卡环形序列切分；
- **顶层算法与模型结构**：采用 **GQA** 压缩推理头数，并引入 **NSA 原生稀疏** 或 **SSM/DeltaNet 与标准 Attention 混合堆叠（Hybrid）**；
- **技术本质提炼**：
  * *Flash 不改 FLOPs，解决的是 IO 墙与激活 OOM；*
  * *Sparse 保留 Softmax 锐度，精髓在于块级硬件对齐；*
  * *Linear 改写结合律，引入 Delta 规则才具备动态记忆擦除；*
  * *Ring 将二次方算力摊平到分布式节点，本质仍是 Exact Attention。*

---

## 模块六：核心工程公式与推导速查清单

### Q1：计算一个序列长度为 $S=4096$、隐藏维度 $D=4096$ 的单层 MHA 线性投影和注意力计算的 FLOPs 分别是多少？
> **答**：
> 1. 线性投影（4 次矩阵乘法）：
>    $$\text{FLOPs}_{\text{proj}} = 8 S D^2 = 8 \times 4096 \times (4096)^2 = 8 \times 4096 \times 1.678 \times 10^7 \approx \mathbf{5.498 \times 10^{11} \text{ FLOPs} \ (550 \text{ GFLOPs})}$$
> 2. 注意力矩阵计算（$QK^T$ 与 $\tilde{A}V$）：
>    $$\text{FLOPs}_{\text{attn}} = 4 S^2 D = 4 \times (4096)^2 \times 4096 \approx \mathbf{2.749 \times 10^{11} \text{ FLOPs} \ (275 \text{ GFLOPs})}$$
> 3. 在此序列长度下（$S=D$），线性投影计算量约是注意力矩阵计算量的 2 倍。

### Q2：为什么 FlashAttention 能够在数学结果完全等价（Exact Attention）的前提下，实现 2~4 倍的速度提升？
> **答**：
> 因为 GPU 计算单元（Tensor Cores）的速度远远快于显存带宽（HBM）。标准 Attention 的瓶颈不在于算力，而在于反复向慢速 HBM 读写 $S \times S$ 的中间激活值矩阵。FlashAttention 通过 Tiling 分块将所有计算锁在片上极速 SRAM 中完成，彻底消除了中间矩阵的 HBM 访存往返，将 HBM 访存复杂度从 $O(S^2)$ 降低到 $O(S)$。
