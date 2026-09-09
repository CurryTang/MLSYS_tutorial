# ML Coding 03 · 注意力机制全家桶：从 MHA 到 GQA、滑动窗口、线性注意力、KV Cache 与 Flash Attention

MLCoding01 已经搭出了 stable softmax、scaled dot-product attention 和带 RoPE 的 causal multi-head self-attention，这是 GPT 系模型跑起来所需的最小集合。但面试里被问到的 attention 远不止这一种：encoder 用的是双向 MHA，翻译模型用 cross-attention，LLaMA 2/3 用 GQA 压 KV cache，Mistral 用滑动窗口砍复杂度，推理引擎离不开 KV cache 和 flash attention。这篇笔记只讲这些变体本身的机制和陷阱，不重复 MLCoding01 已经讲过的 softmax、mask 语义和 RoPE 推导。

这里的题目选取参考了开源题库 [TorchCode](https://github.com/duoan/TorchCode)（一套面向 PyTorch 面试的自动判题练习集）覆盖的主题，代码和讲解均为独立编写。

## 复用前置

后面所有练习都假设你已经有 MLCoding01 里的这几个零件，这里只贴出签名作提醒，实现见 MLCoding01：

```python
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor: ...
def scaled_dot_product_attention(Q, K, V, mask=None) -> torch.Tensor: ...
class Linear(nn.Module): ...  # y = x @ weight.T，无 bias
```

## 模块九：注意力机制变体

### Exercise 1 · MultiHeadAttention（双向，非因果）

和 MLCoding01 的 `CausalMultiHeadSelfAttention`相比，这里唯一的实质区别是 mask：BERT、ViT 这类 encoder 结构里每个位置都能看到所有位置，所以 mask 要么是 `None`，要么只用来屏蔽 padding，绝不是下三角矩阵。除此之外，切 head、拼 head、投影的流程完全一样。

面试里常见的坑是把"multi-head"理解成"跑 H 次单头 attention 再拼起来"：概念上没错，但真实实现是把 `(B, T, D)` reshape 成 `(B, H, T, Dh)`，一次性做 batched matmul，而不是写 Python for 循环。reshape 的顺序也有讲究：`d_model` 维必须先按 `(H, Dh)` 拆开再 transpose 到 `(B, H, T, Dh)`，如果先 transpose 再 reshape，各 head 里混进的就不是同一组通道，训练也能跑但学出来的东西是错的。

#### Quick Coding：`MultiHeadAttention`

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        ...

    def forward(self, x, key_padding_mask=None):
        ...
```

<details>
<summary>参考答案</summary>

```python
from einops import rearrange

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, key_padding_mask=None):
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.num_heads)

        mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, T)，True 表示这个位置是有效 token
            mask = key_padding_mask[:, None, None, :]  # 广播到 (B, 1, 1, T)

        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.o_proj(out)
```

</details>

### Exercise 2 · MultiHeadCrossAttention

Cross-attention 把 Q 和 K/V 的来源拆开：Q 来自当前序列（比如 decoder 的 hidden states），K/V 来自另一个序列（比如 encoder 的输出，或者图文模型里的图像特征）。两个序列长度可以完全不同，`T_q != T_kv` 是常态而不是特例。

这里最容易踩的坑是默认 attention 矩阵是方阵。scores 的 shape 是 `(..., T_q, T_kv)`，causal mask、padding mask 的形状都要按这个非方阵的形状去对齐，直接照抄 self-attention 里 `(T, T)` 的 mask 构造会直接 shape mismatch。

#### Quick Coding：`MultiHeadCrossAttention`

```python
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        ...

    def forward(self, x_q, x_kv, key_padding_mask=None):
        ...
```

<details>
<summary>参考答案</summary>

```python
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x_q, x_kv, key_padding_mask=None):
        # x_q:  (B, T_q,  D)  来自 decoder
        # x_kv: (B, T_kv, D)  来自 encoder，T_kv 可以不等于 T_q
        q = rearrange(self.q_proj(x_q), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.k_proj(x_kv), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.v_proj(x_kv), "b t (h d) -> b h t d", h=self.num_heads)

        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]  # (B, 1, 1, T_kv)

        out = scaled_dot_product_attention(q, k, v, mask)  # (B, H, T_q, Dh)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.o_proj(out)
```

`scores` 的 shape 是 `(B, H, T_q, T_kv)`，`attn @ V` 之后又变回 `(B, H, T_q, Dh)`：输出序列长度永远跟 Q 走，跟 K/V 无关，这一点在调试 shape mismatch 时最值得先确认。

</details>

### Exercise 3 · GroupedQueryAttention（GQA）

标准 MHA 里 Q/K/V 各有 `num_heads` 个头，推理时要为每个头都缓存一份 K/V，KV cache 显存随头数线性增长。GQA（LLaMA 2/3 用的方案）把 K/V 的头数砍到 `n_kv_heads < n_q_heads`，每 `group_size = n_q_heads // n_kv_heads` 个 Q 头共享一份 K/V 头，KV cache 显存直接缩小 `n_q_heads / n_kv_heads` 倍，而 Q 的表达能力不变。极端情况 `n_kv_heads = 1` 就是 Multi-Query Attention（MQA）。

真正的陷阱在头对齐方式上。K/V 只投影出 `n_kv_heads` 份，要广播到 `n_q_heads` 份才能和 Q 做 batched matmul，这里必须用 `repeat_interleave`（对应 numpy 的 `repeat`：`[0, 1]` 变成 `[0, 0, 0, 0, 1, 1, 1, 1]`），而不是 `tensor.repeat`（对应 numpy 的 `tile`：`[0, 1]` 变成 `[0, 1, 0, 1, 0, 1, 0, 1]`）。两种写法输出的 shape 完全一样，代码能跑、loss 也能下降，但 `tile` 把第 0、2、4、6 号 Q 头错误地分给了 kv 头 0，第 1、3、5、7 号错误地分给了 kv 头 1。分组关系整体错位，是一种典型的"跑得动但学歪了"的 bug。

#### Quick Coding：`GroupedQueryAttention`

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, device=None, dtype=None):
        ...

    def forward(self, x, mask=None):
        ...
```

<details>
<summary>参考答案</summary>

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_kv_heads * self.d_head, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, num_kv_heads * self.d_head, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.num_kv_heads)
        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.num_kv_heads)

        # 关键一步：repeat_interleave，让 q 头 i 对应 kv 头 i // group_size
        k = torch.repeat_interleave(k, self.group_size, dim=1)  # (B, H, T, Dh)
        v = torch.repeat_interleave(v, self.group_size, dim=1)

        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.o_proj(out)
```

KV cache 显存对比（每层，float16，seq_len = `T`）：

| 方案 | K/V 头数 | 每层 KV cache 大小 |
| --- | --- | --- |
| MHA | `num_heads` | `2 * num_heads * T * d_head * 2 bytes` |
| GQA | `num_kv_heads` | `2 * num_kv_heads * T * d_head * 2 bytes` |
| MQA | `1` | `2 * T * d_head * 2 bytes` |

`repeat_interleave` 是否按 `i // group_size` 分组、而不是 `tile` 那种交错分组，已经用 NumPy 验证过（`np.repeat` 语义等价于 `torch.repeat_interleave`）。

</details>

### Exercise 4 · Sliding Window Attention

Mistral 用的局部注意力：每个 query 只看最近 `window` 个 key（含自己），不看更早的。mask 从下三角变成一条对角带：位置 `i` 只能看 `j`，满足 `i - window < j <= i`。单层复杂度从 `O(T^2)` 降到 `O(T * window)`。

它不是"看不到长距离信息"的近似，而是把长距离依赖交给层数去完成：这一层看不到 `window` 之外的 token，但堆叠 `L` 层之后，第 `L` 层的某个位置间接依赖的输入范围可以到 `L * window`，和 CNN 里堆叠小卷积核扩大感受野是同一个道理。

#### Quick Coding：`sliding_window_attention`

```python
def sliding_window_attention(Q, K, V, window):
    ...
```

<details>
<summary>参考答案</summary>

```python
def sliding_window_mask(T, window, device=None):
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (j > i - window)  # 下三角 & 窗口内

def sliding_window_attention(Q, K, V, window):
    T = Q.shape[-2]
    mask = sliding_window_mask(T, window, device=Q.device)
    mask = mask[None, None, :, :]  # 广播到 (B, H, T, T)
    return scaled_dot_product_attention(Q, K, V, mask)
```

真正省下计算量的实现不会构造完整 `(T, T)` mask 再算满 attention（那样只是省了带宽没省 FLOPs），而是按 `window` 分块只对窗口内的 K/V 做 matmul。这里为了和 `scaled_dot_product_attention` 保持一致先写出定义版本，性能版本的分块思路和下面的 flash attention 是同一套。

</details>

### Exercise 5 · Linear Attention

Softmax attention 的复杂度瓶颈在于必须先算出完整的 `(T, T)` score 矩阵才能归一化。线性注意力用一个正值特征映射 `φ`（比如 `elu(x) + 1`）替换 `softmax(QK^T)`，得到一个可以重新结合律的核函数近似：

```text
softmax(QK^T) V            关联顺序：(Q K^T) V，O(T^2 d)
φ(Q) (φ(K)^T V)             关联顺序：Q (K^T V)，O(T d^2)
```

当 `d << T`（长序列、适中 head_dim）时，后者显著更快，且不需要显式构造 `(T, T)` 矩阵。归一化项同理，从 `sum_j exp(...)` 换成 `φ(Q) · sum_j φ(K_j)`。

要说清楚的是：这不是 flash attention 那种"数学上完全等价、只是计算顺序不同"的加速技巧，而是换了一个相似度核函数：`φ(Q)φ(K)^T` 不等于 `softmax(QK^T)` 的每一项，线性注意力本质上是对 softmax attention 的一种近似（在做核方法意义下的重新参数化），表达能力和精度都会有取舍，这也是它没有完全取代 softmax attention 的原因。

#### Quick Coding：`linear_attention`

```python
def linear_attention(Q, K, V, causal=False):
    ...
```

<details>
<summary>参考答案</summary>

```python
def feature_map(x):
    return F.elu(x) + 1  # 保证非负，且在 x<0 时不会像 relu 一样直接归零

def linear_attention(Q, K, V, causal=False):
    Qp, Kp = feature_map(Q), feature_map(K)  # (..., T, d)

    if not causal:
        kv = torch.einsum("...kd,...ke->...de", Kp, V)      # (..., d, dv)
        k_sum = Kp.sum(dim=-2)                                # (..., d)
        num = torch.einsum("...qd,...de->...qe", Qp, kv)     # (..., T, dv)
        den = torch.einsum("...qd,...d->...q", Qp, k_sum).unsqueeze(-1)
        return num / den

    # 因果版本：用累积和代替全量矩阵乘，逐 token 更新 O(d * dv) 的状态
    outer = torch.einsum("...tk,...tv->...tkv", Kp, V)  # (..., T, d, dv)
    S_cum = torch.cumsum(outer, dim=-3)                   # running sum_{j<=t} phi(K_j) V_j^T
    z_cum = torch.cumsum(Kp, dim=-2)                      # running sum_{j<=t} phi(K_j)
    num = torch.einsum("...td,...tdv->...tv", Qp, S_cum)
    den = torch.einsum("...td,...td->...t", Qp, z_cum).unsqueeze(-1)
    return num / den
```

`φ(Q)(φ(K)^T V) == (φ(Q)φ(K)^T) V` 这个结合律恒等式、以及因果版本"循环逐步累加"和"cumsum 批量计算"两种写法的一致性，都已经用 NumPy 数值验证过；同时验证了线性注意力的输出确实不等于 softmax attention 的输出，两者是不同的相似度核，不是同一个函数的两种实现。

</details>

### Exercise 6 · KV Cache Attention

自回归生成时，第 `t` 步只需要新 token 的 Q，但要和 `0..t` 所有位置的 K/V 做 attention。如果每步都重新计算全部历史的 K/V，总代价是 `O(T^2)`；把已经算过的 K/V 缓存下来，每步只新增一个位置的 K/V，总代价降到 `O(T)`。这也是推理服务分成 prefill（一次性处理整段 prompt，把 K/V 填满 cache）和 decode（每步只处理一个新 token，cache 长度加一)两个阶段的原因。

有两个细节容易漏掉。第一，如果模型用 RoPE，缓存的必须是"已经在各自绝对位置上旋转过"的 K，而不是旋转前的原始 K。RoPE 的相对位置性质来自于对每个 K 按其绝对位置单独旋转，一旦缓存了没转的 K、之后再统一转，位置信息就全错了。第二，decode 阶段给新 token 算位置时必须从 `cache_len` 开始，而不是从 0，否则新 token 会被当成第 0 个位置，和已经缓存的位置重叠。

#### Quick Coding：`KVCacheAttention`

```python
class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        ...

    def forward(self, x, start_pos, use_cache=True):
        ...

    def reset_cache(self):
        ...
```

<details>
<summary>参考答案</summary>

```python
class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope
        self.cache_k = None  # (B, H, cache_len, Dh)
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, start_pos, use_cache=True):
        B, T, D = x.shape  # prefill: T = prompt_len；decode: T = 1
        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.num_heads)

        if self.rope is not None:
            positions = torch.arange(start_pos, start_pos + T, device=x.device)
            positions = positions[None, None, :].expand(B, 1, T)
            q = self.rope(q, positions)
            k = self.rope(k, positions)  # 必须在缓存前完成旋转

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
            k, v = self.cache_k, self.cache_v

        # 新 query 只需要看 0..(start_pos+T-1) 的所有 key，天然满足因果约束，
        # decode 阶段 T=1 时甚至不需要显式 mask。
        Tk = k.shape[2]
        if T > 1:
            q_idx = torch.arange(start_pos, start_pos + T, device=x.device)[:, None]
            k_idx = torch.arange(Tk, device=x.device)[None, :]
            mask = (k_idx <= q_idx)[None, None, :, :]
        else:
            mask = None

        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.o_proj(out)
```

一次典型调用：`forward(prompt_embeds, start_pos=0)` 做 prefill，之后每步 `forward(next_token_embed, start_pos=cache_len)` 做 decode，`cache_len` 每步自增 1。

</details>

### Exercise 7 · Flash Attention（分块 + online softmax）

Flash attention 要解决的不是“attention 算得对不对”，而是“算的时候要不要把整个 $(T, T)$ 分数矩阵摆在显存里”。标准实现要先算出完整 `scores`，再整体做 softmax；FlashAttention 把 $K, V$ 按 `BLOCK_N` 分块，依次和当前 $Q$ 块做局部注意力，同时在片上寄存器维护流式更新的 running max 和 running sum，将先前累加量按新旧最大值差的指数项 $\exp(m_{\text{old}} - m_{\text{new}})$ 重新缩放，全程只需 $\mathcal{O}(T)$ 的中间状态，而非 $\mathcal{O}(T^2)$。数学上和一次性算完的结果完全一致，不是近似。

本练习分为两个递进层次：
1. **Part A · PyTorch 在线 Softmax 算法逻辑模拟（CPU/原型验证）**：跑通纯 Python 分块与重缩放逻辑；
2. **Part B · Triton GPU 算子级实现（片上 SRAM Tiling + Tensor Core + 因果剪枝）**：工程生产级 GPU Kernel 编程实战。

---

#### Part A · PyTorch 算法级实现：`flash_attention`

##### Quick Coding：`flash_attention`

```python
def flash_attention(Q, K, V, block_size, causal=False):
    ...
```

<details>
<summary>参考答案（PyTorch 算法实现）</summary>

```python
def flash_attention(Q, K, V, block_size, causal=False):
    *lead, T, d = Q.shape
    Tk = K.shape[-2]
    dv = V.shape[-1]

    out = torch.zeros(*lead, T, dv, device=Q.device, dtype=Q.dtype)
    running_max = torch.full((*lead, T, 1), float("-inf"), device=Q.device)
    running_sum = torch.zeros((*lead, T, 1), device=Q.device)

    for start in range(0, Tk, block_size):
        end = min(start + block_size, Tk)
        Kb, Vb = K[..., start:end, :], V[..., start:end, :]
        scores = torch.einsum("...qd,...kd->...qk", Q, Kb) / math.sqrt(d)

        if causal:
            q_idx = torch.arange(T, device=Q.device)[:, None]
            k_idx = torch.arange(start, end, device=Q.device)[None, :]
            scores = scores.masked_fill(q_idx < k_idx, float("-inf"))

        block_max = scores.amax(dim=-1, keepdim=True)
        block_max = torch.where(torch.isneginf(block_max), running_max, block_max)
        new_max = torch.maximum(running_max, block_max)

        # 之前累积的结果要按 exp(旧 max - 新 max) 重新缩放
        alpha = torch.exp(torch.where(torch.isneginf(running_max),
                                       torch.full_like(running_max, float("-inf")),
                                       running_max - new_max))
        alpha = torch.nan_to_num(alpha, nan=0.0, neginf=0.0)
        out = out * alpha
        running_sum = running_sum * alpha

        p = torch.exp(scores - new_max)
        p = torch.nan_to_num(p, nan=0.0)  # 整行被 mask 掉时 scores 和 new_max 都是 -inf
        out = out + torch.einsum("...qk,...kd->...qd", p, Vb)
        running_sum = running_sum + p.sum(dim=-1, keepdim=True)
        running_max = new_max

    return out / running_sum
```

这份实现和一次性算完整 `(T, T)` 矩阵再 softmax 的朴素版本，在因果和非因果两种设置下都用 NumPy 数值验证过 `allclose`，包括某一整行在某个分块里被完全 mask 掉（`block_max = -inf`）这种边界情况。

</details>

---

#### Part B · Triton GPU 算子级实战：`flash_attn_triton`

##### 算子工程设计与硬件映射
1. **Grid 划分与 CTA 映射**：
   - 启动网格为 2D：`grid = (triton.cdiv(N_CTX, BLOCK_M), Batch * NumHeads)`；
   - 每个 Program 实例（CTA / Thread Block）独占处理某一个 Batch、某一个 Head 下的一个 Query Tile（大小为 `BLOCK_M = 64`）；
2. **SRAM 片上数据流**：
   - **外层**：将当前 Query 块（`[BLOCK_M, HEAD_DIM]`）一次性装载至片上 SRAM 寄存器，在整个内层循环中**常驻不变**；
   - **内层**：沿 Key/Value 序列维度按 `BLOCK_N = 64` 流式循环加载；
   - **寄存器累加器**：
     - `m_i`: `[BLOCK_M]` 行最大值；
     - `l_i`: `[BLOCK_M]` 归一化分母；
     - `acc_o`: `[BLOCK_M, HEAD_DIM]` 输出累加值；
3. **因果边界剪枝（Causal Early Termination）**：
   - 若开启因果遮罩，当 Key 块起始位置 `start_n > (start_m + 1) * BLOCK_M` 时，后续所有块均为严格未来时间步，直接截断 `break` 循环，将计算量直接减半；
   - 处于对角线重叠的块，使用 `offs_m[:, None] >= offs_n[None, :]` 进行行级掩码。

##### Quick Coding：`_flash_attn_fwd_kernel`

```python
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    ...

def flash_attention_triton(q, k, v, causal=True, sm_scale=None):
    ...
```

<details>
<summary>参考答案（Triton Kernel 算子级完整实现）</summary>

```python
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, sm_scale,
    L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    # 行与列在当前 Block 内的相对偏移
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)

    # 内存基地址指针
    q_offset = off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_offset = off_z * stride_kz + off_h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_offset = off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # 1. 载入 Query 块至 SRAM 寄存器（外层常驻）
    q = tl.load(Q + q_offset, mask=offs_m[:, None] < N_CTX, other=0.0)

    # 2. 初始化 Online Softmax 寄存器累加器
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 3. 因果注意力的分块循环上界计算
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

    # 4. 内层循环：沿 Key/Value 流式载入分块
    for start_n in range(lo, hi, BLOCK_N):
        curr_offs_n = start_n + offs_n
        
        # 载入 K 块与 V 块到片上 SRAM
        k = tl.load(K + k_offset + start_n * stride_kn, mask=curr_offs_n[None, :] < N_CTX, other=0.0)
        v = tl.load(V + v_offset + start_n * stride_vn, mask=curr_offs_n[:, None] < N_CTX, other=0.0)

        # Tensor Core 点积：Q @ K.T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # 因果下三角屏蔽
        if IS_CAUSAL:
            mask = offs_m[:, None] >= curr_offs_n[None, :]
            qk = tl.where(mask, qk, float("-inf"))

        # 局部分块的最大值与指数和
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # 核心：根据 (旧 max - 新 max) 重新缩放先前的累加和
        alpha = tl.exp(m_i - m_ij)
        acc_o = acc_o * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # 累加当前块的 V 特征贡献
        acc_o += tl.dot(p.to(v.dtype), v)

    # 5. 最终除以归一化分母，写回全局内存 HBM
    acc_o = acc_o / l_i[:, None]
    out_offset = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(Out + out_offset, acc_o.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
    
    # 存储 logsumexp 便于反向传播
    if L is not None:
        l_ptrs = L + off_hz * N_CTX + offs_m
        tl.store(l_ptrs, m_i + tl.log(l_i), mask=offs_m < N_CTX)


def flash_attention_triton(q, k, v, causal=True, sm_scale=None):
    """
    输入张量 shape 均为：(Batch, Heads, SeqLen, HeadDim)
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
        
    Z, H, N_CTX, D = q.shape
    out = torch.empty_like(q)
    L = torch.empty((Z * H, N_CTX), device=q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64

    # 网格维度：(Query 分块数, Batch * Heads)
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    _flash_attn_fwd_kernel[grid](
        q, k, v, sm_scale,
        L, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=D,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )
    return out
```

##### 单元测试与数值对齐验证代码
```python
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        B, H, S, D = 2, 8, 1024, 64
        q = torch.randn((B, H, S, D), device=device, dtype=torch.float16)
        k = torch.randn((B, H, S, D), device=device, dtype=torch.float16)
        v = torch.randn((B, H, S, D), device=device, dtype=torch.float16)

        # 1. 运行 Triton FlashAttention 算子
        out_triton = flash_attention_triton(q, k, v, causal=True)

        # 2. 运行 PyTorch 原生 Eager Attention
        mask = torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D) + mask
        attn = torch.softmax(scores, dim=-1)
        out_ref = torch.matmul(attn, v)

        # 3. 验证精度完全对齐
        assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)
        print("Triton FlashAttention 数值精度验证通过！")
```

</details>

## 本模块易错点

- MHA 和 causal MHA 的区别只在 mask，不在切 head、拼 head 的流程。
- Cross-attention 的 attention 矩阵是 `(T_q, T_kv)`，mask 形状要跟着 K/V 的序列长度走，不能默认方阵。
- GQA 广播 K/V 头必须用 `repeat_interleave`（分组连续），不能用 `repeat`/`tile`（分组交错），两者 shape 一样但语义完全不同。
- 线性注意力是换了一个相似度核的近似，不等价于 softmax attention；flash attention 是纯计算顺序优化，数值上完全等价。
- KV cache 要缓存"旋转后"的 K；decode 阶段的位置编号必须从 `cache_len` 开始，不能从 0 重新数。
- flash attention 的 online softmax 更新里，旧累积量的缩放系数是 `exp(running_max - new_max)`，写反符号或者忘记缩放旧的 `running_sum` 是最常见的两处 bug。
