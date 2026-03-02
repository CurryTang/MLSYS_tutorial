## nano-vllm part 2


## nano-vllm1: 关键配置

| 配置项 | 默认值 | 含义 | 备注 |
|--------|--------|------|------|
| `model` | 必填 | HuggingFace格式的本地模型路径 | 必须是有效目录 |
| `max_num_batched_tokens` | 16384 | 单次forward最多处理的token总数 | 控制GPU显存峰值，必须 ≥ max_model_len |
| `max_num_seqs` | 512 | 同时处理的最大序列数 | CUDA Graph的batch上限 |
| `max_model_len` | 4096 | 单个序列的最大长度（prompt+generation） | 会被 `hf_config.max_position_embeddings` 截断 |
| `gpu_memory_utilization` | 0.9 | KV Cache可用的GPU显存比例 | 留10%余量防OOM |
| `tensor_parallel_size` | 1 | Tensor Parallelism的GPU数量 | 范围 1-8 |
| `enforce_eager` | False | 是否禁用CUDA Graph | True=eager执行（调试用） |
| `hf_config` | None | HuggingFace模型配置 | 自动从model路径加载 |
| `eos` | -1 | End-of-sequence token ID | 用于检测生成结束 |
| `kvcache_block_size` | 256 | 每个KV Cache物理块存储的token数 | 必须是256的倍数 |
| `num_kvcache_blocks` | -1 | KV Cache物理块总数 | -1表示根据显存自动计算 |

关键是要理解max_num_batched_tokens, max_num_seqs, max_model_len这几项。后面会详细介绍，先看一个case

## 两阶段的计算特性

```
┌─────────────────────────────────────────────────────────────────┐
│  Prefill 阶段（处理prompt）                                      │
│  ─────────────────────────────                                  │
│  • 一次性处理整个prompt（比如1024 tokens）                        │
│  • 计算密集型（Compute Bound）                                   │
│  • GPU利用率高，但延迟大                                         │
│  • 受限于 max_num_batched_tokens                                │
├─────────────────────────────────────────────────────────────────┤
│  Decode 阶段（逐token生成）                                      │
│  ─────────────────────────────                                  │
│  • 每次只生成1个token                                            │
│  • 内存密集型（Memory Bound）                                    │
│  • 单序列GPU利用率低，需要batching提升吞吐                        │
│  • 受限于 max_num_seqs                                          │
└─────────────────────────────────────────────────────────────────┘
```

## 三个参数的作用域

| 参数 | 主要约束阶段 | 核心作用 |
|------|-------------|----------|
| `max_model_len` | 两阶段都影响 | 限制单序列KV Cache占用 |
| `max_num_batched_tokens` | **Prefill** | 限制prefill的峰值显存和计算量 |
| `max_num_seqs` | **Decode** | 限制decode的batch size |

## 具体场景分析

### 场景1：Prefill阶段

```
假设有3个新请求同时到达：
  seq1: prompt长度 = 2000 tokens
  seq2: prompt长度 = 3000 tokens  
  seq3: prompt长度 = 5000 tokens

max_num_batched_tokens = 16384
max_model_len = 4096

处理逻辑：
  ✗ seq3 被拒绝（5000 > max_model_len）
  
  第1次forward: seq1 + seq2 = 5000 tokens（< 16384，可以一起prefill）
  
如果是更大的prompt：
  seq4: prompt长度 = 10000 tokens
  seq5: prompt长度 = 8000 tokens
  
  第1次forward: seq4 alone = 10000 tokens
  第2次forward: seq5 alone = 8000 tokens
  （不能合并，因为 10000+8000 > 16384）
```

### 场景2：Decode阶段

```
假设已有400个序列在decode：
  每个序列每次生成1个token
  
max_num_seqs = 512
max_num_batched_tokens = 16384

单次forward处理的tokens = min(400, 512, 16384) = 400 tokens

如果有600个序列：
  第1次forward: 512个序列，生成512 tokens
  第2次forward: 88个序列，生成88 tokens
```

### 场景3：Continuous Batching（Prefill + Decode混合）

```
当前状态：
  - 300个序列在decode（每个生成1 token = 300 tokens）
  - 2个新请求到达（prompt分别1000和2000 tokens）

max_num_batched_tokens = 16384
max_num_seqs = 512

调度决策：
  decode tokens: 300
  prefill tokens: 1000 + 2000 = 3000
  总计: 3300 < 16384 ✓
  序列数: 302 < 512 ✓
  
  → 可以在同一个forward里混合执行！
```

## 参数设计的Trade-off

```
                    ┌─────────────────┐
                    │ max_model_len   │
                    │ 单序列最大长度    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
   ┌─────────────────────┐      ┌─────────────────────┐
   │ max_num_batched_    │      │ max_num_seqs        │
   │ tokens              │      │                     │
   │                     │      │                     │
   │ 调大 → Prefill吞吐↑  │      │ 调大 → Decode吞吐↑   │
   │        显存占用↑     │      │        调度开销↑     │
   │        首token延迟↑  │      │        CUDA Graph大 │
   └─────────────────────┘      └─────────────────────┘
```
