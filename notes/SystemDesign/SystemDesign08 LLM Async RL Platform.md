# System Design 08 · 异步 LLM RL 平台

课程位置：[[SystemDesign07 Photo Sharing Feed|07 图片分享与 Feed]] → 本篇 → [[SystemDesign10 Flash Sale|10 秒杀]]

后训练 (Post-Training) 平台，核心在于通过异步拆分调度隔离环境交互与参数更新，并处理硬件故障。

## 1. Functional requirements

1. 创建训练任务 (包含 base model, LoRA 配置, optimizer, budget)。
2. 在指定的 policy version 下采样 rollout 数据并提交训练数据。
3. 状态查询与 Checkpoint 的 save/restore。

```text
Out of scope: Sandbox / reward model 仅作为 rollout 的环境约束存在。
底层沙箱的具体实现细节不在本系统设计范围内。
Rewards details are handled internally.
Not exposed to user APIs directly.
```

## 2. Non-functional requirements

| 目标 | 指标与约束 |
|---|---|
| Control API | Durable accept p99 < 300 ms (accepted ≠ done) |
| Recovery | Restore 15 min / RPO 15 min |
| Async RL | max_policy_lag <= 2 optimizer steps |

## 3. Workflow, schema, and QPS

```text
Client -> Control API -> Initialize Run, create Operation -> Accepted
Worker -> Pull policy v_k -> Rollout in Sandbox -> Submit Datum -> Queue
Trainer -> Dequeue -> Forward/Backward -> Optimizer Step -> v_k+1 -> Checkpoint
```

| Entity | Primary Keys / Schema |
|---|---|
| Run | run_id, base_model, config, status |
| Operation | op_id, run_id, type, idempotent_key, state |
| PolicyVersion | run_id, version_num, checkpoint_path |
| Checkpoint | run_id, version_num, s3_path |
| RolloutBatch | batch_id, run_id, version_num, data_uri |

| 维度 | 估算值 |
|---|---|
| Sample Admission QPS, Output Token, Sandbox | ~60 /s, ~1M tok/s, ~10K |

## 4. Architecture

```text
Control Plane: API Gateway, Run Manager, Metadata DB (manage operations and run state)
Data Plane: Workers, Trainers, Async Queues, Object Storage
```

- Hosted post-training APIs look similar to this pattern.
- 控制面仅负责持久化状态和队列分配。
- 数据面处理巨量的 token 交互。

## 5. Deep dives

### 1. Sync trainer vs async split
A. Sync trainer
+ 严格遵守 on-policy，每一步基于最新的策略产生 rollout，数学上最严谨。
- Rollout 阶段会导致大规模 GPU 闲置等待，严重浪费算力，极大降低吞吐。
B. Async split
+ Rollout 与 Trainer 解耦独立，吃掉大量 GPU 空闲气泡，最大化利用资源。
- 引入了 stale policy 带来的滞后更新成本。
Prefer B. Change in the picture: Trainer 与 Worker 之间通过带缓冲的异步队列解耦，不再要求强制全局屏障。

### 2. Unbounded queue vs hard max_policy_lag
A. Unbounded queue
+ 彻底消除反压阻塞，保证 Worker 一直满载运行。
- 滞后无限放大可能导致训练严重偏离，带来大量无意义计算。
B. Hard max_policy_lag bound
+ 根据 `max_policy_lag <= 2` 等硬性条件，队列满后阻断 Worker 继续采样。
- 可能导致短时间内部分 Worker 闲置，必须实现主动的反压机制。
Prefer B. Change in the picture: 在数据送入 Trainer 前增加 Freshness Gate，限制队列容量并在超时后直接丢弃。

### 3. Continue one rank vs RestartAll from checkpoint
A. Continue one rank
+ GPU 故障时只替换故障节点并恢复单节点的运算状态，恢复代价小。
- 在集体通信中容易产生脑裂和状态不一致，排错极其困难。
B. RestartAll from checkpoint
+ operation log is truth；in-flight step not committed 直接作废。整体退出并拉起新集群。
- 重启全集群成本较高，会丢失部分正在进行的计算。
Prefer B. Change in the picture: 故障时直接触发 collective abort (参见 01C)，废弃未提交的数据，从最后的可用 Checkpoint 整体重试。

## 6. End-to-end workflow

```text
1. Client API requests new RL run with config and budget.
2. Server durable accepts request in < 300ms, returning op_id (status: ACCEPTED).
3. System provisions GPU worker group and initializes base checkpoint.
4. Worker pulls v_k policy, performs rollout in sandbox, submits to bounded queue.
5. Trainer dequeues fresh data, performs forward/backward passes.
6. Accumulates batch, steps optimizer to v_k+1, writes checkpoint.
7. Worker crashes during forward pass, collective aborts.
8. Control plane detects failure; in-flight step is discarded (not committed).
9. Run manager triggers RestartAll, allocating new workers.
10. Cluster restores from last durable checkpoint (within 15 min RPO).
11. Resumes training with original operation log truth.
```

一手资料：AReaL 论文 (A Large-Scale Asynchronous Reinforcement Learning System)。
