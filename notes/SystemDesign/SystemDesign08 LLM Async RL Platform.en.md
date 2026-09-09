# System Design 08 · LLM Async RL Platform

Course location: [[SystemDesign07 Photo Sharing Feed|07 Photo Sharing and Feed]] → this note → [[SystemDesign10 Flash Sale|10 Flash Sale]]

A post-training platform centered on async split scheduling to isolate environment rollouts from parameter updates, while handling hardware failures securely.

## 1. Functional requirements

1. Create training runs (including base model, LoRA config, optimizer, budget).
2. Sample rollout data at a specific policy version and submit to training queues.
3. Query status and perform Checkpoint save/restore.

```text
Out of scope: Sandbox / reward model implementation details.
They are assumed to exist as external constraints.
Rewards details are handled internally.
Not exposed to user APIs directly.
```

## 2. Non-functional requirements

| Goal | Metrics & Constraints |
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

| Parameter | Estimate |
|---|---|
| Sample Admission QPS, Output Token, Sandbox | ~60 /s, ~1M tok/s, ~10K |

## 4. Architecture

```text
Control Plane: API Gateway, Run Manager, Metadata DB (manage operations and run state)
Data Plane: Workers, Trainers, Async Queues, Object Storage
```

- Hosted post-training APIs look similar to this pattern.
- Control plane manages durable state machine.
- Data plane handles massive token interactions asynchronously.

## 5. Deep dives

### 1. Sync trainer vs async split
A. Sync trainer
+ Strictly on-policy. Computes rollouts step-by-step using the latest parameters, guaranteeing mathematical rigor.
- Massive GPU idleness during rollouts wastes compute resources and heavily impacts throughput.
B. Async split
+ Decouples rollout and training. Absorbs compute bubbles, heavily increasing overall GPU utilization.
- Introduces stale policies. Cost is dealing with off-policy data correctly.
Prefer B. Change in the picture: Workers push to a buffered async queue instead of hitting a global barrier, removing tight synchronization.

### 2. Unbounded queue vs hard max_policy_lag
A. Unbounded queue
+ Zero backpressure blocks, allowing workers to run at maximum independent speeds.
- Infinite staleness backlog leads to heavily off-policy data, destroying convergence.
B. Hard max_policy_lag bound
+ Enforces limits like `max_policy_lag <= 2`, actively rejecting inputs if queues pile up.
- Requires explicit backpressure mechanisms and limits idle worker efficiency temporarily.
Prefer B. Change in the picture: A Freshness Gate bounds the depth of the ready buffer, silently dropping or tagging excessively lagged trajectories.

### 3. Continue one rank vs RestartAll from checkpoint
A. Continue one rank
+ Fixes individual broken GPUs by reloading local rank states, minimizing recovery overhead.
- Collective communication split-brain makes debugging and state consistency practically impossible.
B. RestartAll from checkpoint
+ operation log is truth; in-flight step not committed. Whole cluster restarts cleanly.
- Higher cost to re-provision cluster and wastes partial batch progress.
Prefer B. Change in the picture: Treat in-flight failures as collective aborts (see 01C). Trigger group re-provisioning and resume from the latest durable checkpoint.

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

Primary sources: AReaL (A Large-Scale Asynchronous Reinforcement Learning System) paper.
