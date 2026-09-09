# LLMTrainLab

本地实现一个能处理 **资源竞争、分布式训练、故障恢复和大规模调度** 的 LLM Training Control Plane。

它不是「部署一个 LLM Pod」。跑完之后，你应该能讲清楚：

> Kubernetes 管的是 Pod 的生命；训练控制面管的是 Worker group 的生命。

配套笔记：本仓库上一级的 System Design 01C（`notes/SystemDesign/SystemDesign01C Kubernetes.md`），GitHub 独立仓库：[CurryTang/LLMTrainLab](https://github.com/CurryTang/LLMTrainLab)。

```text
LLMJob CRD
   ↓
LLMJob Controller
   ↓
Queue / Admission / Gang Scheduler
   ↓
Kubernetes-shaped Pods
   ↓
Fake Training Workers
   ↓
Checkpoint Store + Metrics
```

不需要 GPU，不需要 kind / minikube。所有「节点、Device Plugin、kubelet」都是进程内模拟。想压测更大规模时，用 `llmctl scale` 加 KWOK 风格的虚拟节点；真要接到 Kubernetes API，再看文末 Phase 6。

---

## 0. 你要先建立的心智模型

真实集群里，一次 4 Worker 训练会经过：

```text
提交 LLMJob
  → 队列 / 配额 / 优先级（Kueue 这一层）
  → Gang：4 张 GPU 同时可分配才 admit
  → 创建 Pod，scheduler bind，Device Plugin Allocate
  → RANK / WORLD_SIZE / MASTER_ADDR
  → rendezvous barrier
  → step + heartbeat + checkpoint
  → 有人挂了：停整组 → 重建 → load ckpt
```

原生 Kubernetes **不会**自动做 gang，也 **不会**把「一个 rank 挂了」理解成 RestartAll。Deployment 会单独重启那个 Pod，NCCL communicator 已经死了，其余 Worker 空转占着卡。

所以本项目的核心对象是 `LLMJob`，不是 `Deployment`。

对照工业系统：

| 本项目 | 真实世界 |
|---|---|
| `ClusterEngine.try_place` 必须一次放下全部 Worker | Volcano PodGroup / scheduler coscheduling |
| `Team.guaranteed_gpus` + `burst_gpus` + priority | Kueue ClusterQueue / ResourceFlavor |
| `fault_pod` 后 `Recovering` + load ckpt | Job 失败策略 RestartAll + 共享存储 |
| 节点 `network=rdma` / `gpu_type=H100` | Device Plugin + 拓扑标签 |
| `scale` 出 `kwok-*` 节点 | [KWOK](https://kwok.sigs.k8s.io/) 虚拟节点 |

Kueue 官方定位就是 AI/ML 批任务的队列、配额、优先级和共享。[Kueue 概览](https://kueue.sigs.k8s.io/docs/overview/)。GPU 在 Kubernetes 里通常由 Device Plugin 暴露给 kubelet。[Device Plugin 文档](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)。

---

## 1. 安装（5 分钟）

需要 Python 3.10+。建议用虚拟环境（macOS Homebrew Python 会拒绝往系统里装包）：

```bash
cd project/LLMTrainLab          # 若你 clone 的是独立仓库，则 cd LLMTrainLab
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
llmctl demo walk                # 只打印后面要敲的命令，不改状态
```

状态文件写在当前目录的 `.llmtrainlab/state.json`。命令要在同一工作目录里敲，才能串起来。

自测：

```bash
pytest -q
```

六条测试覆盖 gang、抢占、checkpoint 恢复、H100/A100 隔离、RDMA 硬约束、同 rack 打包。

---

## 2. Phase 1 · 集群画像和 LLMJob

默认集群故意做成异构，对应笔记里的「调度-2」：

```text
node-a: 4 x A100   rack-1  ethernet  nvlink
node-b: 8 x H100   rack-1  rdma      nvlink
node-c: 2 x L40S   rack-2  ethernet  pcie
```

三个租户：

```text
team-a  训练   guaranteed 6  burst 10  优先级 80
team-b  实验   guaranteed 2  burst 4   优先级 20
team-c  推理   guaranteed 4  burst 8   优先级 100
```

Namespace 名字等于 team 名。配额打在 team 上，对象打在 namespace 上，对应笔记「Namespace 是第一刀，不是最后一道墙」。

```bash
llmctl cluster init --config examples/cluster.yaml
llmctl cluster status
```

期望：

```text
tick=0
H100 allocatable=8
A100 allocatable=4
L40S allocatable=2
```

看一份 Job YAML（`examples/jobs/job-a.yaml`）：

```yaml
apiVersion: llmtrainlab.io/v1
kind: LLMJob
metadata:
  name: llama-a
  namespace: team-a
spec:
  team: team-a
  priority: 20          # 训练，相对推理更低
  workers: 6
  gpusPerWorker: 1
  gpuType: H100
  steps: 400
  checkpointEvery: 100
  requireRdma: true
  preferSameRack: true
```

这就是 CRD。Controller 读的是这些字段，不是「帮你 docker run 一个 llama」。

---

## 3. Phase 2 · Fake GPU 和 Gang Admission

难点：4 个 Worker 必须全部获得资源才能启动。3 个 Running、第 4 个 Pending，是分布式训练最贵的调度错误——那 3 张卡既没有 step，又堵住别人。

先验证「不够就一个都不放」：

```bash
python3 - <<'PY'
from llmtrainlab.engine import ClusterEngine
e = ClusterEngine({"nodes":[{"name":"n1","gpu_type":"H100","gpus":3,"network":"rdma"}],
                   "teams":{"team-a":{"namespace":"team-a","guaranteed_gpus":8,"burst_gpus":8,"priority":50}}})
e.submit_job({"metadata":{"name":"g4","namespace":"team-a"},
              "spec":{"team":"team-a","priority":50,"workers":4,"gpusPerWorker":1,"gpuType":"H100","steps":20,"checkpointEvery":10,"requireRdma":True}})
e.step(3)
print(e.jobs["g4"].phase, e.gpu_inventory()["H100"]["used"])
PY
```

期望 `Queued 0`。3 张卡闲着，Job 也不准占 3 张。

回到主路径：

```bash
llmctl job submit examples/jobs/job-a.yaml
llmctl tick --steps 8
llmctl job list
```

`llama-a` 需要 6 张 H100，`node-b` 正好有 8 张 RDMA H100。期望：

```text
name     phase    gpus  type  step
llama-a  Running  6     H100  >0
```

`llmctl job status llama-a` 里每个 worker 都有 `rank` 和同一个 `node`（或同一 rack）。这是 gang + 同 rack 软约束同时生效。

H100 任务不能落到 A100：`gpuType` 是 Filter，不是 Score。RDMA 任务不能落到 `network=ethernet` 的 node-a / node-c。这两条在 `tests/test_topology.py`。

---

## 4. Phase 3 · 训练 Worker 和 Checkpoint

每个 Worker 模拟：

```text
rank / world_size / master（名字约定 job-worker-0）
→ Bound → Running → 全员 Running 才进 Training
→ 每 tick 一个 step + heartbeat
→ 每 checkpointEvery 步写一份 checkpoint
```

只有全部 Worker ready，训练才真正开始。`Starting` 期间占用的 GPU 是启动税。

继续跑，让 A 跨过第一个 checkpoint：

```bash
llmctl tick --steps 120
llmctl job status llama-a
```

`events` 里应出现 `checkpoint step=100 intact=True`。`ckpt` 列变成 `100`。

Checkpoint 是训练的 source of truth。Pod 可重建，没落盘的 step 不能重建。实现上：共享存储挂了会写下 `intact=False`，恢复时只读完整副本。

```bash
llmctl fault ckpt --duration 30
llmctl tick --steps 50
```

如果正好撞上写盘窗口，最新一份可能损坏，恢复会退到更早的完整点。这对应笔记「持久化-3」。

---

## 5. Phase 4 · 队列、优先级、Quota、Preemption

8 张 H100 上：

```text
Job A  6 GPU  低优先级训练     已经 Running
Job B  2 GPU  高优先级推理     还能塞进去
Job C  4 GPU  实验             塞不进去 → Queued
```

```bash
llmctl job submit examples/jobs/job-b.yaml
llmctl tick --steps 4
llmctl job list

llmctl job submit examples/jobs/job-c.yaml
llmctl tick --steps 3
llmctl job list
```

期望：A Running，B Running，C **Queued**。H100 used=8。这就是「优先调度」：B 优先级高，剩余 2 张立刻给它，而不是让 C 插队。

C 的 priority 写成 `10`，低于 A 的 `20`，所以它不会去抢 A，只会排队。这才能看到「剩余 2 张不够 4 张 → Queued」。

若把 C 的 priority 改成 `80`，它会 **整组抢占 A**（空出 6 张，B 仍占 2，C 拿走 4 张）。把数字改掉再 `cluster init` 一次，就能对比「排队」和「Job 级抢占」两条语义。优先级数字是调度语义，不是装饰。

硬抢占演示（推理要 8 张，必须拆掉别人）：

```bash
llmctl job submit examples/jobs/preempt-high.yaml
llmctl tick --steps 5
llmctl job list
```

`llama-urgent` priority=99，需要 8 张。低优先级整组被停，GPU 一次性腾给它。被抢的 Job 回到 `Preempted`，保留 checkpoint，之后还会再入队。

Quota：team-b 的 `burst_gpus=4`。它不能同时跑两个 4 GPU 的实验。超过 burst 的 Job 会一直 Queued，即使集群空着——空着也不代表这个租户还能借。这和 Kueue 的「quota 先于 kube-scheduler」是同一层。

公平：`_fair_key` 用 `当前占用 / guaranteed` 做第三关键字。长期超用的队会排到同优先级的后面。

FIFO：同优先级、同公平份额，按 `arrival_tick`。

Namespace：`llmctl job list` 的 `ns` 列。隔离的是对象和默认配额边界，不是 RDMA 网的安全沙箱。

---

## 6. Phase 5 · 拓扑和故障注入

### 6.1 杀掉一个 Worker

训练到有 checkpoint 之后：

```bash
llmctl fault pod llama-a-worker-2
llmctl job status llama-a
llmctl tick --steps 6
llmctl job status llama-a
```

期望路径：

```text
检测失败
  → 停止剩余 Worker（拆 communicator）
  → 重新创建整个 Worker group
  → 加载最近完整 checkpoint
  → 继续训练
```

看这些字段：

| 字段 | 含义 |
|---|---|
| `retries` | 整组重启次数 |
| `lost_steps` | 上次完整 ckpt 之后丢掉的 step |
| `recovery_p50/p95` | 检测 + 再调度的 tick 数 |
| worker `step` | 恢复后应从 ckpt 附近继续 |

`lost_steps` 应 ≤ `checkpointEvery`。如果你看到丢掉几百 step，说明恢复读到了损坏文件或根本没写 ckpt。

### 6.2 节点宕机

```bash
llmctl fault node node-b
llmctl tick --steps 4
llmctl cluster status
```

`node-b` 变成 Faulted，上面的 GPU `available=False`，所有落在它上面的 Worker 失败，对应 Job 进 Recovering。若集群里没有别的 H100，Job 会停在 Recovering/Queued，直到：

```bash
llmctl recover-node node-b
llmctl tick --steps 4
```

或用下一节的虚拟节点扩容。

### 6.3 网络延迟

```bash
llmctl fault network --job llama-a
llmctl tick --steps 20
llmctl metrics
```

跨 rack / 非 RDMA 不会让 Job Failed，但每个 step 要额外 tick。`training_throughput` 会掉。这对应笔记「调度成功 ≠ 训练快」。

### 6.4 GPU 突然变少

```bash
llmctl fault gpu node-b --remaining 2
```

空闲卡被摘掉。正在跑的 Job 不会被这个命令直接杀掉（它们已经 Allocate 了）；下一次 admit 会发现可分配 slot 变少。真实世界对应：MIG 重切、节点 drain、硬件报修。

---

## 7. Phase 6 · 大规模虚拟节点和可观测性

```bash
llmctl scale --nodes 40          # 40 个 kwok-* 节点，每节点 8 x H100
llmctl tick --steps 10
llmctl metrics
```

排队的 C、被抢占的 A 会在新容量上被 admit。这是 KWOK 的本地等价物：不启动 kubelet，只增加可调度 slot，用来看队列等待和 admit 延迟的分位数。

```bash
llmctl metrics --json
```

字段和面试该画的图一一对应：

```text
gpu_slot_util              GPU slot 使用率
queue_wait_p50 / p95       队列等待
admit_latency_p50 / p95    Admission latency
training_throughput        假 Worker 的 step/tick
checkpoint_age             距上一份完整 ckpt 的 tick
recovery_p50 / p95         故障恢复时间
controller_reconcile_latency  固定为 1 tick
pending_pods               还没绑上去的 Worker 数
tenant_gpus                每个租户占用
```

真 Kubernetes 上用 [KWOK](https://kwok.sigs.k8s.io/) 可以模拟五百个节点的 Pod 生命周期，再刮 Prometheus。本项目把同一套指标先在笔记本上算出来。若你已经有 kind 集群，可以把 `scale --nodes 500` 的数字当成「我接下来要用 KWOK 灌进去的容量」。

---

## 8. 一条命令跑完最有说服力的 Demo

```bash
llmctl cluster init --config examples/cluster.yaml   # 可省略，demo 会自己建
llmctl demo canonical --preempt --virtual-nodes 40
```

脚本按笔记里的故事走：

1. 提交 A（6 GPU，低优先级）→ 启动
2. 提交 B（2 GPU，高优先级）→ 占满剩余 H100
3. 提交 C（4 GPU）→ 观察排队或抢占
4. 跑过 checkpoint，杀掉 A 的 worker-2，整组恢复
5. `--preempt` 时再提交 8 GPU 高优先级推理，观察 Job 级抢占
6. 加上 KWOK 风格节点
7. 打印 p50/p95

只想看命令清单、自己敲：

```bash
llmctl demo walk
```

---

## 9. 实现顺序（和代码怎么对上）

```text
Phase 1  LLMJob CRD + Controller     engine.Job / submit_job / step
Phase 2  Fake GPU + Gang Admission   try_place：不够就返回 None
Phase 3  Worker + Checkpoint         Bound→Running→Training，checkpoint 列表
Phase 4  队列 / 优先级 / Quota / 抢占  _fair_key + _preempt
Phase 5  拓扑 + 故障注入             gpu_type / rdma filter，llmctl fault
Phase 6  KWOK 压测 + 指标            scale_virtual_nodes + metrics()
```

核心不变量（测试锁住的）：

1. **没有部分分配。** 4 Worker 的 Job 在 3 张卡上 used 必须是 0。
2. **抢占是整组的。** 高优先级要 8 张时，低优先级 6 Worker 一起变成 Preempted，不会留下 2 个孤儿 rank。
3. **恢复读完整 checkpoint。** `lost_steps <= checkpointEvery`。
4. **类型和网络是 Filter。** H100 不会出现在 node-a；`requireRdma` 不会出现在 ethernet 节点。
5. **同 rack 是 Score。** 容量够时 4 个 Worker 不会跨 rack。

---

## 10. 和 Kueue 的差别（面试时主动讲）

| | LLMTrainLab | Kueue |
|---|---|---|
| 作用层 | 教学用、进程内 | 生产、watch Kubernetes API |
| 队列 | 内存里按 priority / fair / FIFO 排序 | ClusterQueue + LocalQueue |
| 配额 | team guaranteed / burst | ResourceFlavor + borrowed quota |
| Gang | `try_place` 原子占坑 | 需 AllOrNothing 或底层 Volcano/coscheduling |
| 抢占 | Job 级，不够就继续抢下一个低优先级 | 支持，策略可配 |
| 训练语义 | 自带 rank / barrier / ckpt / RestartAll | 不管 PyTorch 生命周期，那是 Training Operator 的事 |

可以说：Kueue 解决「这个 workload 现在能不能进集群」；Training Operator / 本项目的 controller 解决「进了之后 Worker group 怎么活、怎么死、怎么恢复」。两者叠在一起才是训练平台。

---

## 11. 故障注入命令一览

```bash
llmctl fault pod llama-a-worker-2
llmctl fault node node-a
llmctl fault network --job llama-a
llmctl fault ckpt --duration 30
llmctl fault gpu node-b --remaining 2
llmctl recover-node node-a
```

模拟：节点宕机、Pod 被驱逐、网络变慢、heartbeat 丢失、checkpoint store 暂时不可用、GPU 容量突然下降。

---

## 12. 读代码的顺序

1. `examples/jobs/*.yaml` — CRD 长什么样
2. `llmtrainlab/engine.py` — `try_place`、`_preempt`、`_detect_failures`、`_advance_workers`
3. `llmtrainlab/cli.py` — 每条 `llmctl` 命令
4. `tests/` — 不变量

不要从 CLI 的 argparse 开始。先看 `try_place` 为什么返回 `None` 而不是 3 张卡。

---

## 13. 你做完应该能回答的问题

1. 为什么训练 Job 不能用 Deployment？
2. request 和 Device Plugin 各管哪一层账本？
3. 部分分配死锁是怎么形成的，gang 在 Filter 之前还是之后？
4. Kueue admit 成功之后，kube-scheduler 仍可能把 Pod 拆开放到不同 rack。缺哪一层？
5. 一个 rank 的 livenessProbe 失败，kubelet 重启它，为什么通常是错的？
6. RPO/RTO 分别由 checkpoint 间隔和「检测 + 排队 + 调度 + 拉镜像 + load」决定。Kubernetes 「自动重启 Pod」覆盖了其中哪一段？
7. 跨 rack 的 Job 为什么可能「调度成功、业务失败」？

答得上来，这个项目就没有白做。笔记正文在 `notes/SystemDesign/SystemDesign01C Kubernetes.md`。
