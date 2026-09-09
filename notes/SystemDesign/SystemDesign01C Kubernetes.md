# System Design 01C · Kubernetes 与 LLM 训练控制面

课程位置：[[SystemDesign01B Virtualization Containers|01B 虚拟化与容器]] → 本篇 → [[SystemDesign01D Redis|01D Redis]]

上一篇介绍 container 是进程隔离与打包。本篇介绍如何将这些进程分配到节点、挂载 GPU、处理故障和多租户竞争。

Kubernetes 是一组声明式控制面：定义期望状态，controller 和 scheduler 将集群向该状态推进。LLM 训练在此之上需要一层额外语义：

```text
异构 GPU 分配
多 Worker 同时启动（gang）
多团队排队、配额、抢占
故障后整组从 checkpoint 恢复
网络拓扑与通信感知
```

配套实验在本地实现了一个训练控制面。代码位于 `project/LLMTrainLab/`，或 [GitHub: CurryTang/LLMTrainLab](https://github.com/CurryTang/LLMTrainLab)。

```k8s-hierarchy-visual
```

---

## 1 · 集群组件与对象生命周期

一个集群包含 control plane 和 worker node。

| 组件 | 职责 |
|---|---|
| kube-apiserver | 对外 HTTP API；所有状态读写入口 |
| etcd | 集群状态的 source of truth |
| kube-scheduler | 给未绑定的 Pod 分配节点 |
| kube-controller-manager | 执行 reconcile loop（ReplicaSet、Job、Node 等） |
| kubelet | 节点代理：通过 CRI 运行容器，汇报状态 |
| container runtime | containerd 等，创建 Linux 容器 |
| kube-proxy / CNI | Service 转发与 Pod 网络连通 |
| Device Plugin | 向 kubelet 注册 GPU 等扩展资源 |

控制面存期望与观测状态。数据面是 Pod 内进程及其网卡、GPU、磁盘。Token 吞吐不经过 apiserver。

一次 4 Worker 的训练 Job 时间线：

```text
1. 提交 LLMJob / PyTorchJob
2. Admission：验证配额、优先级、进入队列（Kueue）
3. Gang：4 个 GPU slot 同时可用时 admit
4. 创建 4 个 Pod，scheduler bind 到具体节点
5. kubelet 拉取镜像、挂载 volume、Allocate GPU
6. 容器启动：获取 RANK、WORLD_SIZE、MASTER_ADDR
7. rendezvous / NCCL init barrier
8. 所有 Worker ready，开始执行 step
9. 周期性 checkpoint 写入共享存储
10. Succeeded 退出；或部分故障导致整组停止并从 ckpt 恢复
```

Pod 相位流转：

```text
Pending（未调度或镜像/GPU 未就绪）
  -> Running
      -> Succeeded / Failed
```

CrashLoopBackOff 是 kubelet 重启单个容器。NCCL communicator 崩溃后，单 Pod 重启无效。

```k8s-lifecycle-visual
```

---

## 2 · 对象层级

对象看起来像嵌套，实际靠 label 和 ownerReference 关联。

```text
Namespace
  └── Service
        └── Deployment
              └── ReplicaSet
                    └── Pod
                          └── Container
```

### Container
Container 是镜像、命令和资源限制的组合。同 Pod 内容器共享网络 namespace 和 volume。
训练 Worker 常见形态包含一个主容器执行 `torchrun`，可选配置 exporter 或 debug 的 sidecar，以及下载数据或配置的 init container。

### Pod
Pod 是最小调度与部署单元。同 Pod 保证在同一节点、共享网络和存储，且被一起创建和删除。一个 rank 需要一张 GPU 时，通常配置一个 Pod 运行一个 Worker。Pod 被删除后 IP 改变，持久身份需依赖 controller 重建。

### ReplicaSet
ReplicaSet 保持匹配 selector 的 Pod 数量等于 `spec.replicas`。

### Deployment
Deployment 管理 ReplicaSet，提供滚动发布和回滚。Deployment 适合无状态服务。
训练 Job 避免使用 Deployment。Deployment 单独重启失败 Pod 会破坏 NCCL 通信组。它的扩缩容不提供 gang 语义，且各个 rank 具有独立 identity。

### Service 与 Namespace
Service 为一组 Pod 提供稳定虚拟 IP 和 DNS。
Headless Service (`clusterIP: None`) 绕过负载均衡，直接将 DNS 解析到 Pod IP。训练通常利用 Service 给 rank 0 提供 rendezvous 域名。

Namespace 是虚拟集群边界，分隔资源名称、配额、RBAC 和网络策略。多租户训练依靠 Namespace 划分团队。

---

## 3 · 调度-1

kube-scheduler 的核心工作是绑定 Pending Pod：

```text
watch unscheduled Pod
  -> Filter（硬约束）
  -> Score（软偏好）
  -> Bind（写入节点名）
```

默认调度器逐个处理 Pod。分配时可能出现 3/4 Worker 处于 Running 而 1 个处于 Pending 的部分分配死锁。

### request、limit、QoS

```text
request  调度器记账：节点剩余可分配量
limit    kubelet/cgroup 上限
```

| QoS | 触发条件 |
|---|---|
| Guaranteed | 每个容器 request == limit |
| Burstable | 存在 request 且小于 limit |
| BestEffort | 未配置 request/limit |

GPU 是扩展资源，request 必须等于 limit。训练 Worker 通常是 Guaranteed 级别。

### 亲和与污点

| 机制 | 作用 |
|---|---|
| nodeSelector | 基础标签匹配 |
| nodeAffinity / podAffinity | 节点或 Pod 级别的软/硬位置匹配 |
| taint + toleration | 节点拒绝调度，只有容忍的 Pod 才能分配 |
| topologySpreadConstraints | 按 zone/rack 打散副本 |

GPU 节点使用 taint 阻止普通 CPU 服务占用。拓扑分布方面，训练要求 Worker 尽量位于同 rack 或同 NVLink 域，推理才需要多可用区打散。

### 抢占
PriorityClass 定义 Pod 优先级。原生抢占针对单个 Pod。训练任务需要 Job 级别的抢占，保证抢出整组资源。

---

## 4 · 调度-2

```k8s-gang-visual
```

### Device Plugin

kubelet 通过 Device Plugin 注册并分配 GPU：

```text
plugin ListAndWatch
  -> kubelet 汇报 nvidia.com/gpu
Pod 声明 nvidia.com/gpu
  -> scheduler 按量过滤节点
  -> kubelet Allocate()
  -> 容器挂载 GPU 设备
```

NVIDIA GPU Operator 部署驱动和 exporter。Device Plugin 将 GPU 暴露为整数个 slot。DRA 是一种较新的资源模型，支持更复杂的拓扑描述和设备共享。

### Gang
Gang 要求一组 Worker 同时分配资源。资源不足时全部等待。部分分配会造成死锁并浪费 GPU。
Volcano 使用 PodGroup 实现 gang。Kueue 在准入层提供类似保证。对于部分分配的 Pod，控制器应主动释放。

### 队列
Namespace 提供配额边界。Kueue 决定负载能否进入集群，支持 FIFO、优先级队列、Fair share 和抢占。在线推理扩容可抢占低优先级训练任务。

### 拓扑标签
节点具有 `zone`、`rack`、`network` 等标签。
硬约束防止 H100 任务落在 A100 节点。软约束使 Worker 优先位于同 rack 或同网络设备下。跨 rack 训练增加 all-reduce 延迟。

---

## 5 · 应用管理-1

### Job
Job 确保指定数量的 Pod 成功执行。`parallelism` 不提供 gang。Indexed Job 为 Pod 分配索引，但失败仍是单节点重试。

### LLMJob / PyTorchJob
训练任务常定义为 CRD。这些 Job 的失败策略是 RestartAll。Worker 的生命周期要求在全部 Running 后注入 MASTER_ADDR，通过 barrier 后才开始计算 step。

---

## 6 · 应用管理-2

### StatefulSet 与 DaemonSet
StatefulSet 提供稳定标识，适合 ZooKeeper。训练任务依靠 controller 重建整组，较少使用 StatefulSet。
DaemonSet 保证每节点运行一个 Pod，用于日志和 exporter。

### Operator
CRD Controller 负责解析自定义资源，创建底层 Pod 和 Service，管理故障恢复和 checkpoint，并更新状态 phase。

### 探针

| 探针 | 失败动作 |
|---|---|
| startupProbe | 不重启，不引入流量 |
| readinessProbe | 从 Service 端点移除，不重启 |
| livenessProbe | kubelet 重启容器 |

编译器或 NCCL 初始化可能导致进程暂时失去响应，liveness 探针容易误杀。训练 heartbeat 属于 Job controller 关注点。

---

## 7 · 持久化-1

| Volume | 作用域 | 示例场景 |
|---|---|---|
| emptyDir | 同 Pod 生命周期 | 临时数据、shm（NCCL） |
| hostPath | 节点本地盘 | 本地缓存 |
| configMap | Kubernetes 对象 | 超参 |
| PVC | 持久存储声明 | Checkpoint、数据集 |

NCCL 内存不足崩溃通常由 tmpfs (`emptyDir` 设置为 Memory) 过小引起。

---

## 8 · 持久化-2

```text
Pod -> PVC -> PV -> 存储后端
```

PVC 是对存储的请求，PV 是实际供应。StorageClass 控制动态配置。

| 访问模式 | 特性 | 场景 |
|---|---|---|
| RWO | 单节点读写 | 节点独占盘 |
| ROX | 多节点只读 | 预训练数据集 |
| RWX | 多节点读写 | 共享 Checkpoint |

多节点同时写入 checkpoint 需要 RWX。CSI 插件对接底层的 Lustre、Ceph 或云盘。

---

## 9 · 持久化-3

Worker 被杀流程：

```text
检测故障
  → 停止当前组
  → 重建 Worker 组
  → 加载最后完整 checkpoint
  → 继续训练
```

衡量恢复的指标包括故障检测时间、恢复重建时间、丢失的 step 数及重试次数。
文件写入需要原子操作（写临时文件后 rename）。从指针文件读取恢复点，防止加载损坏文件。

---

## 10 · 网络-1

每个 Pod 具有独立 IP。CNI 负责 IP 分配与连通。CoreDNS 解析服务域名。kube-proxy 或 eBPF 维护 ClusterIP 映射。
NetworkPolicy 控制连通范围。
梯度通信直接使用 Pod IP，避免经过 ClusterIP 负载均衡。

---

## 11 · 网络-2

### Ingress
Ingress 和 Gateway API 处理南北向 HTTP 流量，用于控制台和 API 访问。训练内部 all-reduce 不通过 Ingress。

### 东西向网络
节点内使用 NVLink。跨节点大带宽通信使用 RDMA。控制面和日志使用普通以太网。
配置 RDMA 网络的任务必须通过调度器过滤。通过设备方式将网卡分给 Pod，优于直接使用 `hostNetwork: true`。

---

## 12 · 可观测性

| 观测级别 | 指标对象 |
|---|---|
| 集群 | Pending Pod、节点就绪状态、kubelet 错误 |
| 控制面 | 队列延迟、gang 状态、恢复时间 |
| 训练任务 | step time、loss、NCCL 耗时、checkpoint 频率 |

使用 Prometheus 获取指标，DCGM exporter 监控 GPU。日志可通过 `job / rank / step` 过滤。KWOK 可用于模拟集群规模测试。

---

## 13 · 分层架构

对象模型和调度解决「一个 Pod 放哪」。系统还要切两刀：每层干什么，以及这些层落在哪。两刀不要画成一张图。

```k8s-layered-arch-visual
```

### 职责分层

按允许做什么切。接入 / 计算 / 数据 / 异步就是这条轴：

| 层 | 典型组件 | 允许 | 不允许 |
|---|---|---|---|
| 接入 | LB / Gateway / Ingress | 鉴权、限流、路由、TLS | 写库存、跑训练 step |
| 计算 | Stateless Service / Deployment | 编排、校验、发写、入队 | 进程里放不可丢的状态 |
| 数据 | DB / Redis / PVC / etcd | 事实、可重建副本、集群状态 | 处理用户 HTTP 业务规则 |
| 异步 | MQ / Job / Worker | 削峰、重试、checkpoint | 挡在同步 p99 路径上 |

k8s 自己也是职责分层：

```text
apiserver / etcd        存期望与观测
Kueue / scheduler       准入与放置
kubelet / runtime       在节点上执行
Pod 进程 / NCCL / GPU   数据面。token 不经过 apiserver
```

同层水平扩。跨层用明确接口：HTTP、PVC、CRD。kubelet 挂了不等于 etcd 挂了。

### 拓扑分层

按谁和谁一起死切。物理位置：

```text
Region
  └── AZ / zone
        └── Rack / NVLink 域
              └── Node
                    └── Pod
                          └── GPU / NIC
```

k8s 用 label 表达：`topology.kubernetes.io/zone`、`rack`、`network=rdma`。调度读这些 label，不读机柜图纸。

| 工作负载 | 拓扑怎么放 |
|---|---|
| 无状态 API | topologySpread，跨 AZ。一个 AZ 挂了还有副本 |
| 训练 gang | 软约束聚到同 rack / 同 NVLink；硬约束 GPU 型号和 RDMA |
| 控制面 | 独立节点和以太网；不要和 GPU 数据面抢网 |
| Checkpoint | 另一套 RWX 存储，与 Worker 不同故障域 |

### 两轴一起用

先画职责，再把每一层标到拓扑上：这个 Deployment 几个 AZ，这个 Redis 是否跨 AZ，这 4 个 rank 是否同一 rack。

常见混用：

```text
把 zone 当成职责层          「三层：北京、上海、缓存」
职责对了但拓扑叠在一起      API 和 MySQL 同节点
训练按无状态服务打散        4 个 rank 跨 AZ，all-reduce 被延迟打穿
```

训练一条路径上的两轴：

```text
职责    Kueue 准入 → scheduler 绑定 → kubelet 拉起 → NCCL
拓扑    4 个 rank 同一 rack；checkpoint 在 RWX；apiserver 在控制面节点
```

---

## 14 · 总结与配套实验

Kubernetes 管理 Pod。训练控制面管理 Worker 组。
本地实验代码模拟了以下节点配置：

```text
node-a: 4 x A100 ethernet
node-b: 8 x H100 rdma
node-c: 2 x L40S ethernet
```

执行以下命令：

```bash
cd project/LLMTrainLab
python3 -m pip install -e ".[dev]"
llmctl demo canonical
```

该命令在 8 个 GPU slot 上演示：6 GPU 低优先级 Job A 启动，2 GPU 高优先级 Job B 填充剩余资源，4 GPU Job C 排队等待。杀死 A 的一个 Worker 后触发 RestartAll 从 checkpoint 恢复。实验还包含可选的 8 GPU 抢占和虚拟节点 p50/p95 延迟统计。

---

## 一手资料

- [Kubernetes 组件](https://kubernetes.io/docs/concepts/overview/components/)
- [Pod 生命周期](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [Scheduling Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)
- [Device Plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
- [Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [Kueue 概览](https://kueue.sigs.k8s.io/docs/overview/)
- [KWOK](https://kwok.sigs.k8s.io/)
- [ByteDance: Robust LLM Training Infrastructure](https://www.alphaxiv.org/abs/2509.16293)
- [LLMTrainLab](https://github.com/CurryTang/LLMTrainLab) (本仓库 `project/LLMTrainLab/`)
