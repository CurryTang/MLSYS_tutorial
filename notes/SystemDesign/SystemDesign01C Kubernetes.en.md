# System Design 01C · Kubernetes and the LLM Training Control Plane

Course location: [[SystemDesign01B Virtualization Containers|01B Virtualization and Containers]] → this note → [[SystemDesign01D Redis|01D Redis]]

The previous note introduces containers as process isolation and packaging. This note explains how those processes are assigned to nodes, attached to GPUs, restarted after failures, and managed under multi-tenant competition.

Kubernetes operates as a declarative control plane. You define the desired state, and controllers plus the scheduler continuously push the cluster toward it. LLM training requires an additional layer of semantics:

```text
Heterogeneous GPU allocation
Multi-worker simultaneous start (gang)
Multi-team queues, quotas, and preemption
Whole-group recovery from checkpoints
Network topology and communication awareness
```

The companion lab implements a local training control plane. The source code is in `project/LLMTrainLab/` or [GitHub: CurryTang/LLMTrainLab](https://github.com/CurryTang/LLMTrainLab).

```k8s-hierarchy-visual
```

---

## 1 · Cluster Components and Object Lifecycle

A cluster consists of a control plane and worker nodes.

| Component | Role |
|---|---|
| kube-apiserver | External HTTP API; single entry point for all state reads/writes |
| etcd | Source of truth for cluster state |
| kube-scheduler | Binds pending Pods to nodes |
| kube-controller-manager | Executes reconcile loops (ReplicaSet, Job, Node) |
| kubelet | Node agent: runs containers via CRI, reports status |
| container runtime | containerd, etc.; creates Linux containers |
| kube-proxy / CNI | Service forwarding and Pod networking |
| Device Plugin | Registers GPUs and extended resources with kubelet |

The control plane stores desired and observed states. The data plane contains Pod processes, their NICs, GPUs, and disks. Token throughput bypasses the apiserver.

A 4-worker training job timeline:

```text
1. Submit LLMJob / PyTorchJob
2. Admission: Validate quota and priority, enter queue (Kueue)
3. Gang: Admit when 4 GPU slots become available simultaneously
4. Create 4 Pods; scheduler binds them to nodes
5. kubelet pulls images, mounts volumes, and runs Allocate for GPUs
6. Containers start: receive RANK, WORLD_SIZE, MASTER_ADDR
7. rendezvous / NCCL init barrier
8. All workers are ready; step execution begins
9. Periodic checkpoints to shared storage
10. Exit Succeeded; or a failure halts the group and triggers a recovery from ckpt
```

Pod phase progression:

```text
Pending (Unscheduled or image/GPU not ready)
  -> Running
      -> Succeeded / Failed
```

CrashLoopBackOff indicates kubelet restarting a single container. Single-Pod restarts are ineffective after the NCCL communicator crashes.

```k8s-lifecycle-visual
```

---

## 2 · Object Hierarchy

Objects look nested. Association is labels plus ownerReferences.

```text
Namespace
  └── Service
        └── Deployment
              └── ReplicaSet
                    └── Pod
                          └── Container
```

### Container
A container binds an image, a command, and resource limits. Containers in the same Pod share network namespaces and volumes.
A typical training worker contains a main container executing `torchrun`. It may include sidecars for metrics or debugging, and init containers for downloading data.

### Pod
The Pod represents the smallest scheduling and deployable unit. Same-Pod containers share a node, network, and storage, and are created and deleted together. A rank requiring one GPU typically uses one Pod. Pod IPs change upon deletion. Durable identity requires controller reconstruction.

### ReplicaSet
A ReplicaSet maintains the count of Pods matching a selector equal to `spec.replicas`.

### Deployment
A Deployment manages ReplicaSets and handles rolling updates and rollbacks. Deployments fit stateless services.
Do not use Deployments for training Jobs. Deployments restart failed Pods individually, which breaks the NCCL group. Their scaling does not provide gang semantics, and each rank possesses unique identity.

### Service and Namespace
A Service provides a stable virtual IP and DNS name to a group of Pods.
Headless Services (`clusterIP: None`) bypass load balancing and resolve DNS directly to Pod IPs. Training setups use Services to give rank 0 a rendezvous domain name.

A Namespace provides a virtual cluster boundary. It separates resource names, quotas, RBAC, and network policies. Multi-tenant training uses Namespaces to isolate teams.

---

## 3 · Scheduling-1

The core task of kube-scheduler is binding Pending Pods:

```text
watch unscheduled Pod
  -> Filter (hard constraints)
  -> Score (soft preferences)
  -> Bind (write node name)
```

The default scheduler processes Pods individually. This can lead to partial allocations where 3 of 4 workers are Running and 1 remains Pending.

### Requests, Limits, QoS

```text
request  Scheduler ledger: remaining allocatable capacity on node
limit    kubelet/cgroup cap
```

| QoS | Trigger Condition |
|---|---|
| Guaranteed | Every container request == limit |
| Burstable | Request exists and is less than limit |
| BestEffort | No request/limit defined |

GPUs are extended resources. Their requests must equal limits. Training Workers usually run in the Guaranteed class.

### Affinity and Taints

| Mechanism | Role |
|---|---|
| nodeSelector | Basic label matching |
| nodeAffinity / podAffinity | Soft/hard placement matching at node or Pod level |
| taint + toleration | Nodes reject scheduling; only tolerating Pods enter |
| topologySpreadConstraints | Spread replicas across zones/racks |

GPU nodes use taints to block CPU services. Training Workers prefer placement on the same rack or NVLink domain. Spreading across availability zones is an inference practice.

### Preemption
PriorityClass defines Pod priority. Native preemption targets individual Pods. Training tasks require Job-level preemption to secure resources for the entire gang.

---

## 4 · Scheduling-2

```k8s-gang-visual
```

### Device Plugin

kubelet registers and allocates GPUs via Device Plugins:

```text
plugin ListAndWatch
  -> kubelet reports nvidia.com/gpu
Pod requests nvidia.com/gpu
  -> scheduler filters nodes by capacity
  -> kubelet Allocate()
  -> Container mounts GPU devices
```

The NVIDIA GPU Operator deploys drivers and exporters. Device Plugins expose GPUs as integer slots. DRA provides a newer resource model supporting complex topologies and device sharing.

### Gang
Gang scheduling requires a group of Workers to receive resources simultaneously. When resources fall short, the entire group waits. Partial allocations cause deadlocks and waste GPUs.
Volcano implements gang semantics via PodGroup. Kueue offers similar guarantees at the admission layer. Controllers should actively release partially allocated Pods.

### Queues
Namespaces establish quota boundaries. Kueue dictates whether workloads enter the cluster, supporting FIFO, priority queues, fair share, and preemption. Online inference scale-ups can preempt lower-priority training tasks.

### Topology Labels
Nodes carry labels for `zone`, `rack`, and `network`.
Hard constraints prevent H100 tasks from landing on A100 nodes. Soft constraints group Workers under the same rack or network device. Cross-rack placements increase all-reduce latency.

---

## 5 · Application Management-1

### Job
A Job ensures a specified number of Pods complete successfully. `parallelism` lacks gang semantics. Indexed Jobs assign indices to Pods, but failures trigger single-node retries.

### LLMJob / PyTorchJob
Training tasks operate as CRDs. Their failure policy is RestartAll. Worker lifecycles dictate injecting MASTER_ADDR after all Pods are Running. Step computation starts only after passing the barrier.

---

## 6 · Application Management-2

### StatefulSet and DaemonSet
StatefulSets offer stable identities, suiting ZooKeeper. Training relies on controllers to rebuild groups and rarely uses StatefulSets.
DaemonSets run one Pod per node, serving logs and exporters.

### Operator
CRD Controllers parse custom resources, instantiate Pods and Services, govern failure recovery and checkpoints, and update phase status.

### Probes

| Probe | Failure Action |
|---|---|
| startupProbe | No restart, no traffic |
| readinessProbe | Remove from Service endpoint, no restart |
| livenessProbe | kubelet restarts container |

Compiler or NCCL initializations may block processes. Liveness probes can trigger false kills. Training heartbeats are monitored by the Job controller.

---

## 7 · Persistence-1

| Volume | Scope | Example |
|---|---|---|
| emptyDir | Tied to Pod lifecycle | Scratch data, shm (NCCL) |
| hostPath | Node local disk | Local cache |
| configMap | Kubernetes object | Hyperparameters |
| PVC | Persistent claim | Checkpoints, datasets |

Insufficient tmpfs (`emptyDir` set to Memory) causes NCCL out-of-memory crashes.

---

## 8 · Persistence-2

```text
Pod -> PVC -> PV -> Storage Backend
```

PVC requests storage; PV supplies it. StorageClass governs dynamic provisioning.

| Access Mode | Property | Scenario |
|---|---|---|
| RWO | Single-node read/write | Node-exclusive disk |
| ROX | Multi-node read-only | Pretraining datasets |
| RWX | Multi-node read/write | Shared Checkpoints |

Multi-node checkpoint writing requires RWX. CSI plugins interface with backends like Lustre, Ceph, or cloud disks.

---

## 9 · Persistence-3

Worker kill sequence:

```text
Detect failure
  → Stop current group
  → Rebuild Worker group
  → Load last complete checkpoint
  → Resume training
```

Recovery metrics track fault detection time, rebuild duration, lost steps, and retry counts.
File writes demand atomic operations (rename after writing). Loading relies on pointer files to avoid corrupted data.

---

## 10 · Networking-1

Every Pod receives an independent IP. CNI handles IP assignment and connectivity. CoreDNS resolves service domains. kube-proxy or eBPF maintains ClusterIP mappings.
NetworkPolicy restricts connectivity.
Gradient communication uses Pod IPs directly to avoid ClusterIP load balancing.

---

## 11 · Networking-2

### Ingress
Ingress and Gateway API manage north-south HTTP traffic for dashboards and API access. Internal all-reduce operations bypass Ingress.

### East-West Networks
Nodes utilize NVLink internally. High-bandwidth cross-node communication uses RDMA. Control planes and logs rely on standard Ethernet.
The scheduler must filter tasks requiring RDMA networks. Assigning NICs to Pods as devices is preferable to using `hostNetwork: true`.

---

## 12 · Observability

| Level | Target Metrics |
|---|---|
| Cluster | Pending Pods, node readiness, kubelet errors |
| Control Plane | Queue delays, gang status, recovery time |
| Training Task | Step time, loss, NCCL duration, ckpt frequency |

Prometheus collects metrics. DCGM exporter monitors GPUs. Logs are filtered by `job / rank / step`. KWOK simulates large cluster scales.

---

## 13 · Layered architecture

The object model and the scheduler answer where one Pod goes. A system still needs two cuts: what each layer may do, and where those layers sit. Do not draw both cuts as one picture.

```k8s-layered-arch-visual
```

### Responsibility layers

Cut by what a layer is allowed to do. Edge / compute / data / async is this axis:

| Layer | Typical components | Allowed | Not allowed |
|---|---|---|---|
| Edge | LB / Gateway / Ingress | Auth, rate limit, routing, TLS | Stock writes, training steps |
| Compute | Stateless service / Deployment | Orchestrate, validate, issue writes, enqueue | Durable facts inside the process |
| Data | DB / Redis / PVC / etcd | Facts, rebuildable copies, cluster state | User HTTP business rules |
| Async | MQ / Job / Worker | Absorb burst, retry, checkpoint | Sitting on the sync p99 path |

Kubernetes itself is also layered by responsibility:

```text
apiserver / etcd        desired and observed state
Kueue / scheduler       admission and placement
kubelet / runtime       execute on the node
Pod / NCCL / GPU        data plane. tokens do not go through apiserver
```

Scale a layer horizontally. Cross a layer only through an explicit interface: HTTP, PVC, CRD. A dead kubelet is not a dead etcd.

### Topology layers

Cut by what fails together. Physical placement:

```text
Region
  └── AZ / zone
        └── Rack / NVLink domain
              └── Node
                    └── Pod
                          └── GPU / NIC
```

Kubernetes names this with labels: `topology.kubernetes.io/zone`, `rack`, `network=rdma`. The scheduler reads labels, not rack drawings.

| Workload | Placement |
|---|---|
| Stateless API | topologySpread across AZs. One AZ down still leaves replicas |
| Training gang | Soft: same rack / NVLink. Hard: GPU type and RDMA |
| Control plane | Separate nodes and Ethernet. Do not share the GPU data fabric |
| Checkpoint | A different RWX store, a different failure domain from Workers |

### Using both axes

Draw responsibility first, then pin each layer onto topology: how many AZs for this Deployment, whether Redis is cross-AZ, whether these four ranks share a rack.

Common mixes:

```text
Treat a zone as a responsibility layer    "three layers: Beijing, Shanghai, cache"
Right duties, stacked topology            API and MySQL on the same node
Spread a gang like a stateless service    four ranks across AZs; all-reduce dies on latency
```

One training path on both axes:

```text
duty      Kueue admit → scheduler bind → kubelet start → NCCL
topology  four ranks on one rack; checkpoint on RWX; apiserver on control-plane nodes
```

---

## 14 · Summary and Companion Lab

Kubernetes manages Pods. The training control plane manages Worker groups.
The local lab simulates the following node layout:

```text
node-a: 4 x A100 ethernet
node-b: 8 x H100 rdma
node-c: 2 x L40S ethernet
```

Execute this command:

```bash
cd project/LLMTrainLab
python3 -m pip install -e ".[dev]"
llmctl demo canonical
```

This command runs on 8 GPU slots: a 6 GPU low-priority Job A starts, a 2 GPU high-priority Job B fills the remainder, and a 4 GPU Job C queues. Killing one of A's workers triggers a RestartAll recovery from the checkpoint. The lab also features optional 8 GPU preemption and virtual node p50/p95 latency metrics.

---

## Primary Sources

- [Kubernetes components](https://kubernetes.io/docs/concepts/overview/components/)
- [Pod lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [Scheduling Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)
- [Device Plugin](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
- [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [Kueue overview](https://kueue.sigs.k8s.io/docs/overview/)
- [KWOK](https://kwok.sigs.k8s.io/)
- [ByteDance: Robust LLM Training Infrastructure](https://www.alphaxiv.org/abs/2509.16293)
- [LLMTrainLab](https://github.com/CurryTang/LLMTrainLab) (this repo: `project/LLMTrainLab/`)
