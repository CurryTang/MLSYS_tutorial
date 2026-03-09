## More GPU

### Overview

![[assets/Pasted image 20251222135239.png]]
**Figure 1: Abstract diagram of the overall architecture of NVIDIA H100/B100 GPU. ** Shows the hierarchical memory and computing structure of the GPU: multiple streaming multiprocessors (SM 0, SM 1, ... SM N-1) are arranged in parallel. Each SM contains 4 Tensor Cores (responsible for matrix multiplication operations, contributing the main computing power, similar to the MXU of TPU) and 4 Warp Schedulers (SIMD vector units, containing 32 lanes, namely "CUDA Core", all lanes in the same warp must perform the same operation). Each SM has 256KB of L1 Cache/SMEM (shared memory, controllable by the programmer, similar to TPU VMEM but smaller). All SMs share 50MB of L2 Cache (automatically managed by hardware, providing faster bandwidth) and underlying HBM high-bandwidth memory (80GB for H100 and 192GB for B100, used to store model parameters, activation values, and optimizer state).

![[assets/Pasted image 20251222135741.png]]
**Figure 2: NVIDIA H100 single SM (streaming multiprocessor) internal detailed architecture diagram. ** Each SM contains 4 processing blocks (Processing Block), shared L1 instruction cache and 256KB L1 data cache/shared memory. Each processing block contains: L0 instruction cache, Warp Scheduler (scheduling 32 threads per cycle), Dispatch Unit, 16384×32-bit register file, and a large number of computing units - 16 INT32 units, 16 FP32 units, 8 FP64 units, 1 fourth-generation Tensor Core, LD/ST (Load/Store) unit and SFU (Special Function Unit). The bottom also features a Tensor Memory Accelerator and a Tex (texture unit). This design enables the H100 to efficiently execute large-scale matrix operations and deep learning workloads in parallel.


### Components

#### Summary of GPU computing components

| Hierarchy | Component | Quantity (H100) | Role | Responsible Action |
| ------------------ | ----------------------------- | ------------- | ------- | -------------------------------- |
| **GPU level** | GigaThread Engine | 1 | Global scheduler | Allocate thread blocks to individual SMs |
| **SM level** | SM (Streaming Multiprocessor) | 132 | Independent computing unit | Execute one or more thread blocks and manage internal resources |
| **SubPartition Level** | Warp Scheduler | 4 per SM | Warp Scheduling | Select eligible warp launch from warp pool |
| | Dispatch Unit | 2 per SubPart | Instruction distribution | Read operands, select execution unit, issue instructions |
| | Scoreboard | 1 per SubPart | Dependency tracking | Track register status, detect data hazards |
| **Execution unit level** | Tensor Core | 4 per SM | Matrix multiplication | GEMM, ~1024 FLOPs/cycle, accounting for 93%+ computing power |
| | FP32 CUDA Cores | 128 per SM | Single precision floating point | ReLU, pointwise ops, reduction |
| | FP64 CUDA Cores | 64 per SM | Double precision floating point | Scientific computing (rarely used in ML) |
| | INT32 Cores | 64 per SM | Integer operations | Address calculation, indexing, bit operations |
| | Load/Store Units | 32 per SM | Memory access | Initiate load/store request, address calculation |
| | SFU (Special Function Unit) | 16 per SM | Special functions | sin, cos, exp, rsqrt and other transcendental functions |
| | Texture Units | 4 per SM | Texture sampling | Used for graphics rendering, occasionally used for interpolation in ML |

#### Calculate component hierarchical relationship

```
GPU
 └── GigaThread Engine (全局调度)
      └── SM ×132
           ├── Warp Pool (最多 64 warps 常驻)
           └── SubPartition ×4
                ├── Warp Scheduler ──► 选 warp
                ├── Dispatch Unit ×2 ──► 发指令
                └── Execution Units
                     ├── Tensor Core (矩阵乘)
                     ├── FP32 Cores ×32 (向量算术)
                     ├── INT32 Cores ×16
                     ├── FP64 Cores ×16
                     ├── LD/ST Units ×8
                     └── SFU ×4
```

#### Summary of GPU storage components

| Hierarchy | Components | Capacity (H100) | Bandwidth | Latency | Scope | Purpose |
|------|------|-------------|------|------|--------|------|
| **Off-chip** | HBM (Video Memory) | 80 GB | 3.35 TB/s | ~400 cycles | Global | Model weights, activations, large tensor |
| | L2 Cache | 50 MB | ~12 TB/s | ~100 cycles | Global | Automatically cache HBM data |
| **SM level** | SMEM (Shared Memory) | 256 KB per SM | ~33 TB/s | ~20 cycles | Shared within Block | Tile data, inter-thread communication |
| | L1 Cache | Shared with SMEM | ~33 TB/s | ~20 cycles | SM Private | Auto-caching (configurable ratio) |
| | TMEM (Tensor Memory) | B200 New | Very High | Very Low | SubPart Private | Feed Tensor Core's dedicated cache |
| **Thread level** | Register File | 64K ×32bit per SM | ~80 TB/s | 1 cycle | Thread private | Local variables, intermediate results |
| | Local Memory | Spill to HBM | Same as HBM | High | Thread private | Register spill (register spill) |
| **Special** | Constant Memory | 64 KB | Broadcast optimization | ~4 cycles (cached) | Read-only global | Constant parameters, hyperparameters |
| | Texture Memory | Shared with L1 | Spatial locality optimization | Medium | Read-only global | 2D spatial data access |

#### Storage level pyramid

```
                    ┌─────────┐
                    │ Register│  64K×32bit/SM, 1 cycle, ~80 TB/s
                    │  File   │  线程私有
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │  SMEM   │  256 KB/SM, ~20 cycles, ~33 TB/s
                    │L1 Cache │  Block 共享 / 自动缓存
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │L2 Cache │  50 MB, ~100 cycles, ~12 TB/s
                    │         │  全局共享，自动管理
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │   HBM   │  80 GB, ~400 cycles, 3.35 TB/s
                    │ (DRAM)  │  全局，持久存储
                    └─────────┘

容量:    小 ◄─────────────────────────────► 大
速度:    快 ◄─────────────────────────────► 慢
```

#### Typical usage scenarios of each storage

| Storage | Typical uses in ML | Programmatically |
|------|----------------|---------|
| **Register** | Accumulator, loop variables, Tensor Core input and output | Automatic allocation, local variables |
| **SMEM** | GEMM tiling, attention K/V cache, reduction intermediate results | `__shared__` explicit declaration |
| **L2** | Data reused across SMs (such as different heads of the same batch) | Automatic, available `cudaAccessPolicyWindow` prompt |
| **HBM** | Weight matrix, input and output tensor, optimizer state | `cudaMalloc`, global array |
| **Constant** | Layer’s hyperparameters, lookup table | `__constant__` declaration |


### Understand the working mechanism of warp and dispatch through pseudo code


```python
# ============ SM 内部结构 ============
class SM:
    def __init__(self):
        # 执行单元（以 Ampere 架构为例，每个 SM 有 4 个 sub-partition）
        self.sub_partitions = [SubPartition() for _ in range(4)]
        
        # 每个 sub-partition 有自己的 warp scheduler + dispatch unit
        
class SubPartition:
    def __init__(self):
        self.warp_scheduler = WarpScheduler()
        self.dispatch_units = [DispatchUnit(), DispatchUnit()]  # 通常 2 个
        
        # 执行单元
        self.int32_units = [INT32_ALU() for _ in range(16)]
        self.fp32_units = [FP32_ALU() for _ in range(16)]
        self.fp64_units = [FP64_ALU() for _ in range(8)]
        self.ld_st_units = [LoadStoreUnit() for _ in range(8)]
        self.sfu_units = [SpecialFuncUnit() for _ in range(4)]  # sin, cos, exp...
        self.tensor_cores = [TensorCore() for _ in range(1)]


# ============ Warp Scheduler ============
class WarpScheduler:
    """决定下一个周期执行哪个 warp"""
    
    def __init__(self):
        self.warp_pool = []  # 该 scheduler 管理的所有 warp（通常 8 个左右）
        
    def select_warps_to_issue(self):
        """每个周期选择可以发射的 warp"""
        
        ready_warps = []
        for warp in self.warp_pool:
            if self.is_warp_eligible(warp):
                ready_warps.append(warp)
        
        # 调度策略：GTO (Greedy Then Oldest), LRR (Loose Round Robin), 等
        selected = self.scheduling_policy(ready_warps)
        return selected  # 可能返回 1-2 个 warp（取决于 dispatch unit 数量）
    
    def is_warp_eligible(self, warp):
        """检查 warp 是否可以被调度"""
        
        if warp.is_finished():
            return False
            
        # 检查 scoreboard：指令的操作数是否就绪
        next_inst = warp.get_next_instruction()
        if not self.scoreboard.operands_ready(warp.id, next_inst):
            return False  # 数据依赖，stall
            
        # 检查结构冒险：目标执行单元是否可用
        if not self.check_structural_hazard(next_inst):
            return False
            
        # 检查是否在等待 barrier 同步
        if warp.waiting_at_barrier:
            return False
            
        return True
    
    def scheduling_policy(self, ready_warps):
        """调度策略示例：GTO - 优先让同一个 warp 连续执行"""
        if not ready_warps:
            return []
        
        # 优先选上次执行的 warp（局部性）
        if self.last_issued_warp in ready_warps:
            return [self.last_issued_warp]
        
        # 否则选最老的 ready warp
        return [min(ready_warps, key=lambda w: w.age)]


# ============ Dispatch Unit ============
class DispatchUnit:
    """把 warp scheduler 选中的指令分发到执行单元"""
    
    def dispatch(self, warp, instruction):
        """将指令分发到具体执行单元"""
        
        # 1. 从 Register File 读取操作数（32 个线程的数据）
        operands = self.read_operands(warp, instruction)
        # operands 是 32 份数据，每个 lane 一份
        
        # 2. 根据指令类型选择执行单元
        exec_unit = self.select_execution_unit(instruction)
        
        # 3. 发射到执行单元
        exec_unit.issue(warp.id, warp.active_mask, instruction, operands)
        
        # 4. 更新 scoreboard：标记目标寄存器为 pending
        self.scoreboard.mark_pending(warp.id, instruction.dest_reg)
        
    def select_execution_unit(self, instruction):
        match instruction.opcode:
            case 'FADD' | 'FMUL' | 'FFMA':
                return self.find_available(self.fp32_units)
            case 'IADD' | 'IMUL' | 'IMAD':
                return self.find_available(self.int32_units)
            case 'LD' | 'ST':
                return self.find_available(self.ld_st_units)
            case 'SIN' | 'COS' | 'EXP' | 'RCP':
                return self.find_available(self.sfu_units)
            case 'HMMA' | 'IMMA':  # Tensor Core ops
                return self.find_available(self.tensor_cores)


# ============ 执行单元 ============
class FP32_ALU:
    """FP32 执行单元 - SIMT 执行"""
    
    def issue(self, warp_id, active_mask, instruction, operands):
        """执行 32 个线程的计算"""
        
        results = [None] * 32
        for lane in range(32):
            if active_mask & (1 << lane):  # 只执行 active 的线程
                a = operands.src1[lane]
                b = operands.src2[lane]
                
                match instruction.opcode:
                    case 'FADD':
                        results[lane] = a + b
                    case 'FMUL':
                        results[lane] = a * b
                    case 'FFMA':
                        c = operands.src3[lane]
                        results[lane] = a * b + c
        
        # 写回 register file（流水线化，可能需要几个周期）
        self.writeback_queue.enqueue(warp_id, instruction.dest_reg, results)


# ============ 完整的每周期流程 ============
def sm_cycle(sub_partition):
    """每个时钟周期的流水线操作"""
    
    # Stage 1: Warp Scheduler 选择 warp
    selected_warps = sub_partition.warp_scheduler.select_warps_to_issue()
    
    # Stage 2: Dispatch Unit 分发指令
    for i, warp in enumerate(selected_warps):
        if i < len(sub_partition.dispatch_units):
            instruction = warp.fetch_next_instruction()
            sub_partition.dispatch_units[i].dispatch(warp, instruction)
            warp.pc += 1
    
    # Stage 3-N: 执行单元流水线执行（并行进行）
    for unit in all_execution_units(sub_partition):
        unit.pipeline_tick()
    
    # Writeback: 完成的结果写回 register file，更新 scoreboard
    sub_partition.process_writebacks()
```

## Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│                         SM                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Sub-Partition (×4)                         │ │
│  │                                                         │ │
│  │   ┌──────────────┐                                      │ │
│  │   │ Warp Pool    │  (8 warps)                          │ │
│  │   │ W0 W1 W2 ... │                                      │ │
│  │   └──────┬───────┘                                      │ │
│  │          │ 哪个 warp ready?                              │ │
│  │          ▼                                              │ │
│  │   ┌──────────────┐                                      │ │
│  │   │Warp Scheduler│ ──选择 1-2 个 eligible warp          │ │
│  │   └──────┬───────┘                                      │ │
│  │          │                                              │ │
│  │          ▼                                              │ │
│  │   ┌──────────────┐    ┌──────────────┐                  │ │
│  │   │Dispatch Unit │    │Dispatch Unit │  (×2)            │ │
│  │   └──────┬───────┘    └──────┬───────┘                  │ │
│  │          │                   │                          │ │
│  │          ▼                   ▼                          │ │
│  │   ┌─────────────────────────────────────────────┐       │ │
│  │   │           Execution Units                    │       │ │
│  │   │  INT32  FP32  FP64  LD/ST  SFU  TensorCore  │       │ │
│  │   └─────────────────────────────────────────────┘       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

| Components | Responsibilities | Analogy |
| ------------------ | ---------------------- | -------------- |
| **Warp Scheduler** | Decide "who will execute", check dependencies, take risks, choose strategies | Scheduler: choose the next player to play |
| **Dispatch Unit** | Decide "how to execute", read operands, select execution unit, launch | Distributor: send players to the correct track |

Cycle 1: Warp_A: LD r1, [addr] # Initiate memory read, need to wait ~400 cycles
Cycle 2: Warp_B: ADD r2, r3, r4 # Switch to B
Cycle 3: Warp_C: MUL r5, r6, r7 # Switch to C
...
Cycle 400: Warp_A: (Memory return) # The data of A has arrived
Cycle 401: Warp_A: ADD r8, r1, r9 # A continues execution

### Importance of TensorCore
H100 total computing power distribution:
┌───────────────────────────────────────────┐
│ Tensor Core: 990 TFLOPs (bf16) 93.7%
│ CUDA Cores: 66 TFLOPs (fp32) █ 6.3%
└───────────────────────────────────────────┘
Conclusion: In modern ML workloads, Tensor Core is the main force, and CUDA Cores are only responsible for chores such as ReLU and reduction.


Ref:
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.
