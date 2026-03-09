#RooflineAnalysis

> Core question: Why does an algorithm run for 50ms instead of 5ms or 500ms? Where is the bottleneck?
## 1. Motivation

In deep learning, we often encounter such confusion:
- Increasing batch size sometimes improves speed, sometimes it doesn't
- The performance of the same model on different hardware varies greatly
- Certain operators (such as attention) are extremely slow, but matrix multiplication is fast
**Roofline analysis** provides a concise framework to answer these questions: it tells you whether the current bottleneck is **computing power** or **bandwidth**, and how to optimize it.
## 2. Core definition

Any calculation can be broken down into two parts of time:
$$T_{\text{math}} = \frac{\text{FLOPs}}{\text{Accelerator FLOPs/s}}$$
$$T_{\text{comms}} = \frac{\text{Bytes}}{\text{Bandwidth (Bytes/s)}}$$

| Symbol | Meaning | Example (TPU v5e) |
| --------- | ------ | ---------------------------- |
| FLOPs/s | Chip peak computing power | $1.97 \times 10^{14}$ (bf16) |
| Bandwidth | HBM Bandwidth | $8.2 \times 10^{11}$ bytes/s |

### Arithmetic Intensity
$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes}}$$
This is the core concept of roofline analysis: **How ​​many floating point operations can be performed for each byte of data transferred**.
### Critical Intensity
$$\text{Critical Intensity} = \frac{\text{Peak FLOPs/s}}{\text{Peak Bandwidth}}$$
For TPU v5e MXU:
$$\frac{1.97 \times 10^{14}}{8.2 \times 10^{11}} \approx 240 \text{ FLOPs/byte}$$
### 2.4 Compute-bound vs Memory-bound

| Condition | Status | Meaning |
| --------------------------------------------------------------- | -------------------------- | ------------ |
| $\text{Intensity}_{\text{algo}} > \text{Intensity}_{\text{hw}}$ | **Compute-bound** | Computing power is fully utilized ✓ |
| $\text{Intensity}_{\text{algo}} < \text{Intensity}_{\text{hw}}$ | **Memory/Bandwidth-bound** | Computing power is wasted waiting for data ✗ |

## 3. Analysis methods and examples

### 3.1 Systematic approach to Roofline analysis

Performing a roofline analysis requires following these steps:
**Step 1: Determine hardware parameters**
First you need to check the specifications of the target hardware:

| Hardware | HBM capacity | HBM bandwidth $\beta$ | bf16 computing power $\pi$ | int8 computing power | Critical strength $I_c = \pi/\beta$ |
| ------- | ------ | ------------------------ | --------------------- | --------------------- | ---------------------- |
| TPU v5e | 16 GB | $8.1 \times 10^{11}$ B/s | $1.97 \times 10^{14}$ | $3.94 \times 10^{14}$ | 243 |
| TPU v5p | 96 GB | $2.8 \times 10^{12}$ B/s | $4.59 \times 10^{14}$ | $9.18 \times 10^{14}$ | 164 |
| TPU v6e | 32 GB | $1.6 \times 10^{12}$ B/s | $9.20 \times 10^{14}$ | $1.84 \times 10^{15}$ | 575 |

**Step 2: Calculate the FLOPs and Bytes of the algorithm**

For a given algorithm, calculate respectively:
- $W$: total calculation amount (FLOPs)
- $Q$: Total data transfer volume (Bytes) = read + write back

**Step 3: Calculate arithmetic strength**
$$I_{\text{algo}} = \frac{W}{Q} \quad \text{(FLOPs/Byte)}$$
**Step 4: Determine the bottleneck type**

Compare $I_{\text{algo}}$ with $I_c$:
$$T_{\text{actual}} = \max\left( \underbrace{\frac{W}{\pi}}_{T_{\text{compute}}}, \underbrace{\frac{Q}{\beta}}_{T_{\text{memory}}} \right)$$
- If $I_{\text{algo}} > I_c$: **Compute-bound**, $T_{\text{actual}} = T_{\text{compute}}$
- If $I_{\text{algo}} < I_c$: **Memory-bound**, $T_{\text{actual}} = T_{\text{memory}}$

**Step 5: Calculate hardware utilization**
$$\text{Efficiency} = \frac{\text{Achieved FLOPs/s}}{\pi} = \frac{W / T_{\text{actual}}}{\pi}$$
### 3.2 Understanding Roofline Diagram
**Physical meaning of the two areas**:
- **Slash Area (Memory-bound)**: Data transfer is the bottleneck, and the computing unit is "waiting for data"
- Actual throughput = $I_{\text{algo}} \times \beta$ (increases linearly with intensity)
- **Platform area (Compute-bound)**: Computing is the bottleneck and peak computing power has been reached
- Actual throughput = $\pi$ (no more growth)

![[assets/Pasted image 20251216210603.png]]
> *Shows two algorithms with different computational intensity (Algorithm 1 and Algorithm 2) and their theoretical peak throughput under different bandwidths (BW1 and BW2). The red area indicates that the algorithm is bandwidth limited at both bandwidths, wasting a portion of the hardware's peak FLOPs/s. The yellow area indicates that the algorithm is bandwidth limited only at the lower bandwidth (BW1). The green area indicates that the algorithm is computationally limited at all bandwidths. Here, we are already fully utilizing the accelerator's peak FLOPs/s, and there is no benefit from increasing bandwidth or increasing compute intensity. *

### 3.3 Example: Dot Product (vector dot product)

**Question**: Calculate `x · y`, where `x, y ∈ bf16[N]`, output `bf16[1]`

| Project | Calculation process | Results |
| --------------- | ---------------------------------------------------- | ------------ |
| Read | `x` requires $2N$ bytes, `y` requires $2N$ bytes (bf16 = 2 bytes) | $4N$ |
| write back | output 1 bf16 scalar | $2$ |
| **Total Bytes $Q$** | $4N + 2$ | $\approx 4N$ |
| FLOPs | $N$ multiplications + $(N-1)$ additions | $\approx 2N$ |

$$I_{\text{dot}} = \frac{W}{Q} = \frac{2N}{4N + 2} \xrightarrow{N \to \infty} \frac{1}{2}$$
For TPU v5e, $I_c = 243$, and $I_{\text{dot}} = 0.5 \ll 243$
**Conclusion**: The vector dot product is always memory-bound, no matter how big $N$ is. This is because each element is used only once (no data reuse) and the arithmetic strength is upper bounded.
> [!warning]
> This explains why elementwise operations (e.g. ReLU, LayerNorm) often need to be optimized via **kernel fusion** - almost always memory-bound when executed alone.
### 3.4 Example: Matrix Multiplication⭐

**Question**: Calculate `C = A @ B`, where `A ∈ bf16[M, K]`, `B ∈ bf16[K, N]`, output `C ∈ bf16[M, N]`

| Project | Calculation process | Results |
| --------------- | ------------------------------------- | ----------- |
| Read A | $M \times K$ bf16 | $2MK$ bytes |
| Read B | $K \times N$ bf16 | $2KN$ bytes |
| Write back to C | $M \times N$ bf16 | $2MN$ bytes |
| **Total Bytes $Q$** | $2(MK + KN + MN)$ | |
| FLOPs | Each output element requires $K$ multiplications + $K$ additions, for a total of $MN$ outputs | $2MNK$ |

> [!note]Why 2MNK?
> Matrix multiplication $C_{ij} = \sum_{k=1}^{K} A_{ik} B_{kj}$ does $K$ multiply-adds for each output element.
> One multiply-add = 2 FLOPs, so a total of $2 \times M \times N \times K$ FLOPs.

$$I_{\text{matmul}} = \frac{2MNK}{2(MK + KN + MN)} = \frac{MNK}{MK + KN + MN}$$
**Special Case Analysis** (Suppose $M = B$ (batch), $K = D$ (hidden), $N = F$ (output)):

| Situation | Condition | Approximate Strength | Physical Meaning |
|------|------|----------|----------|
| Batch inference | $B \ll D, F$ | $I \approx B$ | batch size determines whether to compute-bound |
| Square matrix multiplication | $M = K = N$ | $I \approx \frac{N}{3}$ | The larger the dimension, the better |
| GEMV | $N = 1$ | $I \approx 1$ | Vector-matrix multiplication, almost always memory-bound |

For `bf16[B, D] @ bf16[D, F] → bf16[B, F]` (typical FFN layer):
When $B \ll D, F$:
$$I \approx \frac{BDF}{DF} = B$$
On TPU v5e, matmul becomes compute-bound when **Batch size $B > 243$**.

**Common Matrix Multiplication FLOPs Cheat Sheet**:

| Operations | Shape | FLOPs | Description |
|------|-------|-------|------|
| GEMM | `[M,K] @ [K,N]` | $2MNK$ | General matrix multiplication |
| GEMV | `[M,K] @ [K,1]` | $2MK$ | Matrix-vector multiplication, $I \approx 1$ |
| Square matrix multiplication | `[N,N] @ [N,N]` | $2N^3$ | $I \approx N/3$ |
| Batch GEMM | `[B,M,K] @ [B,K,N]` | $2BMNK$ | Batch matrix multiplication |

### 3.5 Example: Complete Roofline Calculation

**Question**: Compute `bf16[256, 4096] @ bf16[4096, 4096]` on TPU v5e and analyze its performance.

**Known parameters**:
- $\pi = 1.97 \times 10^{14}$ FLOPs/s (bf16 peak computing power)
- $\beta = 8.1 \times 10^{11}$ B/s (HBM bandwidth)
- $I_c = \pi / \beta = 243$ FLOPs/Byte

**Step 2: Calculation amount**
- $M = 256, K = 4096, N = 4096$
- $W = 2MNK = 2 \times 256 \times 4096 \times 4096 = 8.59 \times 10^9$ FLOPs
- $Q = 2(MK + KN + MN) = 2(256 \times 4096 + 4096 \times 4096 + 256 \times 4096)$
$= 2(1.05 \times 10^6 + 1.68 \times 10^7 + 1.05 \times 10^6) = 3.77 \times 10^7$ Bytes

**Step 3: Arithmetic Strength**
$$I = \frac{8.59 \times 10^9}{3.77 \times 10^7} = 228 \text{ FLOPs/Byte}$$
**Step 4: Determine the bottleneck**
- $I = 228 < I_c = 243$ → **Memory-bound** (slightly below the critical point)

**Step 5: Calculate time and efficiency**
- $T_{\text{compute}} = W / \pi = 8.59 \times 10^9 / 1.97 \times 10^{14} = 43.6 \mu s$
- $T_{\text{memory}} = Q / \beta = 3.77 \times 10^7 / 8.1 \times 10^{11} = 46.5 \mu s$
- $T_{\text{actual}} = \max(43.6, 46.5) = 46.5 \mu s$
- Efficiency = $43.6 / 46.5 = 93.8\%$

**Conclusion**: Although slightly memory-bound, the efficiency has reached 94%, which is close to optimal.

### 3.6 Example: Int8 quantization Matmul

**Question**: `int8[B, D] @ int8[D, F] → int8[B, F]`

**Change Analysis**:

| project | bf16 version | int8 version | changes |
|------|-----------|-----------|------|
| Data type size | 2 bytes | 1 byte | $\times 0.5$ |
| Total Bytes $Q$ | $2(BD + DF + BF)$ | $BD + DF + BF$ | $\times 0.5$ |
| Peak computing power $\pi$ | $1.97 \times 10^{14}$ | $3.94 \times 10^{14}$ | $\times 2$ |
| FLOPs $W$ | $2BDF$ | $2BDF$ | No change |

**New Arithmetic Strength**:
$$I_{\text{int8}} = \frac{2BDF}{BD + DF + BF}$$
When $B \ll D, F$:
$$I_{\text{int8}} \approx \frac{2BDF}{DF} = 2B$$
**NEW CRITICAL STRENGTH**:
$$I_c^{\text{int8}} = \frac{3.94 \times 10^{14}}{8.1 \times 10^{11}} = 486$$
**Compute-bound conditions**:
$$2B > 486 \implies B > 243$$
**in conclusion**:
- The critical batch size for Int8 is still around 243 (same as bf16!)
- But after reaching compute-bound, **throughput doubles**

> [!tip]
> The main benefit quantified is improved throughput in the compute-bound region, rather than changing the critical point.

### 3.7 Example: Mixed Precision (Int8 Weights + BF16 Activations)

**Question**: `bf16[B, D] @ int8[D, F] → bf16[B, F]`

This scheme is often used for inference optimization: weights are quantized to int8, but activations maintain bf16 accuracy.

**analyze**:

| Projects | Calculations |
|------|------|
| Read activation A | $2BD$ bytes (bf16) |
| Read weight B | $DF$ bytes (int8) |
| Write back output C | $2BF$ bytes (bf16) |
| **Total Bytes $Q$** | $2BD + DF + 2BF$ |
| FLOPs | $2BDF$ (still calculated as bf16) |

**Arithmetic Strength**:
$$I_{\text{mixed}} = \frac{2BDF}{2BD + DF + 2BF}$$
When $B \ll D, F$ and $D \approx F$:
$$I_{\text{mixed}} \approx \frac{2BDF}{DF} = 2B$$
**Compute-bound conditions** (using bf16 computing power $\pi = 1.97 \times 10^{14}$):
$$2B > \frac{1.97 \times 10^{14}}{8.1 \times 10^{11}} = 243 \implies B > 122$$
**Conclusion**: The mixed precision scheme only needs **$B > 122$** to be compute-bound, which is easier to achieve than pure bf16 ($B > 243$)!

### 3.8 Impact of different memory levels

TPU has multiple layers of memory with huge bandwidth differences:

| Memory type | Bandwidth | Relative to HBM | Typical uses |
|----------|------|----------|----------|
| VMEM (SRAM) | ~18 TB/s | 22× | Tile internal computing |
| HBM | ~0.8 TB/s | 1× | Primary Storage |
| ICI (Inter-Chip) | ~0.09 TB/s | 0.1× | Multi-chip communication |
| PCIe | ~0.015 TB/s | 0.02× | Host-Device transmission |

**Example**: `int8[B, 4096] @ int8[16384, 4096]`

| Memory source | Critical Batch Size |
|----------|-----------------|
| HBM | $B > 271$ |
| VMEM | $B > 11$ |

**Conclusion**: If the weights can be fit into VMEM, the critical point is reduced by a factor of 25! This is why **tiling** and **weight caching** are so important.
### 3.9 Batch-Specific weight matrix (negative textbook)

**Question**: If each batch element has a different weight matrix:
`int8[B, D] @ int8[B, D, F] → int8[B, F]`

Find the arithmetic strength.

**analyze**:

This situation occurs in some special scenarios (such as extreme cases of LoRA, per-sample adaptation).

| Project | Standard matmul | Batch-specific weights |
|------|-------------|---------------------|
| Read X | $BD$ | $BD$ |
| Read Y | $DF$ | $\mathbf{BDF}$ |
| Write back Z | $BF$ | $BF$ |
| **Total Bytes** | $BD + DF + BF$ | $BD + BDF + BF$ |
| FLOPs | $2BDF$ | $2BDF$ |

**Arithmetic Strength**:
$$I = \frac{2BDF}{BD + BDF + BF}$$
$BDF$ dominates the denominator (because $D, F$ are usually large):
$$I \approx \frac{2BDF}{BDF} = 2$$
**in conclusion**:
$$\boxed{I \approx 2 \text{ (常数)}}$$
> [!warning]This is a negative teaching material!
>
> Constant arithmetic strength (approximately 2) means:
> - **Always memory-bound**, no matter how big the batch size is
> - Each weight element is used only once, no data reuse
> - Hardware utilization is extremely low: $\text{Efficiency} = \frac{2}{486} \approx 0.4\%$
>
> **Way to avoid this pattern**:
> - try to share weights (standard matmul)
> - If different weights must be used, consider grouping/chunking reuse
> - Use low-rank adaptation methods such as LoRA

### 3.10 GPU (H100) Roofline Analysis

**Question**: Using NVIDIA H100 specifications, calculate the critical batch size.

**H100 SXM Specifications**:

| Parameter | Value | Description |
|------|------|------|
| bf16 Tensor Core FLOPs | $1.979 \times 10^{15}$ | **With sparsity** |
| Actual bf16 FLOPs | $\sim 1 \times 10^{15}$ | No sparsity (divided by 2) |
| HBM3 bandwidth | 3.35 TB/s = $3.35 \times 10^{12}$ B/s | |
| HBM Capacity | 80 GB | |

> [!note]About sparsity
> NVIDIA advertises Tensor Core FLOPs including 2:4 structured sparse acceleration.
> In actual use, if the model is not sparsified, the official number needs to be divided by 2.

**Critical Strength**:
$$I_c = \frac{\pi}{\beta} = \frac{1 \times 10^{15}}{3.35 \times 10^{12}} = 298 \text{ FLOPs/byte}$$
**Critical Batch Size** (when $B \ll D, F$):
$$B > I_c \implies \boxed{B > 298}$$
**Comparison with TPU v5e**:

| Hardware | Peak computing power $\pi$ | HBM bandwidth $\beta$ | Critical strength $I_c$ | Critical B |
|------|----------------|------------------|----------------|--------|
| TPU v5e | $1.97 \times 10^{14}$ | $8.1 \times 10^{11}$ | 243 | ~243 |
| H100 SXM | $1.0 \times 10^{15}$ | $3.35 \times 10^{12}$ | 298 | ~298 |
| **ratio** | 5× | 4× | 1.2× | ~1.2× |

> [!important]Key findings
> Although the absolute computing power and bandwidth of H100 far exceed TPU v5e, the **critical batch size is almost the same**!
>
> This is because the "computing power/bandwidth" ratio of the two is close (about 240-300 FLOPs/byte).
> This ratio is determined by the chip architecture and is a common feature of modern AI accelerators.
## 4. Practice: Code and Tools

### 4.1 PyTorch Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

def torch_roofline(B, D, F, device='cuda'):
    x = torch.randn(B, D, dtype=torch.bfloat16, device=device)
    y = torch.randn(D, F, dtype=torch.bfloat16, device=device)
    
    # Warmup
    for _ in range(10):
        _ = x @ y
    torch.cuda.synchronize()
    
    # Profile
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True
    ) as prof:
        for _ in range(100):
            _ = x @ y
        torch.cuda.synchronize()
    
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=10
    ))
    
    # 导出 Chrome trace
    prof.export_chrome_trace("torch_trace.json")

torch_roofline(256, 4096, 4096)
```

### 4.2 NVIDIA Nsight analysis (root permission required)

```bash
# 收集 roofline 数据
ncu --set roofline -o profile ./your_program

# 查看报告
ncu-ui profile.ncu-rep
```

### 4.3 Analyze the roofline of hello world kernel/matmul kernel


![[assets/Pasted image 20251223165208.png]]
All ops are in the same location


![[assets/Pasted image 20251223164911.png]]

This is the roofline curve of matmul. You can see that as the scale increases, it gradually changes from memory bound to compute bound (it will go online here, why? Because this picture is actually wrong, this is a picture of cuda core, but matmul of bf16 will use tensor core!)
### 5 Summary

![[assets/Pasted image 20251223105527.png]]

* Position of point relative to Ridge Point
* Point is to the left of Ridge Point (AI < Ridge Point):
* The algorithm is in Memory-Bound state. The performance bottleneck is memory bandwidth, where the computing units are waiting for data. Theoretical maximum performance = bandwidth × AI. There is no point in increasing computing power at this time because the data supply is not available.
* Optimization direction: reduce memory access (operator fusion, quantization, sparsification) or improve data reuse (change algorithm).
* Point to the right of Ridge Point (AI > Ridge Point):
* The algorithm is in Compute-Bound state. The performance bottleneck is computing power, and there is excess memory bandwidth. Theoretical maximum performance = peak computing power. There is no point in increasing the memory bandwidth at this point because the computation can't keep up.
* Optimization direction: use more efficient computing instructions (Tensor Core), improve parallelism, and reduce instruction dependencies.
* The position of the point relative to the Roofline line
* Points on the line (efficiency > 80%):
* The implementation is close to the hardware limit, and there is almost no room for optimization under current AI. If you still want to improve performance, you must change the algorithm itself to improve AI (such as operator fusion), or change to stronger hardware.
* Point offline (efficiency < 80%): The implementation does not fully utilize the hardware and there is room for optimization. The specific cause needs to be diagnosed.
* If it is in the Memory-Bound area and the efficiency is low, it may be due to: discontinuous memory access (non-coalesced), low cache hit rate, bank conflict, and data alignment issues.
* If it is in the Compute-Bound area and the efficiency is low, it may be due to: insufficient occupancy, register overflow, not using Tensor Core, or instruction dependencies causing pipeline stalls.
* Dot above the line: Theoretically impossible. If the measurement results show that the point is above the roofline, it means the measurement is wrong or the AI ​​calculation is wrong. Common reasons include: not counting the cache effect, resulting in actual memory access being less than the theoretical value, omissions in FLOPs statistics, and inaccurate timing.

Ref:
Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.
