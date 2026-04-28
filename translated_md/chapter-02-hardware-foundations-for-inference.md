# 第 2 章：推理的硬件基础

_来源页：PDF 第 9-12 页_

## 2.1 面向推理工程师的 GPU 架构

现代 GPU 采用分层组织。在最高层次上，一块 NVIDIA H100 SXM5 包含：

- 132 个 Streaming Multiprocessors（SMs），每个 SM 包含 128 个 CUDA cores 和 4 个第四代 Tensor Cores。
- 80 GB HBM3 memory，带宽为 ≥3.35 TB/s。
- 每个 SM 256 KB 的 shared memory / L1 cache，速度比 HBM 快数个数量级。
- 所有 SM 共享的 50 MB L2 cache。
- NVLink 4.0，为多 GPU 通信提供 900 GB/s 的双向带宽。

内存层次结构对推理优化至关重要。可以在 shared memory 中执行的操作（如 FlashAttention 中的 fused attention）完全避免了到 HBM 的缓慢往返。

## 2.2 Tensor Cores 与混合精度

Tensor Cores 是专用的矩阵乘法单元，可在低精度格式下对操作数的小 tile（例如 16 × 16 × 16）进行运算。H100 的第四代 Tensor Cores 支持：

| 格式 | 峰值 TFLOP/s | 用例 |
| --- | ---: | --- |
| FP64 | 67 | 科学计算 |
| TF32 | 989 | 训练默认格式 |
| FP16/BF16 | 1,979 | 推理标准格式 |
| FP8 | 3,958 | 量化推理 |
| INT8 | 3,958 | 量化推理 |

关键洞见是：推理可以利用比训练更低的精度，因为我们只需要保持 forward pass fidelity，不需要以高精度进行梯度累积。这就是为什么 INT8 和 FP8 推理能够达到 FP16 理论峰值吞吐量约 2× 的原因。

## 2.3 内存层次结构与带宽

理解数据在任何时刻位于何处，对于推理性能至关重要：

**GPU Memory Hierarchy（H100 SXM）**

| 层级 | 容量 | 带宽 | 延迟 |
| --- | --- | --- | --- |
| Registers | ≈256 KB/SM | >20 TB/s | ≈1 cycle |
| Shared Mem（L1） | 228 KB/SM | ≈20 TB/s | ≈20 cycles |
| L2 Cache | 50 MB | ≈12 TB/s | ≈150 cycles |
| HBM3（VRAM） | 80 GB | 3.35 TB/s | ≈300 cycles |
| NVLink | Multi-GPU | 900 GB/s | microseconds |
| PCIe | CPU-GPU | 64 GB/s | microseconds |

FlashAttention 的核心洞见是：标准 attention 会在 HBM 中计算完整的 attention matrices，但通过对操作进行 tiling，使其适配 shared memory，我们可以避免大部分 HBM 流量，并达到接近最优的内存效率。

## 2.4 CPU 与边缘硬件

并非所有推理都发生在数据中心 GPU 上。CPU 推理对于延迟敏感型应用、本地部署以及成本优化都很关键：

用于推理的现代 CPU 使用 AVX-512 VNNI（Vector Neural Network Instructions）高效执行 INT8 矩阵-向量乘法。Intel 的第 4 代 Xeon Scalable processors 可以提供约 10 TFLOP/s 的 INT8 吞吐量，足以支持小模型。

Apple Silicon（M3 Ultra）将 CPU 和 GPU 内存统一到单一池中（最高 192 GB），带宽约为 800 GB/s。这种 unified memory architecture 使其特别适合对大型模型进行推理，否则这些模型需要多个离散 GPU。

移动 SoC 中的 Qualcomm Hexagon DSPs 和 NPUs 是专为低功耗 neural network inference 而构建的，可在毫瓦级功耗预算下提供数 TOPS（Tera-Operations Per Second）。

## 2.5 新兴硬件趋势

- Google TPU v5p：自定义矩阵引擎，配备芯片间互连，可构成包含数千颗芯片的 pod，用于大模型推理。
- Groq LPU：确定性推理加速器，不使用 DRAM，所有权重都存储在片上 SRAM 中，可在超低延迟下实现每芯片约 500 token/s。
- Cerebras WSE-3：晶圆级引擎，具有 44 GB 片上 SRAM，对于能够放入其中的模型，可完全消除 HBM 瓶颈。
- Photonic computing：早期阶段技术，使用光而不是电子进行矩阵乘法，有望显著降低功耗。

## 面试准备：模拟问答

### Q2.1. 解释 HBM bandwidth 和 NVLink bandwidth 的区别。二者分别在什么时候会成为瓶颈？

答案：HBM（High Bandwidth Memory）是连接到单个 GPU 的设备内存；在 H100 上，它提供 3.35 TB/s，并且在单 GPU 推理中逐 token 加载模型权重时会成为瓶颈。NVLink 是同一节点内多个 GPU 之间的高速互连；在 H100 NVLink 上，它提供 900 GB/s 的总带宽。NVLink 会在 tensor-parallel inference 期间成为瓶颈：当我们把一个大模型分片到多个 GPU 上时，每一层之后，每个 attention head 的输出都必须跨 GPU 执行 all-reduce。如果模型被分片到 8 个 GPU 上，每次 all-reduce 都会消耗 NVLink bandwidth。对于非常大的模型（175B+），NVLink 可能限制 tensor parallelism 的扩展效率。

### Q2.2. 为什么 Apple Silicon 的峰值 FLOP/s 低于离散 GPU，却仍能在 LLM 推理中表现出竞争力？

答案：Apple Silicon 使用 unified memory architecture，其中 CPU、GPU 和 Neural Engine 都共享同一个物理内存池；在 M3 Ultra 上最高可达 192 GB。这消除了困扰离散 GPU 配置的 PCIe 瓶颈（离散 GPU 配置必须通过 64 GB/s 的总线复制数据）。800 GB/s 的 unified memory bandwidth 与离散 GPU 相当；更关键的是，一个 70B 参数模型（FP16 下为 140 GB）可以在不量化的情况下完全放入内存，而当前任何单块离散 GPU 都无法做到这一点。对于 memory-bandwidth-bound 的 decode workloads，相关指标是每美元带宽，而不是每美元 FLOP/s。

### Q2.3. Tensor Core formats（FP16 vs INT8 vs FP8）对推理有什么意义？你会在什么时候选择每一种格式？

答案：Tensor Cores 会以特定低精度格式加速矩阵乘法；在 H100 上，INT8/FP8 可提供 FP16 约 2× 的吞吐量。对于精度至关重要且硬件原生支持的模型，我会选择 FP16/BF16，这是安全基线。INT8（通过 GPTQ、AWQ 或 LLM.int8()）非常适合对最高 70B 参数的模型进行权重量化，在大多数任务上精度损失很小，同时大约减少 2× 内存。FP8（H100 原生支持）正在成为最佳折中点：它比 INT8 具有更好的动态范围，有硬件加速，并且越来越多地用于 TensorRT-LLM 等生产系统。对于 H100 级硬件上的新部署，我会选择 FP8。

### Q2.4. 你会如何 profile 一个 LLM 推理 workload，以判断它是 compute-bound 还是 memory-bandwidth-bound？

答案：我会使用 NVIDIA Nsight Systems 查看时间线视图（SM activity、HBM transfers、PCIe transfers），并使用 Nsight Compute 进行 kernel 级 profiling（achieved FLOP/s vs. peak、memory throughput vs. peak）。具体来说：（1）测量 achieved memory bandwidth 占 peak HBM bandwidth 的比例。（2）测量 achieved FLOP/s 占 peak Tensor Core FLOP/s 的比例。（3）计算关键 kernels 的 operational intensity。如果 memory bandwidth 接近饱和而 FLOP/s utilization 很低，那么我们就是 memory-bound。我还会对 batch sizes 做一次 sweep：如果 throughput 随 batch size 线性扩展，我们处于 memory-bound 区间。如果它趋于平台期，则说明已经达到 compute saturation。

### Q2.5. 描述在 Groq LPU 与 NVIDIA H100 上部署推理的权衡。

答案：Groq LPU 优先考虑确定性的超低延迟：由于所有权重都存储在片上 SRAM 中，因此没有 DRAM 延迟，可以产生极其稳定的约 500 token/s 单流吞吐量。然而，SRAM 昂贵且容量有限，因此 LPU 每颗芯片只能服务达到一定规模的模型，而扩展到非常大的模型需要许多芯片，成本很高。H100 提供了大得多的灵活性：每个 GPU 80 GB HBM、支持跨多个 GPU 的几乎任意模型规模、通过 CUDA 实现可编程性，并拥有优化库生态系统。对于具有固定模型规模的延迟关键型应用（例如使用已知 7B 模型的生产聊天机器人），我会选择 Groq。对于灵活性、大模型（70B+）或绝对延迟不那么关键的高吞吐 batch inference，我会选择 H100。
