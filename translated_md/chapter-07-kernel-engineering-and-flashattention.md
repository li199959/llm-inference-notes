# 第 7 章：Kernel Engineering 与 FlashAttention

_来源页：PDF 第 27-30 页_

--- 第 27 页 ---

高效 LLM 推理 第 7 章：KERNEL ENGINEERING 与 FLASHATTENTION

# 第 7 章

# Kernel Engineering 与 FlashAttention

## 7.1 为什么自定义 CUDA Kernel 很重要

像 PyTorch 这样的高级框架会用库原语组合操作：一个线性层就是一次对 `torch.nn.Linear` 的调用，它会进一步调用 cuBLAS 执行 GEMM，然后把结果写入 HBM，再为下一个操作把结果读回来。这种 eager execution 模型会为中间张量产生大量 HBM 流量，而这些中间张量本来可以保存在快速的片上内存中。

Kernel fusion 会把多个操作合并到单个 GPU kernel 中，使中间结果保留在寄存器或共享内存里，从而避免往返 HBM。对于融合后的 LayerNorm+Linear+GeLU kernel，相比三个独立 kernel，我们可以将 HBM 流量减少 3×。

## 7.2 FlashAttention：IO 感知的精确 Attention

标准 attention 会在 HBM 中物化完整的 `n × n` attention 矩阵：

```text
S = QK^T / sqrt(d_k), P = softmax(S), O = PV
```

当 `n = 16,384` 且使用 FP16 时，这个矩阵大小为 `16,384^2 × 2 ≈ 537 MB`。在 A100 上，从 HBM 读写这个矩阵大约需要 0.5 ms，而且每次前向传递都必须发生两次（一次用于计算 `P`，训练时的反向传播还需要一次）。

FlashAttention（Dao et al., 2022）通过 tiling 消除了这一问题：它把 attention 计算分解为能够放入 SRAM 的块，使用 online softmax trick 以增量方式计算数值稳定的 softmax，并且从不在 HBM 中物化完整的 attention 矩阵。

### FlashAttention 复杂度

| 项目 | 标准 Attention | FlashAttention |
| --- | --- | --- |
| HBM 读/写 | `O(n^2)` | `O(n^2/M)`（更少的遍历次数） |
| SRAM 使用量 | `O(n^2)` | `O(M)`（tile 大小） |
| 速度 | 基线 | 快 2-4× |
| 序列长度限制 | 受 HBM 限制 | 可支持更长上下文 |

FlashAttention-2（Dao, 2023）通过沿序列长度维度把工作拆分到多个 thread block，并减少非矩阵乘法操作，从而提升了并行度。

FlashAttention-3（Shah et al., 2024）进一步利用了 H100 特有的能力：通过 WGMMA（Warpgroup Matrix Multiply Accumulate）指令进行异步内存拷贝与矩阵乘累加，并支持 FP8 精度，达到约 75% 的 H100 峰值 FLOP/s 利用率。


--- 第 28 页 ---

高效 LLM 推理 第 7 章：KERNEL ENGINEERING 与 FLASHATTENTION

## 7.3 Online Softmax Trick

对 attention 进行 tiling 的核心数学挑战，是在分块处理序列时仍然计算精确的 softmax。Online softmax（Milakov & Gimelshein, 2018）维护一个运行中的最大值 `m` 和一个运行中的和 `ℓ`，从而在不预先看到所有值的情况下计算 softmax：

```text
m_i = max(m_{i-1}, max_j s_{ij}),
ℓ_i = e^{m_{i-1} - m_i} ℓ_{i-1} + Σ_j e^{s_{ij} - m_i}
```

最后，`O = O_running / ℓ` 给出精确的 softmax 输出。这在数值上是稳定的，并且对 attention 矩阵的每一行只需要 `O(1)` 的额外状态。

## 7.4 Triton：Python 风格的 Kernel 编程

CUDA kernel 开发需要具备 C++ 以及 GPU 专用优化方面的专业知识。Triton（Tillet et al., 2019）是一种用于编写 GPU kernel 的 Python DSL，它抽象掉底层细节，同时生成高度优化的代码：

```python
import triton
import triton.language as tl

@triton.jit
def fused_softmax_kernel(X, Y, stride, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_ptrs = X + row * stride + tl.arange(0, BLOCK)
    x = tl.load(x_ptrs, mask=tl.arange(0, BLOCK) < N)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator
    tl.store(
        Y + row * stride + tl.arange(0, BLOCK),
        y,
        mask=tl.arange(0, BLOCK) < N,
    )
```

Triton 会自动处理共享内存分配、warp 级同步和向量化，使 ML 工程师无需深厚的 CUDA 专业知识也能编写高性能 kernel。

## 面试准备 - 模拟问答

### Q7.1. 解释为什么 FlashAttention 比标准 attention 更快。它解决的关键瓶颈是什么？

回答：标准 attention 会在 HBM 中物化完整的 `n × n` attention 矩阵（`QK^T`）。对于长序列，这个矩阵可能达到数百 MB 到数 GB，而把它读写到 HBM 是主要成本，真正的算术计算反而不是瓶颈。FlashAttention 更快，是因为它是 IO-aware 的：它把 attention 计算 tiled 成能放入 GPU 片上 SRAM（共享内存）的块，并使用 online softmax trick 以增量方式计算 softmax。中间结果保留在快速 SRAM 中，永远不会写入 HBM。每个块的最终输出会被累加，并且只写入 HBM 一次。HBM 流量从 `O(n^2)` 降低到 `O(n · d)`，与读取输入和写出输出的理论最小值一致。在 A100 上，对于长序列，FlashAttention 相比标准 attention 可实现 2-4× 加速。


--- 第 29 页 ---

高效 LLM 推理 第 7 章：KERNEL ENGINEERING 与 FLASHATTENTION

### Q7.2. 什么是 kernel fusion，为什么它能提升推理性能？

回答：Kernel fusion 是把多个计算操作（这些操作通常会各自启动一个单独的 GPU kernel）合并为单个 kernel，使中间结果保留在片上内存（寄存器或共享内存）中，而不是在操作之间写入 HBM。没有 fusion 时：每个操作都会把输出写入 HBM，下一个操作再把它读回来，这会产生冗余的内存流量。有 fusion 时：中间张量驻留在寄存器或共享内存中，其有效带宽超过 10 TB/s，而 HBM 的带宽约为 2-3 TB/s。LLM 推理中的常见 fusion 模式包括：（1）QKV projection 与 RoPE application 融合；（2）LayerNorm 与后续 linear layer 融合；（3）SiLU/GeLU gating 与 FFN up-projection 融合。TensorRT-LLM 和 vLLM 中的 fused kernel 可以同时降低 kernel launch 开销和 HBM 流量，相比未融合的基线可带来 10-30% 的延迟改进。

### Q7.3. Online softmax trick 如何支持 tiled attention 计算？

回答：标准 softmax `softmax(x)_i = e^{x_i} / Σ_j e^{x_j}` 需要先看到一整行中的所有值，才能计算任何输出，因此无法进行 tiling。Online softmax 按块处理一行，同时维护一个运行中的最大值 `m` 和指数和 `ℓ`。当一个包含值 `x'` 的新块到来时：（1）更新运行最大值：`m' = max(m, max(x'))`；（2）重新缩放已有的和：`ℓ' = ℓ · e^{m - m'} + Σ_j e^{x'_j - m'}`；（3）重新缩放已有输出：`O' = (O · ℓ · e^{m - m'} + Σ_j V'_j · e^{x'_j - m'}) / ℓ'`。完成后，`O'` 等于精确的 softmax 加权和。这使 FlashAttention 能够以 tile 方式处理 `Q`、`K`、`V`，在不物化完整 attention 矩阵的情况下产生精确结果。

### Q7.4. FlashAttention-3 针对 H100 引入了哪些改进，它如何利用 H100 特有的能力？

回答：FlashAttention-3 是专门针对 H100 的新硬件能力设计的：（1）WGMMA（Warpgroup Matrix Multiply Accumulate）：H100 引入了新的 warpgroup 级异步矩阵乘法指令，使 FA3 能够在一个 tile 内重叠内存拷贝与计算操作，从而隐藏内存延迟；（2）TMA（Tensor Memory Accelerator）：由硬件管理的 HBM 与共享内存之间的异步传输，降低了数据移动的软件开销；（3）FP8 支持：FA3 支持 FP8 attention 计算，使 attention 的有效 Tensor Core 吞吐量翻倍；（4）Ping-pong pipelining：两个 warpgroup 交替加载下一个 tile 并计算当前 tile，使 Tensor Core 和内存系统都保持充分忙碌。综合这些能力，FA3 在 attention 上达到约 75% 的 H100 理论峰值 FLOP/s，而 FA2 约为 35%。

### Q7.5. 什么时候会使用 Triton，而不是编写原始 CUDA？取舍是什么？

回答：我会在以下情况下使用 Triton：（1）我需要一个自定义 kernel（例如一种新的 fused operation），但团队缺乏深厚的 CUDA 专业知识；（2）kernel 属于“中等复杂度”：它不是简单的 elementwise operation（这类操作由 PyTorch 覆盖），但也不需要手工调优的 warp 级原语；（3）快速迭代很重要，Triton 的 JIT 编译意味着我可以迅速实验。Triton 的优势包括：Python 语法、自动 shared memory tiling、更简单的调试，以及生成代码在许多工作负载上通常能达到手工优化 CUDA 的 80-90% 性能。我会在以下情况下使用原始 CUDA：（1）最高性能至关重要（例如 FlashAttention 中的 attention kernel，需要对内存访问模式和 warp 同步进行极其精确的控制）；（2）我需要对 warp 级原语、bank conflict 避免或异步拷贝流水线（H100 上的 TMA/WGMMA）进行细粒度控制。取舍在于开发时间（CUDA：数周；Triton：数天）与峰值性能。


--- 第 30 页 ---

高效 LLM 推理 第 7 章：KERNEL ENGINEERING 与 FLASHATTENTION
