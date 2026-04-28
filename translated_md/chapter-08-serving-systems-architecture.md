# 第 8 章：服务系统架构

## 8.1 生产级推理服务器的组成

生产级 LLM 推理服务器是一个复杂的分布式系统。它的核心组件包括：

- Frontend / API Gateway：接收 HTTP 请求，执行认证、限流，并将请求入队。
- Scheduler：服务系统的大脑。决定当前 batch 中运行哪些请求，同时管理 GPU 内存和 SLO 约束。
- Execution Engine：执行实际的 GPU forward pass，并管理 KV cache block。
- Tokenizer：将文本转换为 token ID，并将 token ID 转回文本（通常在 CPU 上运行）。
- Sampler：对 logits 应用 temperature、top-k、top-p 以及其他采样逻辑。

## 8.2 Batching 策略

### 8.2.1 Static Batching

最简单的方法是：收集固定数量的请求，将它们全部 padding 到最长序列的长度，然后处理这个 padded batch。这种方法效率很低：padding 会浪费计算和内存，浪费量与长度方差成正比。

### 8.2.2 Continuous（Iteration-Level）Batching

Continuous batching 由 Orca（Yu et al., 2022）开创，并在 vLLM 和 TGI 中实现。它不是在 request level 做 batching，而是在 iteration（decode step）level 做 batching。在每个 decode step，scheduler 都可以把新的请求（其 prefill 已完成）加入正在运行的 batch，并立即移除已经完成的序列。通过消除 static batching 的“long tail”问题，这能让 GPU utilization 保持在较高水平。

### 8.2.3 Chunked Prefill

Sarathi-Serve（Agrawal et al., 2024）观察到，单个很长的 prefill 请求可能会独占 GPU 数百毫秒，使 decode 阶段的请求饥饿（“prefill piracy”）。Chunked prefill 将 prefill 计算拆分为更小的 chunk，并与 decode step 交错执行，从而为所有请求类型提供更可预测的 latency。

## 8.3 Disaggregated Prefill 与 Decode

prefill（compute-bound）与 decode（memory-bound）之间的根本张力表明，它们应该由针对各自模式优化的不同硬件来服务。Disaggregated serving（Splitwise、Mooncake）在物理上分离了：

- Prefill nodes：最大化 compute utilization（大 batch、长 prompt）。可以使用更便宜、更旧的 compute-optimized GPU。
- Decode nodes：最大化 memory bandwidth，支持高 concurrency。受益于每参数 HBM bandwidth 更高的新一代 GPU。

挑战在于 KV cache migration：prefill 完成后，KV cache 必须从 prefill node 传输到 decode node（通过 NVLink、InfiniBand 或 RDMA），这会增加 latency。像 Mooncake（来自 Kimi/Moonshot AI）这样的系统使用位于二级存储上的分布式 KV cache pool，将这一过程完全解耦。

## 8.4 SLO-Aware Scheduling

生产系统必须遵守 Service Level Objectives（SLOs），通常表示为 p95 或 p99 TTFT 与 TPOT 目标。朴素的 FIFO scheduler 可能会让短请求在长请求后面饥饿。SLO-aware scheduling 技术包括：

- Priority queuing：更短或更高优先级的请求可以插队。
- Preemption：当一个长时间运行请求的 SLO 轨迹是安全的，就暂停它，释放其 KV cache block，并服务更高优先级的请求。
- Request routing：根据估计长度和 server load，将请求路由到不同的 server pool。
- Admission control：当系统过载时拒绝请求或让请求排队，以防止 SLO violation。

## 面试准备：模拟问答

### Q8.1. 解释 continuous batching。为什么它在 LLM serving 中优于 static batching？

回答：Static batching 会把长度相近的请求分组，对较短请求进行 padding，并一直处理该 batch，直到所有请求完成。问题在于：如果一个包含 32 个请求的 batch 中，有一个请求比其他请求长 10 倍，那么其他 31 个请求对应的计算槽位都会为这个请求空等。浪费在 padding token 上的计算是永久性的。Continuous batching（Orca、vLLM）在 iteration level 工作：在每个 decode step，scheduler 都会检查是否有正在运行的请求已经完成（发出 `<EOS>`），并立即用队列中新到达的请求替换它们。不同长度请求之间无需 padding；每个请求只占用它实际需要的 token。由于不存在为了落后请求而进行的“batch wait”，GPU utilization 会持续保持较高水平。根据 Orca 论文的 benchmark，在相近 latency 下，continuous batching 相比 static batching 在实践中可将 throughput 提升 5 到 23 倍。

### Q8.2. 什么是“prefill piracy”？chunked prefill 如何解决它？

回答：Prefill piracy 描述的是这样一种情况：带有很长 prompt 的请求在其 prefill 阶段独占 GPU，可能持续数百毫秒，导致 batch 中所有 decode 阶段的请求停滞。在 prefill 期间，GPU 正在对一个很大的 token 矩阵执行 GEMM；那些正在生成 token 的 decode 请求必须等待，导致其 TPOT（time per output token）飙升。Chunked prefill（Sarathi-Serve）会把长 prompt 的 prefill 拆分为小 chunk（例如每个 chunk 512 个 token），并将每个 chunk 与 decode step 交错执行。这意味着一个 4096-token 的 prefill 在让出执行权给 decode batch 之前，只会贡献 512 个 token 的计算，从而将 decode TPOT degradation 降低 5 到 10 倍，同时只给总 prefill 时间增加很小的 overhead。

### Q8.3. 描述 vLLM 的架构。它的关键创新是什么，局限在哪里？

回答：vLLM（Virtual LLM）是一个高吞吐 serving 系统，围绕三项核心创新构建：（1）PagedAttention：基于非连续 block 的 KV cache memory management，消除 fragmentation；（2）Continuous batching：通过 iteration-level batching 提高 GPU utilization；（3）Custom attention kernels：用于 paged attention computation 的 CUDA kernel。其架构是：请求到达 API server，经过 tokenize，然后由 LLMEngine 调度；LLMEngine 通过 BlockManager 管理 KV block，并通过自定义 execution engine 在 GPU worker 上执行。局限包括：（1）vLLM 当前的 scheduler 并非完全 SLO-aware，它可能会让短请求饥饿；（2）tensor parallelism communication overhead 对小模型可能很显著；（3）偏重 Python 的 scheduling layer 会给非常短的请求增加 latency；（4）基础 vLLM 不原生支持 disaggregated prefill/decode；（5）非标准架构（MoE、非 transformer 模型）需要开发自定义 plugin。

### Q8.4. Disaggregated prefill/decode serving 如何工作？有哪些工程挑战？

回答：Disaggregated serving 将 prefill 计算路由到专用 prefill node，将 decode 计算路由到专用 decode node。一个请求到达 router 后，会被发送到 prefill node；prefill node 运行 prompt forward pass 并生成 KV cache，然后 KV cache 通过 RDMA/InfiniBand 传输到 decode node，由 decode node 生成 token 直到完成。工程挑战相当大：（1）KV cache transfer latency：在 400 Gb/s InfiniBand 链路上传输 10 GB KV cache 需要约 200 ms，往往比 prefill 本身还长。优化方式是逐层传输 KV block，并与 decode 进行 pipeline；（2）Resource balancing：prefill node 和 decode node 具有不同的 compute/memory ratio；任一池过度配置或配置不足都会损害整体 throughput；（3）Fault tolerance：如果 decode node 在生成中途失败，KV cache 必须能够恢复；（4）Routing complexity：router 必须估计 request length，才能以最优方式平衡 load。

### Q8.5. 你会使用哪些指标来评估和比较两个 LLM serving system？

回答：我会从以下方面评估：（1）TTFT p50/p95/p99：对交互式 application 至关重要，即 first token 出现得有多快；（2）TPOT p50/p95：streaming smoothness，即后续 token 到达得有多快；（3）Throughput（tokens/s）：给定硬件配置下的系统总输出；（4）Goodput：同时满足所有 SLO 目标（TTFT 和 TPOT）的请求比例；（5）GPU utilization（MFU）：Model FLOP Utilization，即 GPU 峰值 FLOP/s 中有多大比例被有效使用；（6）Cost per million tokens：按输出归一化后的总 TCO；（7）Scalability：随着 concurrent users 增加，performance 如何变化；（8）Tail latency behavior under overload：系统在过载下是优雅降级还是灾难性退化。我会基于真实的 request distribution（混合短 prompt、长 prompt、短输出和长输出）运行 benchmark，而不是使用合成的固定长度 benchmark。
