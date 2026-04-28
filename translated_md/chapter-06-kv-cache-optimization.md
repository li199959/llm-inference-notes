# 第 6 章：KV Cache 优化

## 6.1 KV Cache 内存危机

如第 3 章所示，单个长上下文请求的 KV cache 可能消耗数十 GB 的 GPU 内存。对于一个有数百个并发请求的服务系统来说，朴素地预分配最大长度的 KV cache 会浪费大部分已分配内存（因为实际请求长度不可预测），并严重限制 batch size。这是 2023 年之前 LLM 服务中的核心未解问题。

## 6.2 PagedAttention：KV Cache 的虚拟内存

PagedAttention（Kwon et al., 2023）是 vLLM 的核心创新，它将操作系统中的虚拟内存概念应用到 KV cache 管理中。

PagedAttention 不再为每个请求的 KV cache 分配一段连续内存，而是将 KV cache 划分为固定大小的 block（page），每个 block 包含 `B` 个 token。block table 将每个请求的逻辑 KV cache 位置映射到 GPU 内存中的物理 block，这些 block 不需要连续。

### PagedAttention 的关键特性

- 无需预分配：随着生成过程推进，按需分配 block。
- 无内部碎片：每个请求浪费的内存最多为 `(B - 1)` 个 token。
- 写时复制：共享同一 prefix（例如 system prompt）的多个请求可以共享物理 block，直到它们的生成路径发生分叉。
- 灵活驱逐：可以将某个请求的 block 交换到 CPU 或磁盘，为另一个请求腾出空间。

PagedAttention 需要修改 attention kernel，以便在 attention 计算过程中收集非连续 block。这是一个不平凡的工程挑战，vLLM 通过自定义 CUDA kernel 实现了它。

## 6.3 Prefix Caching（Prompt Caching）

许多生产工作负载会在数千个请求之间共享一个很长的公共 prefix，例如 system prompt、文档或 few-shot 示例。如果没有缓存，每个请求都要从头重新计算这个共享 prefix 的 KV cache。

Prefix Caching（也称为 Prompt Caching，已在 vLLM、SGLang 中实现，并由 Anthropic/OpenAI 作为功能提供）会存储常见 prefix 的 KV cache，并在请求之间复用。当 prefix 相对于用户 prompt 很长时，这可以将 prefill 成本降低 90% 以上。

## 6.4 StreamingLLM：Attention Sink 现象

对于需要无限长上下文的应用（流式对话、长时间运行的 agent），我们无法存储所有 token 的 KV cache，因为它会无界增长。朴素的滑动窗口 attention（只保留最近 `W` 个 token 的 KV cache）会导致灾难性的性能退化。为什么？LLM 会形成 “attention sink”：序列最开始的几个 token（无论其语义内容如何）会在所有层中获得不成比例的高 attention score。移除它们会破坏模型的内部表示。

StreamingLLM（Xiao et al., 2023）会保留初始 token 中一小段 “sink”（通常为 4 个 token），同时保留最近 token 的滑动窗口，从而在有界内存下支持无限长度的 LLM 推理：

```text
KV retained = sink tokens + recent window
```

## 6.5 KV Cache 量化

KV cache 本身可以独立于模型权重量化，因为它在每个 attention step 中只被读取一次，并且任何量化误差只会影响当前 decode step，而不会累积。INT8 KV cache 量化现在已经成为 TensorRT-LLM 和 vLLM 等框架中的标准做法，可带来约 2 倍的 KV 内存降低。更激进的方案会在谨慎进行 per-head 或 per-channel scaling 的情况下，对 KV cache 使用 INT4 甚至 FP8。

## 面试准备：模拟问答

### Q6.1. 解释 PagedAttention。它与预分配的连续 KV cache 管理有什么不同？

回答：传统的 KV cache 管理会为每个请求预分配一段连续 GPU 内存，其大小按可能的最大序列长度确定。这会浪费大量内存，因为大多数请求都远短于最大长度。此外，它还会造成内存碎片：一旦某个 block 被释放，留下的空隙可能太小，无法用于新的 block，从而导致外部碎片。PagedAttention 将 KV cache 内存划分为固定大小的 block（约 16 个 token/block），并为每个请求维护一个 block table，将逻辑位置映射到物理 block。随着新 token 生成，block 按需分配；请求完成后，block 立即释放。这消除了预分配浪费，将碎片减少到每个请求最多 `B - 1` 个 token，并支持跨请求对公共 prefix block 进行写时复制共享。结果通常是在相同 GPU 内存下获得 2-4 倍更高的 batch size。

### Q6.2. Prefix Caching 如何工作？哪类应用从中受益最大？

回答：Prefix Caching 会存储某段 prefix token（例如 system prompt）已计算出的 KV cache，并在后续共享该 prefix 的请求中复用。系统维护一个从 prefix token 序列（或内容哈希）到物理 KV cache block 的哈希表。当新请求到达时，服务系统会检查其开头的 token 位置是否匹配某个已缓存 prefix；如果匹配，则完全跳过这些 token 的 prefill step。

最受益的应用包括：（1）带有固定 system prompt 的 API 产品：例如，一个包含 2000 token system prompt 的客服机器人，每条用户消息都能受益于缓存 prefix；（2）多轮对话：后续轮次可以复用此前所有轮次的 KV；（3）批量文档处理：例如，围绕同一篇文档提出的 1000 个问题共享该文档的 KV cache。现实中的 prefill 阶段延迟可降低 50-90%。

### Q6.3. 什么是 Attention Sink 现象？为什么朴素滑动窗口 attention 对 StreamingLLM 会失败？

回答：当 LLM 在固定长度序列上训练时，它们会学会将大量 attention 指向每个序列最开始的几个 token，这些 token 被称为 “attention sink”。最初的 token 充当聚合器，吸收 softmax 归一化所需的额外 attention 概率质量。这些 sink token 在语义上并不重要；它们只是成为结构性锚点。朴素滑动窗口 attention 会丢弃最旧的 token 以维持固定大小的 KV cache，当窗口滑过序列开头后，它会无意中移除这些 sink token。没有 sink 时，attention 归一化会被破坏：剩余 token 上的 softmax score 会与训练分布显著不同，从而导致输出灾难性退化。StreamingLLM 通过始终在 KV cache 中保留最开始的 4 个初始 sink token，并同时保留最近窗口，恢复正常的 attention 行为。

### Q6.4. 如果系统服务的请求长度高度可变，你会如何设计 KV cache 驱逐策略？

回答：我会实现一种基于优先级的驱逐策略，结合多个信号：（1）最近性（LRU）：优先驱逐最近最少访问的 block；（2）序列完成概率：使用请求的当前位置和典型长度分布来估计它离完成还有多近，优先驱逐可能很快完成的请求；（3）block 可共享性：保护被多个请求共享的 block（prefix cache block），因为它们具有较高的摊销价值；（4）驱逐成本：优先从能够以低成本重新计算 KV cache 的请求（短 prompt）中驱逐，而不是从具有长且昂贵 prefill 的请求中驱逐。我还会在彻底驱逐之前，将 block swap 到 CPU DRAM 作为第二层，因为 CPU 内存很充足。vLLM 的抢占机制就是一个实际例子：当内存压力较高时，它会暂停并重新调度请求。

### Q6.5. 解释 KV cache 量化。你能比模型权重更激进地量化 KV cache 吗？为什么？

回答：KV cache 量化会将 key 和 value（attention cache）存储为较低精度（INT8、INT4 或 FP8），而不是 FP16。支持对 KV cache 采用比权重更激进量化的理由包括：（1）KV cache value 在每个 attention step 中只读取一次，不会跨 step 累积误差（每个 decode step 都从 cache 中读取新的内容）；（2）attention 操作本身经过归一化（softmax），因此对 key 量化误差具有一定鲁棒性；（3）KV cache 在 head 和 layer 之间的冗余度高于权重矩阵。然而，也存在反方理由：activation（KV cache 正是由 activation 派生而来）会表现出类似 activation 量化中的 outlier 现象，因此比权重更难量化。实践中，INT8 KV cache 已经相当成熟，质量损失极小；INT4 KV cache 在使用 per-head 或 per-token scaling 时可行，但需要谨慎实现。
