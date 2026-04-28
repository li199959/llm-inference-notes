# 第 9 章：大型模型推理的并行化

_来源页码：PDF 第 34-36 页_

## 9.1 为什么并行化是必要的

一个 405B 参数的模型在 BF16 下仅权重就需要 810 GB 的 GPU 显存。即使是目前最大的单块 GPU（配备 80GB HBM 的 H100 SXM）也无法容纳这个模型。对于前沿规模模型来说，并行化不是可选项，而是服务它们的唯一方式。

## 9.2 Tensor Parallelism

Tensor Parallelism（Shoeybi 等，2019；Megatron-LM）会将单个权重矩阵切分到多块 GPU 上。对于线性层 `Y = XW`：

Column Parallelism：沿列方向切分 `W`：`[W1 | W2]`。每块 GPU 独立计算 `Y_i = XW_i`，随后通过一次 AllGather collective 将结果拼接起来。

Row Parallelism：沿行方向切分 `W`。每块 GPU 计算一个部分和，随后通过 AllReduce 将结果合并。

在 MLP block 中，对第一层使用 Column Parallelism、对第二层使用 Row Parallelism，每个 transformer layer 只需要一次 AllReduce。这在 NVLink 上很高效，但在 PCIe/InfiniBand 上代价较高。

Scaling efficiency：Tensor Parallelism 在扩展到 attention head 数量之前表现良好（对于具有 8 个 KV heads 的模型，可扩展到 8 块 GPU；再往上，某些 GPU 上会没有 head 可处理）。通信开销会随 Tensor Parallelism degree 近似二次增长，因此实践中通常限制在单个 NVLink 互联节点上的 8 块 GPU 以内。

## 9.3 Pipeline Parallelism

Pipeline Parallelism 会将 transformer 的不同层分配到不同 GPU 上。GPU 0 持有第 1-10 层，GPU 1 持有第 11-20 层，依此类推。数据像流水线一样流动：GPU 0 计算并将 activations 传给 GPU 1，GPU 1 计算后传给 GPU 2，如此继续。

关键挑战是 pipeline bubbles：后续 stage 中的 GPU 必须等待前序 stage 产生 activations。Micro-batching 可以降低 bubble fraction；对于 pipeline depth 为 `p`、micro-batch 数为 `m` 的情况，bubble fraction 约为 `(p - 1) / (m + p - 1)`。

Pipeline Parallelism 的通信效率较高（只在 stage 边界传递 activations，而不是完整的 AllReduce），并且可以跨由较慢 InfiniBand 连接的节点扩展，因此适合非常大的模型（>100B 参数）。

## 9.4 Mixture of Experts Inference

Mixture of Experts（MoE）模型（Mixtral、DeepSeek-V2、Grok-1）会用一组 `E` 个 expert networks 替换 dense FFN layers。router 会为每个 token 选择 top-K experts：

```math
y = \sum_{k \in \operatorname{top-K}} g_k \cdot \operatorname{Expert}_k(x)
```

MoE 模型具有很高的参数总量，但每个 token 的 active parameter count 较低：Mixtral 8×7B 共有 47B 参数，但每个 token 只激活 13B 参数。这使推理更快（每个 token 的计算量更少），但服务起来更困难：

- Expert Parallelism：每块 GPU 承载一部分 experts。tokens 通过 AllToAll collectives 在 GPU 之间路由。
- Expert load imbalance：某些 experts 可能接收大量 tokens，而其他 experts 接收很少，导致 GPU 负载不均。
- Expert offloading：对于非常大的 MoE 模型，experts 会被 offload 到 CPU/NVMe，并按需加载。

## 面试准备：模拟问答

### Q9.1. 解释 transformer 的 MLP block 中的 Tensor Parallelism。需要哪些通信 collectives，分别在什么时候需要？

回答：在标准 MLP block `Y = GeLU(XW1)W2` 中，跨 `N` 块 GPU 进行 Tensor Parallelism：（1）Column-parallel first layer：`W1` 沿列方向切分，因此每块 GPU 持有 `[W1]_{:, i:j}`，并独立计算 `H = GeLU(XW1)` 的一个 shard。这里不需要通信。（2）Row-parallel second layer：`W2` 沿行方向切分，与 `H` 的列切分相匹配。每块 GPU 计算 `Y_i = H_iW_{2,i}`，即一个 partial sum。跨全部 `N` 块 GPU 的 AllReduce 会将这些 partial sums 合并为完整的 `Y`。总计：每个 MLP block 一次 AllReduce，每个 attention block 也一次 AllReduce（结构相同）。AllReduce volume：每层 `2 × batch size × sequence length × d_model`。在 NVLink 上这很快；在 PCIe 上它可能主导延迟。

### Q9.2. 在推理中，什么时候会选择 Tensor Parallelism 而不是 Pipeline Parallelism？

回答：我会在以下情况下选择 Tensor Parallelism：（1）模型可以放入单节点的 GPU，并且节点内有 NVLink interconnect。TP 需要频繁 AllReduce，因此需要高带宽。（2）最小化延迟至关重要。TP 会按 GPU 数量成比例降低每层计算时间（每块 GPU 只承担每层 `1/N` 的工作）。（3）模型有许多 attention heads，且能够被均匀切分。以下情况我会选择 Pipeline Parallelism：（1）模型太大，无法放入单节点。PP 只需要在 layer 边界通信 activations，因此可以在较慢的跨节点 InfiniBand 上运行。（2）吞吐量比延迟更重要。PP 会引入 pipeline bubble latency，但能够实现较高 batch throughput。（3）MoE 模型的 Expert Parallelism 可以自然地映射到 PP。实践中，生产系统会将两者结合使用（3D parallelism）：节点内使用 TP，节点间使用 PP。

### Q9.3. 解释 MoE 模型中的 expert routing mechanism。它会给推理服务带来哪些挑战？

回答：MoE 模型会用 `E` 个 expert networks 和一个 router 替换每个 FFN layer。router（通常是一个 learned linear projection 加 softmax）会为每个 token 计算 routing weights：`g = softmax(xW_r)`。随后选出 top-K experts（通常 `K = 2`），并对它们的输出进行加权求和。对推理服务来说，挑战包括：（1）Dynamic routing：哪个 expert 处理哪个 token 会随每个输入变化，使负载均衡变得困难。在一个 batch 中，某些 experts 接收的 tokens 数可能比其他 experts 多 10×（expert collapse/load imbalance）。（2）Expert Parallelism communication：在分布式设置中，必须使用 AllToAll collective 将 tokens 路由到正确的 expert GPU，这种通信可能成为吞吐瓶颈。（3）Memory：即使每个 token 只激活 2 个 experts，所有 `E` 个 experts 的权重也必须加载到某处可访问的位置。Expert caching strategies（预加载可能被访问的 experts）可以缓解这一问题。

### Q9.4. Pipeline bubble fraction 如何影响 Pipeline Parallelism 的效率？如何缓解？

回答：当 Pipeline Parallelism 使用 `p` 个 pipeline stages，并逐个处理请求时，pipeline 的前 `p - 1` 个阶段会处于空闲（“bubble”）状态；当最后一个 stage 完成时，前面的 stage 也会空闲。bubble fraction 为 `(p - 1) / p`：对于 `p = 8` 个 stages，会浪费 `7/8` 的 GPU 时间。缓解方式是 micro-batching：将 batch 切分为 `m` 个 micro-batches，并让它们在流水线中并行推进。此时 bubble fraction 变为 `(p - 1) / (m + p - 1)`。对于 `p = 8, m = 32`，bubble 为 `7/39 ≈ 18%`。其他技术包括：（1）1F1B scheduling：交替执行 forward 和 backward passes。（2）Interleaved pipelines（Narayanan 等）：给每块 GPU 分配多个不连续的 layer chunks，从而减少 bubble，但会增加通信。（3）对于仅推理场景（没有 backward），bubble 没有那么严重，因为我们可以简单地用更多请求填满流水线。

### Q9.5. 大型 MoE 模型中的 expert offloading 是什么？它对性能有什么影响？

回答：Expert offloading 会将 expert weight matrices 从 GPU memory（HBM）移动到 CPU DRAM 或 NVMe storage。对于 DeepSeek-V2 这样的模型，它有 236B 总参数量（在某些 MoE layers 中约 160 个 experts），但每个 token 只激活约 20B 参数；不过所有参数都必须位于某个可访问的位置。Expert offloading 的工作方式如下：一个 expert-aware scheduler 会基于 routing decisions 预测下一个 batch 需要哪些 experts，并提前将其权重从 CPU/NVMe prefetch 到 GPU。当前未使用的 experts 的权重会被 evict。

性能影响：PCIe bandwidth（约 32 GB/s）远慢于 HBM bandwidth（2-3 TB/s），因此任何未命中缓存的 expert load 都会增加显著延迟（每个 expert 约毫秒级）。这种方法只有在 batch size 足够大（使 expert access patterns 可预测）且 expert cache hit rate 很高时才效果良好。当 cache hit rate 达到 70% 以上时，expert offloading 可以让单个 8-GPU 节点服务 200B+ MoE 模型。
