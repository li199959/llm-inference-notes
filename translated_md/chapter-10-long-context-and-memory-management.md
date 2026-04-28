# 第 10 章：长上下文与内存管理

_来源页码：PDF 第 37-39 页_

--- 第 37 页 ---

## 10.1 二次方注意力问题

Attention 在序列长度 `n` 上的 `O(n^2)` 复杂度，是长上下文 LLM 推理的核心障碍。对于 `n = 1,000,000` 个 token（1M 上下文，如 Gemini 1.5 Pro），朴素 attention 矩阵需要 `1,000,000^2 × 2 bytes = 2 petabytes`，显然不可行。即使对于 `n = 128,000`（128K 上下文），attention 矩阵也需要 32 GB。

现代长上下文方法从多个角度应对这一问题。

## 10.2 面向长上下文的 FlashAttention

如第 7 章所述，FlashAttention 通过分块避免物化完整的 attention 矩阵。这并不会降低 `O(n^2)` 的计算复杂度，但会把 `O(n^2)` 的 HBM 内存需求降低到 `O(n)`，使得在当前硬件上对 128K token 执行 attention 计算成为可能。

## 10.3 Ring Attention

Ring Attention（Liu et al., 2023）将长上下文 attention 计算分布到多个按环形排列的设备上。每个设备持有序列 QKV 张量的一段 chunk，设备在环中传递 KV chunk，同时计算本地 attention。关键洞见是，online softmax 技巧允许在设备每次只看到一个 KV chunk 的情况下仍然得到正确归一化。Ring Attention 将每设备内存从 `O(n^2)` 降低到 `O(n/p)`，其中 `p` 是设备数量，从而能够在设备集群上支持百万 token 级上下文。

## 10.4 上下文压缩

上下文压缩技术并不对所有上下文计算完整 attention，而是选择性地总结或移除低信息量 token：

LLMLingua（Jiang et al., 2023）使用一个小语言模型，根据每个 token 的上下文为其困惑度打分；低困惑度（容易预测）的 token 会被丢弃。在许多任务上，可以达到 4-20× 的压缩率，同时性能下降小于 5%。

AutoCompressor 对模型本身进行微调，使其将长上下文总结为固定数量的“summary vectors”，随后将这些向量用作压缩后的 KV cache。

## 10.5 作为线性 Attention 的状态空间模型

Mamba（Gu & Dao, 2023）及其后继模型是状态空间模型（SSM），它们不是通过 attention 处理序列，而是通过学习得到的循环状态处理序列：

```text
h_t = A h_{t-1} + B x_t, y_t = C h_t
```

--- 第 38 页 ---

其中，在 selective SSM（S6）中，`A`、`B`、`C` 依赖输入。这实现了推理时 `O(n)` 时间和 `O(1)` 内存（无论序列长度如何，隐藏状态 `h` 都是固定大小）。

在推理时，SSM 等价于 RNN：每个新 token 只需要更新状态，不需要 KV cache。其权衡在于，对于需要精确回忆特定先前上下文的任务，与 attention 相比，SSM 的上下文内检索能力较弱。

## 面试准备：模拟问答

### Q10.1. Ring Attention 如何实现分布式长上下文推理？它的通信成本是多少？

回答：Ring Attention 将序列分布到 `p` 个设备上。设备 `i` 持有 query chunk `Q_i` 以及 key-value chunk `(K_i, V_i)`。在 `p` 个通信步骤中的每一步，设备都会把自己的 KV chunk 传给环中的下一个设备，并接收前一个设备的 KV chunk。在等待下一个 KV chunk 的同时（与计算重叠），每个设备计算本地 `Q_i` 与当前 KV chunk 之间的局部 attention score，并累积 online softmax 统计量。经过 `p` 步之后，每个设备都为自己的 query chunk 计算出了正确的 attention 输出。通信成本：每一步传输 `n/p` 个 token 的 KV cache（两个张量：`K` 和 `V`）。每层总通信量为 `n × d_head × n_heads × 2 × precision`，与总 KV 数据量相同，没有额外通信开销。通信与计算以流水线方式重叠。

### Q10.2. 比较 Mamba（SSM）和 Transformer attention 在长上下文推理中的表现。什么时候会使用各自方案？

回答：Transformer attention：`O(n^2)` 计算和 `O(n)` KV cache 内存。它擅长需要精确回忆上下文中任意位置具体细节的任务（例如，“文档第 3 页关于 X 说了什么？”）。如果使用 FlashAttention，随着上下文增长，性能会较平滑地退化。Mamba：`O(n)` 计算和 `O(1)` 状态（固定大小的循环状态）。它对超长序列非常高效，每个 token 所需计算与总上下文长度无关，保持常数级。弱点：selective SSM 难以从远距离上下文中进行精确检索，在以检索为主的基准（RULER、passkey retrieval）上表现弱于 attention。使用 Mamba 的场景：对超长文档进行流式推理、时间序列预测，以及模型需要跟踪演化状态而不是检索特定事实的应用。混合模型（Jamba、Zamba、Mamba-2 + attention）结合了两者，使用 SSM 层完成大部分处理，并用 sparse attention 层进行检索。

### Q10.3. LLMLingua 如何压缩 prompt？prompt 压缩有哪些风险？

回答：LLMLingua 使用一个小型代理 LLM，根据 token 的（可能已经压缩的）上下文估计每个 token 的困惑度。低困惑度 token 被认为是可预测的，因此信息量低，可作为移除候选。算法包括：（1）粗粒度：确定保留哪些句子或短语；（2）细粒度：在保留的片段内部移除低困惑度 token；（3）目标压缩率决定压缩激进程度。风险：（1）代理模型对“可预测性”的判断可能不同于目标模型，导致重要 token 被丢弃；（2）移除 token 会改变所有后续 token 的位置上下文，使用绝对位置编码的模型可能尤其敏感；（3）对于检索密集型任务（精确查找姓名/数字），即使低困惑度 token 也可能至关重要；（4）压缩是不可逆的，错误无法恢复。实践准则：在生产部署前，应在自己的特定任务分布上验证 LLMLingua。

--- 第 39 页 ---

### Q10.4. 解释 YaRN 与 NTK-aware scaling 在 RoPE 上下文扩展中的区别。

回答：YaRN 和 NTK-aware scaling 都试图将一个使用 RoPE、在上下文长度 `L` 上训练的模型扩展到更长上下文 `L' > L`，且不进行微调（或只进行少量微调）。NTK-aware scaling（bloc97, 2023）会重新缩放 RoPE 基频：不再使用 `base b = 10,000`，而是使用 `b' = b * (L'/L)^{d/(d-2)}`。这会让旋转在扩展范围内分布得更均匀，避免高频维度发生环绕。然而，它对所有频率维度应用相同缩放。YaRN（Peng et al., 2023）使用更复杂的方法：它完全不缩放低频维度（这些维度本来就有良好的远距离覆盖能力），对中频应用 NTK 风格缩放，并对高频进行线性插值。这既保留了短序列上的分布内行为，又支持更长序列。YaRN 还应用 attention temperature 校正（将 attention score 按 `sqrt(1/t)` 缩放），以补偿变化后的 attention 分布。在扩展上下文长度下，YaRN 在困惑度和检索基准上持续优于 NTK-aware scaling。

### Q10.5. 什么是 RAG？它与长上下文推理有什么关系？什么时候应选择其中一种？

回答：RAG（Retrieval-Augmented Generation）会在查询时从向量数据库中选择性地获取相关文档，只把最相关的段落插入 prompt。它实际上用检索系统替代了对超长上下文窗口的需求。长上下文推理则是把所有潜在相关信息直接放入模型的上下文窗口。选择 RAG 的场景：（1）语料库非常大（数百万篇文档），远超任何可行的上下文窗口；（2）检索足够精确，查询能够清楚映射到特定文档；（3）检索索引带来的延迟可以接受；（4）长上下文计算成本过高。选择长上下文的场景：（1）任务需要综合或比较文档中许多部分的信息（例如，对整本书进行复杂推理）；（2）检索噪声较大，模型能从访问完整上下文中受益，以区分相关内容和无关内容；（3）文档语料足够小，可以放入上下文窗口（小于 1M token）。理想系统往往会结合两者：先用 RAG 缩小候选集合，再用长上下文对检索出的段落进行推理。
