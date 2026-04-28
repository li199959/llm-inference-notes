# 第 3 章：Transformer 推理机制

_来源页：PDF 第 13-16 页_

## 3.1 Transformer 前向传播剖析

标准的仅解码器 Transformer（例如 GPT、LLaMA 或 Mistral）会让 token 经过由 \(L\) 个相同层组成的堆栈。每一层包含两个子组件：Multi-Head Attention（MHA）块和 Feed-Forward Network（FFN），二者后面都接有层归一化和残差连接。

对于一个包含 \(n\) 个输入 token、模型维度为 \(d\) 的序列：

1. Embedding：将 token ID 映射为向量 \(X \in R^{n \times d}\)。
2. 对每一层 \(\ell = 1 ... L\)：
   1. 计算 queries、keys、values：\(Q = XW_Q,\ K = XW_K,\ V = XW_V\)
   2. 计算注意力：\(\operatorname{Attn}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V\)
   3. 应用带门控的 FFN：\(\operatorname{FFN}(x) = \sigma(xW_1) \odot (xW_3) \cdot W_2\)
3. Language model head：投影到词表 logits 并采样。

## 3.2 KV Cache：它是什么，为什么重要

如果没有缓存，自回归生成就需要在每一步为所有先前 token 重新计算 keys 和 values，这会带来序列长度上的 \(O(n^2)\) 成本。KV cache 解决了这个问题：在 prefill 阶段之后，我们会存储每一层、每个已处理 token 的 key 和 value 矩阵，并在后续 decode 步骤中复用它们。

**定义：KV Cache**

KV cache 是一种数据结构，用于存储注意力计算中所有先前已生成 token 的 key（K）和 value（V）张量，使 decode 步骤能够在不重新计算的情况下关注先前上下文。

单个序列的 KV cache 内存占用：

\[
\text{Memory} = 2 \times L \times n \times n_{\text{heads}} \times d_{\text{head}} \times \text{bytes\_per\_element}
\]

对于 LLaMA-2 70B（80 层、64 个 head、128 head dim、FP16、4096 context）：

\[
2 \times 80 \times 4096 \times 64 \times 128 \times 2 \approx 10.7\ \text{GB}
\]

这种巨大的内存成本，正是让 KV cache 管理成为 LLM serving 核心挑战的原因。

## 3.3 Prefill 与 Decode：两种截然不同的工作负载

**Prefill 阶段**

在一次前向传播中并行处理全部 \(n\) 个 prompt token。主导操作是 FFN 层中形状为 \([n, d] \times [d, 4d]\) 的 GEMM（General Matrix Multiplication）。对于长 prompt（较大的 \(n\)），这是 compute-bound 的，并能在 Tensor Cores 上实现较高的 GPU 利用率。

**Decode 阶段**

每一步处理一个 token。主导操作是形状为 \([1, d] \times [d, 4d]\) 的 GEMV（General Matrix-Vector Multiply）。这是 memory-bandwidth-bound 的：每一步我们都要加载约 140 GB 的权重（对于 FP16 的 70B 模型），只为执行一个非常小的计算。Tensor Cores 的 GPU 利用率通常低于 10%。

## 3.4 注意力变体：从 MHA 到 MLA

上下文长度的爆炸式增长推动了一系列注意力架构创新，用于降低 KV cache 大小：

**Multi-Head Attention（MHA）：** 标准注意力包含 \(H\) 个 head。每个 head 都有自己的 K 和 V 矩阵。KV cache 与 \(H\) 成正比。

**Multi-Query Attention（MQA）：** 所有 query heads 共享一组 K 和 V 矩阵。它可将 KV cache 缩小 \(H\times\)，但可能损害模型质量。

**Grouped Query Attention（GQA）：** 每 \(G\) 个 query heads 组成一组，共享一组 K、V。LLaMA-3、Mistral、Gemma 都使用了它。它是在 MHA 和 MQA 之间一个有原则的折中方案。

**Multi-head Latent Attention（MLA）：** 由 DeepSeek-V2 引入。它将 keys 和 values 投影到低维 latent space \(c \ll d\)，并在 KV cache 中只存储 latent vectors。与 MHA 相比，它能将 KV cache 缩小约 \(5-13\times\)，同时质量下降很小。

KV cache 大小比例：MHA : GQA : MLA \(\approx 8 : 2 : 1\)

## 3.5 推理时的位置编码

Rotary Positional Embeddings（RoPE）现在是仅解码器 LLM 中占主导地位的位置编码方案。RoPE 通过在复平面中旋转 query 和 key 向量来编码位置：

\[
q_m = q \cdot e^{im\theta},\quad k_n = k \cdot e^{in\theta}
\]

因此 \(\langle q_m, k_n \rangle\) 只依赖于相对位置 \(m - n\)。在推理时，RoPE 可以通过 NTK-aware scaling 或 YaRN 实现上下文长度外推，让在 4K context 上训练的模型借助适当的位置插值泛化到 128K+ context。

## 面试准备：模拟问答

### Q3.1. 解释 Multi-Query Attention 与 Grouped Query Attention。为什么它们对推理很重要？

**回答：** 在标准 Multi-Head Attention 中，\(H\) 个 attention heads 中的每一个都有自己独立的 K 和 V 投影矩阵。这意味着 KV cache 会随 head 数量线性增长。Shazeer（2019）提出的 Multi-Query Attention（MQA）为所有 query heads 使用单个共享的 K 和 V，从而将 KV cache 缩小 \(H\times\)。不过，在所有 head 之间共享可能会降低模型质量。Grouped Query Attention（GQA）是一个泛化形式：heads 被划分为 \(G\) 组，每组共享一个 K/V 对。这会将 KV cache 缩小 \(H/G\times\)，同时比 MQA 保留更多容量。对推理而言，这些技术会直接降低内存压力，从而支持更长上下文和更大的 batch size。LLaMA-3 70B 使用 GQA，为 64 个 query heads 配置 8 个 KV heads，实现了 \(8\times\) 的 KV 缩减。

### Q3.2. 请说明在 7B 模型上生成 1000 个 token 时，KV cache 会发生什么。

**回答：** 在 prefill 期间，所有 prompt token 会被并行处理，它们的 keys 和 values 会被计算出来，并为所有层存入 KV cache。对于一个 7B 模型（32 层、32 heads、128 head dim、FP16），每个 token、每层的 KV cache 为 \(2 \times 32 \times 128 \times 2 = 16{,}384\) bytes。若在 512-token prompt 之上生成 1000 个 token：KV cache 会从开始时的 \(512 \times 16{,}384 \times 32\) bytes 增长到结束时的 \(1512 \times 16{,}384 \times 32 \approx 790\) MB。每个 decode 步骤都会读取整个 KV cache（它会随着每一步增长），计算新 query 对所有缓存 keys 的注意力，并将新的 key/value 对追加到 cache 中。这种不断增长的内存读取成本，就是长生成过程中 decode latency 会略微增加的原因。

### Q3.3. 什么是注意力复杂度瓶颈，现代系统如何解决它？

**回答：** 标准注意力在时间和空间上都有 \(O(n^2)\) 复杂度：attention matrix \(QK^\top\) 的大小是 \(n \times n\)。当 \(n = 128{,}000\) 个 token 时，这个矩阵大小为 \(128{,}000^2 \times 2\) bytes，约为 32 GB，太大而无法在 HBM 中物化。现代系统通过以下方式解决这一问题：（1）FlashAttention：基于 tile 的计算，避免在 HBM 中物化完整 attention matrix；（2）Sparse attention patterns（Longformer、BigBird）：只关注 token 的一个子集；（3）Linear attention approximations：用 kernel methods 近似 softmax attention，以实现 \(O(n)\)；（4）State Space Models（Mamba）：用一种循环结构完全替代 attention，以 \(O(n)\) 时间和 \(O(1)\) 内存处理长序列。

### Q3.4. RoPE 如何实现上下文长度外推？如果直接把上下文扩展到训练长度之外，会出现什么问题？

**回答：** RoPE 使用应用于 query 和 key 向量的旋转矩阵来编码位置，其中维度 \(i\) 的旋转角为 \(\theta_i = 1 / 10000^{2i/d}\)。这意味着高频维度会随位置快速旋转，而低频维度旋转较慢。如果直接外推到训练上下文长度 \(n\) 之外，高频维度会遇到训练中从未见过的旋转角，导致注意力分数分布变成 out-of-distribution。模型会失去准确区分相对位置的能力。YaRN 和 NTK-aware scaling 通过基于扩展比例重新缩放基频 \(\theta\) 来解决这一问题，确保所有维度在新的上下文长度下仍保持 in-distribution。

### Q3.5. 为什么 decode 阶段是 memory-bandwidth-bound，而 prefill 是 compute-bound？这对 serving system design 有什么影响？

**回答：** 在 decode 期间，我们处理的是单个 token，这意味着矩阵操作是一个维度为 1 的 matrix-vector multiplications（GEMV）。GEMV 的 arithmetic intensity 约为每 byte 2 FLOP（加载一次权重矩阵以完成一个点积），远低于 compute-bound 操作的 roofline threshold。GPU tensor cores 被严重低效利用。在 prefill 期间，我们同时处理 \(n\) 个 token，这意味着操作是 matrix-matrix multiplications（GEMM），具有高得多的 arithmetic intensity（约 \(2n/d\) FLOP per byte），当 \(n\) 较大时会接近计算饱和。对于 serving 设计，这意味着我们应该：（1）使用 continuous batching 来增加 decode 期间的有效 batch size；（2）考虑 disaggregated prefill-decode serving（Splitwise、Mooncake），让 prefill 运行在 compute-optimized nodes 上，而 decode 运行在 memory-bandwidth-optimized nodes 上。
