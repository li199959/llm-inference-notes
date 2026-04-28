# 第 5 章：投机解码

## 5.1 核心洞察：草拟与验证

投机解码（Leviathan et al., 2023；Chen et al., 2023）利用了一个根本性的不对称：验证一串 token 是否会由大模型生成，可以在一次并行前向传播中完成；而以自回归方式生成这些 token，则需要每个 token 进行一次前向传播。

该算法非常优雅：

1. 使用一个小型 draft model（3-7B 参数）以较低成本自回归生成 \(K\) 个候选的“草稿”token。
2. 对全部 \(K\) 个草稿 token 并行运行一次大 target model 的前向传播。
3. 使用拒绝采样，根据 target model 的分布验证每个草稿 token。匹配的 token 被接受；出现分歧的位置则拒绝该 token，并执行校正步骤。
4. 结果可以证明与从 target model 分布中采样完全等价。

预期加速比为

\[
\frac{1-\alpha^K}{1-\alpha}
\]

其中 \(\alpha\) 是逐 token 接受率。当 \(\alpha = 0.8\) 且 \(K = 5\) 时，每步预期接受的 token 数约为 3.3，从而带来理论上 3.3× 的解码吞吐提升。

## 5.2 Token 树验证

基础投机解码生成单条草稿序列。基于树的投机解码则生成一棵草稿序列树：在每个位置，draft model 提出多个候选 token，形成分支树结构。target model 使用树掩码注意力模式，在一次前向传播中验证所有树分支。通过同时探索多个备选项，这会显著提高高接受率出现的概率。

## 5.3 Medusa：多头投机解码

Medusa（Cai et al., 2024）消除了对独立 draft model 的需求。它改为在 target model 顶部添加 \(K\) 个额外的“Medusa heads”，即轻量级线性层。每个 head \(k\) 在一次前向传播中预测 token \(t + k\)（第 \(k\) 个未来 token）：

\[
\mathrm{logits}_{t+k} = \mathrm{MedusaHead}_k(h_t)
\]

其中 \(h_t\) 是位置 \(t\) 的最后一个隐藏状态。这使模型能够以自投机方式提出多个未来 token。Medusa-2 进一步加入自蒸馏以提升草稿质量，在生成基准上实现了 2-3× 加速。

## 5.4 EAGLE：高效自回归生成

EAGLE（Li et al., 2024）解决了 Medusa 的一个关键限制：仅从最后一个隐藏状态预测未来 token，会丢失关于精确 token 序列的信息，从而限制接受率。EAGLE 转而训练一个轻量级 draft model，它同时接收隐藏状态和先前已生成 token 的特征嵌入作为输入，使其能够产生精度高得多的 next-token 预测。EAGLE-2 进一步引入动态草稿树，根据草稿预测的置信度自适应调整树结构，在代码和推理任务上实现了 3-4× 加速。

## 5.5 投机解码何时有帮助，何时没有帮助

有利于投机解码的条件：

- 单流、低延迟服务：当 batch size 为 1 时，target model 利用率不足。投机解码可以摊销空闲计算。
- 高接受率：输出更可预测的任务（代码补全、结构化生成、事实问答）会得到较高的 \(\alpha\) 和较大的加速。
- 足够的 GPU 内存：需要同时容纳 draft model 和 target model。

> **注意**  
> 在高 batch size 的服务场景中，如果 GPU 已经饱和，投机解码不会带来吞吐收益。它是一种延迟优化，而不是吞吐优化。

## 面试准备：模拟问答

### Q5.1 证明投机解码生成的输出来自与 target model 相同的分布。

答案：证明依赖于拒绝采样。假设 draft model 以概率 \(q(x)\) 提出 token \(x\)，target model 的分布为 \(p(x)\)。我们以概率 \(\min(1, p(x)/q(x))\) 接受该草稿 token。如果被拒绝，则从残差分布 \(p'(x) \propto \max(0, p(x) - q(x))\) 中采样。任一步的边际接受概率为

\[
\sum_x q(x)\min(1, p(x)/q(x)) = \sum_x \min(q(x), p(x)) = 1 - \mathrm{TV}(p, q)
\]

最终输出分布为：被接受的 token 遵循 \(q(x)\cdot \min(1, p(x)/q(x))\)，而校正项确保剩余概率质量与 \(p(x)\) 匹配。二者组合后的分布精确积分为 \(p(x)\)。

### Q5.2 投机解码中的接受率 \(\alpha\) 是什么？哪些因素会影响它？

答案：接受率 \(\alpha\) 是单个草稿 token 被 target model 的拒绝采样准则接受的概率。它由 draft model 分布 \(q(x)\) 与 target model 分布 \(p(x)\) 之间的重叠决定：

\[
\alpha = 1 - \mathrm{TV}(p, q) = \sum_x \min(p(x), q(x))
\]

提升 \(\alpha\) 的因素包括：（1）任务可预测性：代码和结构化文本具有更确定的 next token；（2）draft-target 对齐：draft model 来自同一模型家族，或是 target model 的蒸馏版本；（3）温度：较低的采样温度会让两个分布都更尖锐，也更可能一致；（4）较小的投机窗口：较少的草稿 token（\(K\)）会降低拒绝概率的累积。典型取值是：代码生成中 \(\alpha = 0.7-0.9\)，开放式聊天中 \(\alpha = 0.5-0.7\)。

### Q5.3 Medusa 与独立 draft model 方法有什么不同？权衡是什么？

答案：独立 draft model 是一个拥有自身权重、KV cache 和前向传播的小型独立模型。它的优势是灵活：draft model 可以是任何兼容的更小模型，其质量也可以独立改进。缺点是内存开销（需要存储两个模型）和复杂度（需要管理两个 KV cache）。

Medusa 在 target model 的隐藏状态顶部添加轻量级线性 head，不需要独立模型。它直接使用 target model 的内部表示，而这些表示比小型 draft model 的表示更丰富。权衡在于：Medusa heads 的表达能力弱于完整的 draft model（它们只看到最后一个隐藏状态，而不是完整上下文），因此接受率可能更低。Medusa-2 通过自蒸馏缓解了这一点。Medusa 更易部署；如果精心选择，独立 draft model 可以达到更高质量。

### Q5.4 在什么场景下你不建议使用投机解码？

答案：我不建议在以下场景中使用投机解码：（1）高 batch size 的生产服务：当 GPU 在 32+ 的 batch size 下已被计算饱和时，并行验证步骤会消耗原本可以用于服务额外请求的计算资源，吞吐反而下降。（2）内存受限环境：投机解码需要为 draft model（独立模型方案）或 Medusa heads 提供额外内存。如果 GPU 内存已经完全用于 KV cache 和权重，则不可行。（3）接受率天然较低的任务：高温度、大词汇多样性的开放式创意生成会产生较低的 \(\alpha\)（<0.5），意味着频繁拒绝且加速很小。（4）极短输出：草稿循环的开销是固定的；对于 1-2 个 token 的输出，投机解码没有收益。

### Q5.5 描述 EAGLE 架构，并解释为什么它能获得比基础 Medusa 更高的接受率。

答案：EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）训练了一个轻量级自回归 draft model。与 Medusa 一样，它接收来自 target model 的隐藏状态 \(h_t\)；但除此之外，它还接收最近生成 token 的 token embedding \(e_t\)。这个联合输入 \([h_t; e_t]\) 会被送入一个单层 transformer decoder，用于预测下一步隐藏状态 \(h_{t+1}\)，未来 token 则由该隐藏状态预测。

相较于 Medusa，关键优势是自回归草稿生成：EAGLE 可以按序生成多个草稿 token，并让每个 token 都以前一个 token 为条件，而不是全部从同一个固定隐藏状态预测。这种序列条件化捕获了 Medusa 并行 head 所遗漏的 token 间依赖，使接受率提高 5-15 个百分点。EAGLE-2 进一步加入了基于置信度的动态树扩展。
