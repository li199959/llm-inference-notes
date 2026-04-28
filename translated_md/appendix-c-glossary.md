# 附录 C：术语表

| 术语 | 中文译名 / 解释 |
| --- | --- |
| AWQ | Activation-aware Weight Quantization，激活感知权重量化。一种训练后量化（PTQ）方法，在量化前根据激活值幅度对通道进行缩放。 |
| BF16 | Brain Float 16，脑浮点 16。16 位浮点格式，具有与 FP32 相同的指数范围，广泛用于大语言模型（LLM）的训练和推理。 |
| FLOP | Floating Point Operation，浮点运算。 |
| GQA | Grouped Query Attention，分组查询注意力。一种注意力变体，其中多组查询头共享一组键值（KV）对。 |
| GPTQ | 一种逐层的二阶训练后量化（PTQ）方法，利用 Hessian 信息进行权重量化。 |
| HBM | High Bandwidth Memory，高带宽内存。现代 GPU 中使用的 DRAM 类型，例如 HBM2e、HBM3。 |
| KV Cache | Key-Value cache，键值缓存。用于存储先前已生成 token 的注意力键和值。 |
| MFU | Model FLOP Utilization，模型 FLOP 利用率。表示峰值 FLOP/s 中被有效用于计算的比例。 |
| MLA | Multi-head Latent Attention，多头潜在注意力。一种使用压缩潜在 KV 表示的注意力变体。 |
| MoE | Mixture of Experts，专家混合。包含多个专家前馈网络（FFN）并通过 token 路由进行选择的模型架构。 |
| MQA | Multi-Query Attention，多查询注意力。所有查询头共享单一 KV 对。 |
| NVLink | NVIDIA 的高带宽 GPU 到 GPU 互连技术（在 H100 上为 900 GB/s）。 |
| PagedAttention | 使用虚拟内存分页机制进行 KV 缓存内存管理的方法。 |
| PRM | Process Reward Model，过程奖励模型。对中间推理步骤进行评分的奖励模型。 |
| PTQ | Post-Training Quantization，训练后量化。在训练完成后应用的量化方法。 |
| QAT | Quantization-Aware Training，量化感知训练。将量化过程纳入训练中的方法。 |
| RoPE | Rotary Positional Embeddings，旋转位置嵌入。通过复数旋转实现的位置编码。 |
| SSM | State Space Model，状态空间模型。使用学习得到的循环状态的序列模型，例如 Mamba、S4。 |
| TPOT | Time Per Output Token，单个输出 token 耗时。连续输出 token 之间的时间间隔。 |
| TTFT | Time To First Token，首 token 延迟。从请求发出到第一个输出 token 产生所需的时间。 |
| TensorCore | Tensor Core，张量核心。用于加速矩阵乘法的专用 GPU 计算单元。 |
| vLLM | Virtual LLM，虚拟 LLM。使用 PagedAttention 和连续批处理的高吞吐量服务系统。 |
