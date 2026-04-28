# 第 11 章：边缘和端侧推理

_来源页码：PDF 第 40-42 页_



--- 第 40 页 ---

高效 LLM 推理 第 11 章：边缘和端侧推理

第 11 章
边缘和端侧推理

## 11.1 边缘推理格局

在边缘设备上运行 LLM，如智能手机、笔记本电脑、嵌入式系统，正因隐私、延迟和离线能力而变得越来越重要。约束非常严苛：一部旗舰智能手机可能只有 8-16 GB 统一内存（CPU 和 GPU 共享），功耗预算约为 10 瓦。相比之下，一块 H100 GPU 消耗 700 瓦功率，并配备 80 GB 专用 HBM。

边缘推理工程的核心，是在这些约束内获得尽可能高的智能水平。

## 11.2 边缘场景的内存约束与量化

一个 FP16 的 7B 参数模型需要 14 GB 内存，超过了大多数智能手机的内存容量。4-bit 量化（AWQ/GPTQ）可将其降低到 3.5 GB，使 7B 模型能够在高端手机上运行。配备 8 GB RAM 的 Apple iPhone 16 Pro 可以通过共享系统内存管理运行 4-bit Llama 3 8B。

更激进的量化（2-bit、QuIP#）可将其降低到约 1.8 GB，使 7B 模型能够在中端设备上运行，不过会带来一定质量下降。

## 11.3 运行时框架

llama.cpp（Gerganov，2023）是 LLaMA 及其兼容模型的纯 C/C++ 实现，具备以下能力：

- 通过 AVX-512、NEON（ARM）和 WebAssembly 进行 CPU 推理
- 将部分或全部计算卸载到 Metal（Apple）、CUDA、Vulkan、OpenCL 进行 GPU 推理
- 自定义 k-quant 格式（Q4_K_M、Q5_K_M 等），通过分组量化实现质量-尺寸的帕累托效率
- 通过基于 mmap 的权重加载实现极高内存效率的 attention

Apple MLX 是 Apple 面向 Apple Silicon 的原生 ML 框架，利用了统一内存架构。MLX 支持在配备 M3 Ultra（192 GB）的 Mac Studio 上运行 4-bit 70B 模型，达到约 20 tokens/s。

ONNX Runtime 提供跨平台推理能力，可在来自不同厂商的 CPU、GPU 和 NPU 上使用硬件加速。

## 11.4 面向效率的神经架构搜索

神经架构搜索（Neural Architecture Search，NAS）会自动设计针对特定硬件约束优化的模型架构。对于边缘部署，NAS 可以发现：

- 给定延迟预算下的最优层宽度和深度



--- 第 41 页 ---

高效 LLM 推理 第 11 章：边缘和端侧推理

- 混合精度分配（不同层使用不同 bit-width）
- attention head 裁剪模式
- block skipping 或 early exit 策略

硬件感知 NAS（HAT、OFA）会训练一个“supernet”，可在推理时从中抽取不同规模的子网络，而无需重新训练。

## 面试准备 - 模拟问答

### Q11.1. 要求你在一部 6 GB RAM 的智能手机上部署一个有竞争力的语言模型。你的方案是什么？

回答：6 GB 可用 RAM 必须与 OS 和其他应用程序共享，留给模型的大约只有 4-4.5 GB。我的方案是：（1）从 3B 或 4B 参数模型开始（Llama 3.2 3B、Phi-3 Mini 3.8B、Gemma 2 2B），它们在 4-bit 下只需要 1.5-2 GB；（2）通过 llama.cpp 或 AWQ 应用 Q4_K_M 量化，这能为小模型取得最佳质量-尺寸平衡；（3）在 iOS 上使用 Metal GPU offloading（llama.cpp + Metal backend）加速推理；（4）实现 prompt caching，避免重新计算 system prompt 的 KV cache；（5）如果延迟至关重要，则使用更小的 draft 模型（例如 1B 模型）进行 speculative decoding；（6）目标设置为 10-20 tokens/s，这对聊天界面来说会有响应迅速的体感。如果 3B 的质量不足，可以考虑 Q2_K 的 7B（约 2.5 GB），并接受一定质量下降。

### Q11.2. 解释 Apple MLX，以及它为什么非常适合 Apple Silicon 上的 LLM 推理。

回答：MLX 是 Apple 面向 Apple Silicon 的数组计算库，其设计充分考虑了统一内存架构。PyTorch 将 CPU 和 GPU 视为相互独立的内存空间，需要显式复制数据；相比之下，MLX 在一个 CPU 和 GPU 都可访问的共享内存空间上运行，无需复制。对于 LLM 推理来说，这是变革性的：模型权重只需加载一次到系统内存中，GPU（Apple Silicon 的集成 GPU）即可立即访问，从而消除 PCIe 瓶颈。在配备 192 GB 统一内存和 800 GB/s 带宽的 M3 Ultra 上，一个 4-bit 70B 模型（约 40 GB）可以轻松装入，并为 KV cache 留出空间，达到约 20 tokens/s。MLX 还提供 Python 接口、JIT 编译、惰性求值，并通过 mlx-lm 包持续扩展模型支持。它代表了在消费级硬件上运行前沿规模模型的最实用途径。

### Q11.3. llama.cpp 中的 k-quant 系统是什么？它与标准均匀量化有什么不同？

回答：llama.cpp 实现了“k-quants”，这是一族混合精度 block quantization 格式。k-quants 并不是把每个权重独立地量化为 k bits，并使用 per-tensor 或 per-channel scale；相反，它采用分层方案：权重被分组为 block（通常为 32 个权重），在每个 block 内，某些权重以比其他权重更高的精度存储。以 Q4_K_M 为例：“K”后缀表示 super-block 量化方案，“M”表示质量档位（small、medium、large）。scale 和 minimum value 以更高精度按 super-block 存储，而单个权重使用 4-bit 编码。这优于均匀量化，原因是：（1）block 级 scale 能适应局部权重分布，减少量化误差；（2）重要权重（接近分布尾部的权重）获得更好的表示；（3）在相同 bit budget 下，最终质量/尺寸比优于朴素 INT4。Q4_K_M 被广泛认为是 llama.cpp 中最实用的 4-bit 格式。



--- 第 42 页 ---

高效 LLM 推理 第 11 章：边缘和端侧推理

### Q11.4. 比较 llama.cpp 的 CPU 推理和 GPU 推理。你什么时候会更倾向于 CPU 推理？

回答：GPU 推理（Metal、CUDA、Vulkan）使用设备的并行计算单元和高带宽内存来获得最大吞吐量，在同一模型上通常比 CPU 快 5-10 倍。CPU 推理使用 AVX-512/NEON SIMD 指令、系统 RAM（远慢于 GPU HBM）和顺序处理。CPU 推理较慢，但具有一些优势：（1）无 VRAM 限制：系统 RAM（桌面端为 64-512 GB）可以容纳比任何 GPU VRAM 都大得多的模型；（2）散热：CPU 在被动散热环境中可以更长时间地持续推理；（3）可用性：CPU 推理可在任何机器上运行，不需要 GPU。我会在以下情况下优先选择 CPU：（1）模型太大，无法放入 GPU VRAM；（2）设备没有 GPU，或只有非常弱的集成 GPU；（3）吞吐量并不关键，低速 batch processing 可以接受。部分 GPU offloading（llama.cpp 的 -ngl 标志）允许你卸载尽可能多的层到 GPU VRAM 中，其余部分在 CPU 上运行，从而在两者之间取得平衡。

### Q11.5. 什么是 Hardware-Aware NAS？OFA（Once-For-All）如何实现高效边缘部署？

回答：Hardware-Aware NAS 会搜索在准确率与目标硬件指标（延迟、能耗、内存）之间达到帕累托最优的模型架构。它不是产生单个固定模型，而是产生一族具有不同准确率-效率权衡的模型，每个模型都针对特定硬件进行优化。Once-For-All（OFA）（Han et al.，2020）训练一个大型“supernet”，可以从中抽取不同深度、宽度和 kernel size 的子网络，而无需重新训练。OFA 使用 progressive shrinking：先训练完整网络，然后通过从完整网络中采样较小子网络，并应用来自完整网络的 knowledge distillation，逐步训练这些较小子网络。在部署时，给定目标设备和延迟约束后，查找表或预测器会估计子网络延迟，并选择预算内准确率最高的子网络。对于 LLM 推理，OFA 风格方法可以产生规模相差 2-4 倍的模型，而这些模型都来自同一次训练运行。
