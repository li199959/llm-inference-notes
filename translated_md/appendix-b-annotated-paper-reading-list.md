# 附录 B：带注释的论文阅读清单

1. Attention Is All You Need (Vaswani et al., 2017) —— 奠定 Transformer 基础的论文。请先阅读这篇。
2. FlashAttention (Dao et al., 2022) 和 FlashAttention-2 (Dao, 2023) —— 理解 IO-aware attention computation 的必读论文。
3. Efficient Memory Management for LLM Serving (vLLM/PagedAttention) (Kwon et al., 2023) —— 重新定义 LLM serving 的论文。
4. GPTQ (Frantar et al., 2022) 和 AWQ (Lin et al., 2023) —— 基础性的 quantization 方法。
5. Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023) —— 对 speculative decoding 的数学严谨论述。
6. Medusa (Cai et al., 2024) —— 不使用 draft model 的 multi-head speculative decoding。
7. EAGLE-2 (Li et al., 2024) —— 使用 dynamic trees 的 state-of-the-art speculative decoding。
8. DeepSeek-V2 (DeepSeek, 2024) —— 用于高效推理的 MLA 和 MoE 创新。
9. Mamba (Gu & Dao, 2023) —— 作为 attention 高效替代方案的 state space models。
10. Sarathi-Serve (Agrawal et al., 2024) —— 用于延迟公平性的 chunked prefill。
11. Scaling LLM Test-Time Compute Optimally (Snell et al., 2024) —— 对 inference-time compute scaling 的形式化论述。
12. DeepSeek-R1 (DeepSeek, 2025) —— 通过 RL 训练、采用 GRPO 的推理能力。
13. Multi-Token Prediction (Gloeckle et al., 2024) —— 预测多个 tokens 在训练和推理中的收益。
14. Mixture of Depths (Raposo et al., 2024) —— 自适应的逐层计算。
15. StreamingLLM (Xiao et al., 2023) —— attention sinks 与无限长度推理。
