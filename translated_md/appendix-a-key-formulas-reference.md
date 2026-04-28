# 附录 A：关键公式参考

_来源页码：PDF 第 53-53 页_

--- 第 53 页 ---

高效 LLM 推理 第 14 章：推理的未来

## 附录 A：关键公式参考

| 概念 | 公式 |
|---|---|
| 注意力 | $\operatorname{Attn}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ |
| KV 缓存大小 | $2 \times L \times n \times H \times d_h \times \text{bytes}$ |
| Roofline 性能 | $\operatorname{Perf} = \min(\text{Peak FLOP/s}, I \times B_{\text{mem}})$ |
| 推测解码加速 | $\mathbb{E}[\text{accepted}] = \frac{1 - p^{\gamma + 1}}{1 - p}$ |
| 流水线气泡率 | $\frac{p - 1}{m + p - 1}$ |
| 量化尺度 | $s = \frac{x_{\max} - x_{\min}}{2^b - 1}$ |
| MFU | $\frac{\text{Achieved FLOP/s}}{\text{Peak FLOP/s}}$ |
| 每 Token 成本 | $\frac{\text{GPU cost/hr}}{\text{tokens/hr}}$ |

AI 工程内幕 49
