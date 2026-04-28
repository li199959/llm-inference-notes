# 第 13 章：可观测性与生产工程

_来源页码：PDF 第 46-49 页_

## 13.1 生产推理栈

将 LLM 部署到生产环境，并不意味着模型通过离线基准测试后就算完成。生产环境要求具备可观测性、可靠性、成本管理和持续改进能力。生产推理栈会在 serving engine 之上增加若干层：

- Logging and Tracing：记录每个请求的 TTFT、TPOT、token 数量、模型版本和错误状态。
- Metrics and Alerting：通过仪表盘跟踪 p50/p95/p99 延迟、吞吐量、GPU 利用率和错误率。
- A/B Testing Infrastructure：在不同模型版本之间拆分流量，以便安全发布。
- Capacity Planning：预测查询量，并据此预先配置硬件。
- Cost Attribution：按团队或功能拆分 token 成本。

## 13.2 关键指标详解

**TTFT (Time to First Token)**：从收到请求到第一个生成 token 返回给用户所经历的时长。TTFT 主要由 prefill 计算主导。用户会把 TTFT 感知为“思考时间”：即使后续 token 可以快速流式输出，TTFT > 1s 也会让应用显得缓慢。

**TPOT (Time Per Output Token)**：流式输出阶段相邻输出 token 之间的平均时间。人类阅读速度约为 250 words/minute（约 5 tokens/s），因此 TPOT < 200 ms 通常会感觉流畅。TPOT 主要由 decode step 延迟和 GPU 内存带宽主导。

**Goodput**：同时满足所有 SLO 目标的请求占比（同时满足 TTFT SLO 和 TPOT SLO）。如果两者相互独立，一个系统有 95% 的 TTFT SLO 达标率和 95% 的 TPOT SLO 达标率，则 goodput 为 0.95^2 ≈ 90%。Goodput 比单个指标的达标率更有信息量。

**Model FLOP Utilization (MFU)**：理论峰值 FLOP/s 中，被有效用于模型计算的比例。一个优化良好的系统在 prefill 阶段可达到 30-50% MFU，而在 decode 阶段要低得多（因为受内存带宽瓶颈限制）。MFU 是理解硬件效率的关键指标。

## 13.3 故障模式与调试

| 症状 | 可能原因 | 诊断方式 |
| --- | --- | --- |
| TTFT 高分位尖刺 | 长排队时间；prefill 饥饿或异常 | 队列深度监控；prefill 延迟分解 |
| TPOT 回退 | KV cache 驱逐；batch size 增大 | KV cache 命中率；decode step 延迟 |
| OOM (Out of Memory) | KV cache 溢出；请求激增 | 内存监控；PagedAttention block 统计 |
| 吞吐量断崖式下降 | tensor parallel 通信瓶颈 | GPU 利用率对比 NVLink 带宽 |
| 质量回退 | 模型版本问题；sampling 参数变化 | 日志中的模型版本标记；在线评估 |

## 13.4 成本建模

理解并最小化每百万输出 token 的成本，对于业务可行性至关重要。主要成本组成如下：

$$
\text{Cost per 1M tokens} = \frac{\text{GPU cost per hour} \times \text{hours}}{\text{total tokens generated}}
$$

降低成本的杠杆包括：

1. 最大化 GPU 利用率：对更多请求进行 batching，使用 continuous batching。
2. 减少模型计算量：量化、蒸馏、更小的模型。
3. Spot/preemptible instances：比 on-demand 便宜 60-90%，适合 batch workloads。
4. 优化硬件选择：对 memory-bound workloads 来说，较老一代 GPU（A100 vs. H100）可能更具成本效率。
5. 减少输出 token 数量：通过 instruction-tune 让模型更简洁。

## 面试准备：模拟问答

### Q13.1. 你会如何调试生产 LLM serving 系统中突然出现的 TTFT p99 尖刺？

**回答**：我的排查会遵循一个结构化漏斗：（1）**网络层**：检查 p99 TTFT 的增加是否与请求到达率上升相关。如果相关，系统可能正在排队。检查队列深度指标。（2）**Prefill 计算**：将 prefill 延迟与队列等待时间隔离开来。如果 prefill 出现尖刺，检查输入 prompt 长度分布是否发生变化（更长的 prompt 需要更长的 prefill 时间）。（3）**资源竞争**：检查 GPU 利用率和 NVLink 带宽。如果另一个 workload 被共同部署，可能正在竞争 GPU 资源。（4）**KV cache 压力**：如果 PagedAttention block 分配率上升，可能是请求吞吐量增加，导致计算竞争。（5）**模型版本或配置变更**：对比日志，识别同一时间窗口内是否有部署。（6）**硬件故障**：性能下降的 GPU 或 NVLink 链路可能导致吞吐量下降。使用 nvidia-smi 检查错误计数。我会把所有这些信号与尖刺发生的时间戳关联起来，然后向前追溯。

### Q13.2. 什么是 goodput？为什么它比原始吞吐量或单个指标的 SLO 达标率更有意义？

**回答**：吞吐量衡量每秒生成的 token 总数，但它并不能说明用户是否获得了可接受的体验。一个系统可以通过对许多长请求进行 batching 来获得高吞吐量，同时单个请求的 TTFT 却违反 SLO。单个 SLO 达标率（例如“95% 的请求满足 TTFT SLO”）更好一些，但多个 SLO 可能被独立度量，从而隐藏复合违规。Goodput 是同时满足所有 SLO 要求的请求占比：TTFT SLO AND TPOT SLO AND 其他任何已定义目标。它捕捉的是系统实际以令人满意的方式服务用户的速率。如果 TTFT 达标率为 90%，TPOT 达标率为 90%，在违规相互独立时，goodput 可能低至 81%。Goodput 是我会在调度决策中优化的指标：一个 SLO-aware scheduler 应该最大化 goodput，而不只是总吞吐量。

### Q13.3. 你会如何设计一个 A/B testing framework，用于在生产环境中比较两个 LLM 版本？

**回答**：我的设计如下：（1）**Traffic splitting**：使用基于 user ID（而不是 request ID）的 consistent hash 来确定性地拆分流量。同一用户始终命中同一个模型变体，从而减少用户偏好差异带来的混淆。典型初始拆分为 90/10，随着信心增加逐步扩大。（2）**Metrics**：分别跟踪两个变体的延迟（TTFT/TPOT）、质量信号（点赞/点踩、会话长度、追问次数）以及每请求成本。（3）**Statistical significance**：使用 sequential A/B testing（例如 SPRT）判断何时已经收集足够数据，并预先指定 Type I/II error bounds。（4）**Guardrails**：自动 rollback 触发器。如果测试变体的安全违规、错误率或延迟退化超过阈值，则自动将所有流量重定向到 control。（5）**Logging**：为每个请求和响应打上 variant ID、模型版本和完整 timing breakdown 标签，以供事后分析。（6）**Rollout stages**：shadow mode（无用户影响）-> 1% -> 10% -> 50% -> 100%。

### Q13.4. 假设一个 LLM serving 系统预期 6 个月内用户增长 10×，你会如何进行 capacity planning？

**回答**：Capacity planning 步骤如下：（1）**Baseline measurement**：当前每秒查询数（QPS）、平均输入/输出 token 长度以及 p95 延迟。（2）**Forecast QPS**：结合季节性（峰值/非峰值）预测 10× QPS 增长。（3）**Throughput modeling**：给定当前 GPU 数量和每 GPU 吞吐量，计算峰值所需的 GPU-hours。为突发性增加 30-50% headroom。（4）**Cost optimization**：在横向扩展之前，先穷尽纵向优化：我们能否应用 4-bit quantization，使每 GPU 吞吐量翻倍？是否升级到 H100，其内存带宽约为 A100 的 2×？是否为 system prompts 启用 prefix caching？（5）**Auto-scaling**：实现基于 Kubernetes 或 cloud-native 的 autoscaling，并把 GPU provisioning lead time 纳入考虑（GPU 实例初始化需要 2-10 min）。（6）**Multi-region**：规划地理分布，以降低延迟并避免单区域容量约束。预算假设：每块 H100 按 $2/hour 计算，在满利用率下可服务约 1M output tokens/hour。

### Q13.5. 你会如何度量并改进 Model FLOP Utilization (MFU)？

**回答**：度量 MFU：计算每次 forward pass 的 FLOPs 数量（对 transformer 来说：每 token 约为 6N，其中 N 是模型参数量，再乘以 tokens per second），然后除以 GPU 峰值 FLOP/s。使用 NVIDIA Nsight Compute 时，测量“SM Throughput”相对于峰值的比例。典型值：prefill 为 30-35% MFU（GEMM-bound），decode 小于 10%（memory-bound）。改进 MFU：对于 prefill：（1）使用 FlashAttention-3 提高 attention kernel 利用率；（2）profile 并 fuse layernorm/residual 操作；（3）对大 batch 使用跨多 GPU 的 tensor parallel；（4）确保 batch size 与 Tensor Core tile size 对齐（16 的倍数）。对于 decode：MFU 天然较低，此时更合适的指标是 memory bandwidth utilization，目标应是达到峰值 HBM 带宽的 >80%。改进方式包括：（1）增大 batch size，以摊销权重加载成本；（2）量化权重，减少每 token 需要加载的字节数；（3）使用 speculative decoding，在每个 step 中处理多个 token。
