# 推理 Benchmark / Profile 检查清单草稿

日期：2026-07-03

## 目标

建立第一份可复用的推理 Benchmark / Profile 检查清单。

当前版本优先面向单机单模型场景，用来支持 `experiments/2026-07-vllm-vs-sglang-qwen7b/`。后续再扩展到多卡、多机、MoE、PD 分离和长上下文。

## Benchmark 前必须固定的变量

### 硬件环境

- GPU 型号。
- GPU 数量。
- 显存大小。
- NVLink / PCIe 拓扑。
- CPU 型号和核心数。
- 内存大小。
- 本地盘 / 网络盘。
- 驱动版本。
- CUDA 版本。

### 软件环境

- OS。
- Python 版本。
- PyTorch 版本。
- vLLM / SGLang / TensorRT-LLM 版本。
- transformers 版本。
- flash-attn / flashinfer / triton 版本。
- NCCL 版本。
- Docker 镜像或 Conda 环境。

### 模型与精度

- 模型名称。
- 参数规模。
- Dense / MoE / MLA / 其他结构。
- dtype：fp16 / bf16 / fp8 / int8 / int4。
- max model length。
- quantization 方法。
- trust remote code 是否开启。

### 服务参数

- tensor parallel size。
- pipeline parallel size。
- max num seqs。
- max num batched tokens。
- gpu memory utilization。
- enable chunked prefill。
- kv cache dtype。
- swap / offload 配置。
- tokenizer 模式。
- 是否流式返回。

## Benchmark 负载设计

### 固定输入长度，变并发

目标：观察并发上升时吞吐、TTFT、TPOT 和显存如何变化。

建议组合：

```text
input_len: 512
output_len: 256
concurrency: 1, 4, 8, 16, 32, 64
```

重点判断：

- 吞吐是否随并发上升而提升？
- TTFT 从哪个并发开始明显恶化？
- TPOT 是否稳定？
- 显存是否接近上限？

### 固定并发，变输入长度

目标：观察 Prefill 压力和长上下文影响。

建议组合：

```text
concurrency: 16
input_len: 128, 512, 2048, 8192
output_len: 256
```

重点判断：

- 输入变长后 TTFT 增长是否线性？
- 长输入是否导致 decode TPOT 变差？
- KV Cache 是否成为并发限制？

### 混合负载

目标：模拟更接近线上流量的情况。

建议组合：

```text
input_len: random 128-4096
output_len: random 64-512
concurrency: 32
duration: 5 minutes
```

重点判断：

- P99 TTFT 是否被长 prompt 拖高？
- 短请求是否被长请求阻塞？
- batch 是否稳定，还是出现明显抖动？

### 长短混合负载

目标：验证不规则超长上下文是否污染短请求 SLO。

建议组合：

```text
short:
  input_len: 128-512
  output_len: 64-256
  ratio: 80-95%

long:
  input_len: 16K-128K
  output_len: 128-1024
  ratio: 5-20%

arrival:
  随机到达 + 长请求突刺
```

重点判断：

- 短请求 TTFT/TPOT P99 是否恶化。
- 长请求是否导致 KV block usage 接近上限。
- 并发不高时是否仍然出现 GPU idle gap 或单轮长 prefill。
- 单池、chunked prefill、分池三种方案的 SLO goodput 差异。

## 必采指标

### 吞吐

- total tokens/s。
- prompt tokens/s。
- output tokens/s。
- request/s。

解释方式：

- prompt tokens/s 更反映 Prefill 能力。
- output tokens/s 更反映 Decode 能力。
- total tokens/s 不能单独作为结论。

### 延迟

- TTFT P50 / P90 / P99。
- TPOT P50 / P90 / P99。
- end-to-end latency P50 / P90 / P99。
- SLO violation rate。

解释方式：

- TTFT 差通常先看排队、Prefill、Chunked Prefill。
- TPOT 差通常先看 Decode batch、KV Cache、memory bandwidth。
- end-to-end latency 会被输出长度影响，必须结合 output_len 看。

### SLO Goodput

每次部署优化必须定义 SLO，然后比较 goodput：

```text
goodput = 满足 SLO 的请求数或 token 数 / 时间
```

示例 SLO：

```text
TTFT P99 <= 2s
TPOT P99 <= 50ms
error rate <= 0.1%
```

优化目标：

- 不是 total tokens/s 最大。
- 而是在 SLO violation rate 可接受时，TPM / QPS / GPU 利用率最大。

### 资源

- GPU utilization。
- GPU memory used。
- HBM bandwidth。
- SM occupancy。
- CPU utilization。
- host memory。
- 网络带宽和延迟，多卡/多机时必看。

解释方式：

- GPU utilization 低不一定是 GPU 慢，可能是调度、CPU、tokenizer、网络或请求生成器没压满。
- 显存接近上限时，吞吐下降可能来自 KV Cache 容量，而不是算力。
- 多卡时，通信等待可能伪装成 GPU 利用率下降。

### 资源竞争假设

每次 Benchmark 前先写一个假设：

```text
本次实验最可能竞争的资源是：
竞争双方是：
如果竞争发生，预期症状是：
用什么指标验证：
```

常见例子：

- Prefill 和 Decode 竞争 GPU compute：TTFT 上升，TPOT 抖动。
- Decode 请求竞争 HBM bandwidth：TPOT 上升，output tokens/s 到达平台上限。
- 长上下文请求竞争 KV Cache blocks：waiting 变长，OOM 或 preemption 增加。
- API/tokenizer 与 Scheduler 竞争 CPU：GPU idle gap 增加，服务端 QPS 上不去。
- 多卡 worker 竞争通信带宽：NCCL kernel 占比高，多卡扩展不线性。

## Profile 前的检查

Profile 不要一上来抓全量长时间，先抓一个可解释的小窗口。

### 先确认场景

- 抓 Prefill 还是 Decode？
- 抓单请求还是高并发？
- 抓稳定阶段还是冷启动？
- 抓正常 case 还是异常 case？

### 推荐三类 Profile

#### 1. 单请求短 prompt

用途：

- 看基础调用链。
- 确认模型 forward 和 sampler 位置。
- 排除环境明显异常。

#### 2. 单请求长 prompt

用途：

- 看 Prefill 热点。
- 观察 Attention / MLP kernel。
- 判断长上下文开销。

#### 3. 高并发混合负载

用途：

- 看 Scheduler 和 Worker 是否持续有活干。
- 观察 batch 抖动。
- 查 TTFT / TPOT P99 变差的原因。

## Profile 证据要记录什么

### 时间线

- CPU 是否有长时间空等。
- GPU kernel 是否连续。
- 是否存在明显 gap。
- gap 发生在 tokenizer、scheduler、communication 还是 kernel 之间。

### Kernel

- Attention kernel 时间。
- MLP kernel 时间。
- Sampling 时间。
- memcpy / reshape / cast 等非计算开销。
- Kernel launch 是否过碎。

### KV Cache

- block 分配是否失败或等待。
- 显存占用是否持续上升。
- 请求结束后显存是否回收。
- 长上下文下是否触发 swap/offload。

### Communication

多卡时记录：

- AllReduce。
- AllGather。
- ReduceScatter。
- All-to-All。
- P2P send/recv。
- NCCL kernel 占比。

## 结果表格模板

| 框架 | 模型 | 并发 | 输入 | 输出 | prompt tok/s | output tok/s | TTFT P50 | TTFT P99 | TPOT P50 | TPOT P99 | 峰值显存 | 备注 |
|------|------|------|------|------|--------------|--------------|----------|----------|----------|----------|----------|------|
| vLLM | Qwen2-7B | 16 | 512 | 256 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| SGLang | Qwen2-7B | 16 | 512 | 256 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## 分析结论模板

每次 Benchmark 后，用下面的结构写结论：

```text
结论：
- 哪个框架在什么负载下更好？
- 差异主要体现在 TTFT、TPOT 还是吞吐？
- 差异属于哪一层：Scheduler / KV Cache / Kernel / Communication / API overhead？

证据：
- 指标表里哪几列支持这个判断？
- Profile 中哪段时间线或哪个 kernel 支持这个判断？
- 是否有显存、GPU 利用率或网络证据？

局限：
- 硬件是否单一？
- 模型是否单一？
- 请求分布是否过于理想？
- 是否只测了冷启动或短时间窗口？

下一步：
- 改哪个参数？
- 补哪组负载？
- 抓哪类 Profile？
- 是否需要进入特定模型自定义实现：model adapter、custom op、attention backend、KV Cache、scheduler？
```

## 进入自定义实现前的门槛

只有满足以下条件，才考虑改 vLLM 代码：

- 已经有 baseline。
- 已经有 Profile 证据。
- 已经确认瓶颈层级。
- 参数、并行拓扑、分池、路由无法解决。
- 自定义实现有明确回退开关。

候选实现方向：

- 模型结构或 weight loader 不匹配：model adapter。
- Attention 热点：attention backend / custom kernel。
- 小 op 过碎：CustomOp / fused op。
- KV Cache 成为硬约束：KV layout / cache policy / prefix cache / KV router。
- 长短混合 P99 差：scheduler / routing / admission control。

## 当前本周最小可执行版本

本周先完成最小闭环：

1. 固定 Qwen2-7B。
2. 固定单卡。
3. 固定 vLLM vs SGLang。
4. 跑一组固定输入长度、变并发。
5. 记录 TTFT、TPOT、tokens/s、峰值显存。
6. 抓一次长 prompt 的 Profile。
7. 写一段“差异属于哪一层”的分析。

先做到可复现，再追求覆盖完整。
