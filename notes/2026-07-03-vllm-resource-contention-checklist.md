# vLLM 资源竞争排查清单

日期：2026-07-03

## 目标

以后阅读 vLLM、做 Benchmark、排查线上问题时，不只看代码调用链，还要同时看资源竞争链。

核心问题：

```text
哪些进程/线程/组件在竞争哪些资源？
这种竞争会表现成什么症状？
应该用哪些指标和 Profile 证据确认？
```

## 资源竞争总图

```text
Client / Benchmark Generator
  -> API Server / Tokenizer
  -> Engine / Scheduler
  -> KV Cache Manager
  -> Worker / Model Executor
  -> GPU Kernels
  -> NCCL / P2P Communication
  -> Streaming Response
```

每一层都可能成为瓶颈。GPU 利用率低不一定说明模型慢；TTFT 高也不一定说明 prefill 慢。很多问题本质是资源竞争导致某一段等待。

## 1. CPU 竞争

### 竞争方

- API Server。
- tokenizer / detokenizer。
- request validation。
- async event loop。
- Scheduler。
- benchmark client 或压测工具。
- logging / metrics / tracing。

### 竞争资源

- CPU core。
- Python GIL。
- event loop。
- thread pool。
- process pool。
- host memory bandwidth。

### 可能症状

- GPU utilization 不高，但 TTFT / end-to-end latency 很高。
- 请求进入系统慢，排队时间异常。
- tokenization 成为长 prompt 场景瓶颈。
- 高并发时 API Server CPU 打满。
- Profile 中 GPU kernel 之间有明显空洞。
- client 侧压不满服务端，导致吞吐看起来偏低。

### 排查指标

- CPU utilization per process。
- API Server QPS。
- tokenizer latency。
- scheduler loop latency。
- event loop lag。
- GPU idle gap。
- client 端发送速率和服务端接收速率。

### 判断问题

如果 GPU 空着，但请求还在等待，优先怀疑：

- API/tokenizer 太慢。
- Scheduler loop 没有及时推进。
- client 压测工具没有压满。
- Python/CPU 侧存在阻塞。

## 2. GPU 计算资源竞争

### 竞争方

- Prefill batch。
- Decode batch。
- Attention kernel。
- MLP kernel。
- Sampling kernel。
- CUDA graph / eager execution。
- 其他同机 GPU 任务。

### 竞争资源

- SM。
- Tensor Core。
- register。
- shared memory。
- kernel launch slot。

### 可能症状

- GPU utilization 高，但 tokens/s 不理想。
- TTFT 随输入长度明显上升。
- 大 prefill 抢占 decode，导致 TPOT 抖动。
- Profile 里 kernel 时间很长，但中间 gap 不明显。
- 同机其他任务影响推理延迟。

### 排查指标

- SM utilization。
- kernel duration。
- prompt tokens/s。
- output tokens/s。
- TTFT P50/P99。
- TPOT P50/P99。
- 是否存在混部任务。

### 判断问题

如果 GPU 很忙且 prefill 请求多，优先看：

- 是否需要限制 `max_num_batched_tokens`。
- 是否需要 Chunked Prefill。
- Prefill 和 Decode 是否互相干扰。
- batch 构造是否过大导致单轮执行时间太长。

## 3. GPU 显存与 KV Cache 竞争

### 竞争方

- 模型权重。
- KV Cache blocks。
- activation / temporary buffers。
- CUDA graph memory。
- NCCL buffers。
- framework workspace。
- prefix cache / block table metadata。

### 竞争资源

- HBM capacity。
- HBM bandwidth。
- KV Cache block pool。
- contiguous memory / allocator fragments。

### 可能症状

- 高并发或长上下文下 OOM。
- 请求无法进入 running，长时间 waiting。
- TTFT 上升，因为新请求拿不到 KV block。
- TPOT 变差，因为 decode 读取 KV Cache 压力变大。
- 显存接近上限但 GPU 计算利用率不高。
- 请求结束后显存未明显回落，怀疑缓存保留、碎片或泄漏。

### 排查指标

- GPU memory used。
- KV Cache block usage。
- active sequences。
- total tokens in batch。
- max model length。
- request input/output length 分布。
- OOM 日志。
- block allocation failure / preemption / swap 记录。

### 判断问题

如果显存接近上限，同时吞吐或 TTFT 恶化，优先怀疑：

- KV Cache 成为硬约束。
- `max_num_seqs` 或 `max_num_batched_tokens` 太激进。
- 长上下文请求挤占短请求。
- Prefix Cache / KV Cache 策略导致缓存保留过多。
- GPU memory utilization 配置不合理。

## 4. HBM 带宽竞争

### 竞争方

- Decode 阶段读取权重。
- Decode 阶段读取 KV Cache。
- Attention kernel。
- Sampling / logits 处理。
- tensor copy / reshape / cast。

### 竞争资源

- HBM bandwidth。
- L2 cache。
- memory transaction。

### 可能症状

- Decode TPOT 变差。
- output tokens/s 到达平台上限后不再增长。
- GPU utilization 看起来高，但 SM 计算并不满。
- 长上下文或大 batch decode 时性能下降明显。

### 排查指标

- HBM bandwidth。
- TPOT。
- output tokens/s。
- decode batch size。
- context length。
- memory-bound kernel 占比。

### 判断问题

如果 prefill 正常但 decode 变慢，优先怀疑：

- KV Cache 读取带宽成为瓶颈。
- batch 内上下文长度过长。
- decode 阶段 memory-bound。
- KV Cache dtype / layout / block size 影响读取效率。

## 5. 通信资源竞争

### 竞争方

- TP AllReduce / AllGather。
- PP P2P activation transfer。
- EP All-to-All。
- PD 模式下 KV Cache transfer。
- NCCL 通信。
- 同机或跨机其他网络流量。

### 竞争资源

- NVLink。
- PCIe。
- IB / RoCE。
- NIC。
- NCCL stream。
- GPU copy engine。

### 可能症状

- 多卡性能扩展不线性。
- TP 从单机扩到跨机后延迟暴涨。
- GPU utilization 低但通信 kernel 占比高。
- Decode TPOT 抖动。
- PD 分离时 TTFT 或 handoff latency 高。
- 拓扑不同的机器表现差异很大。

### 排查指标

- NCCL kernel time。
- AllReduce / AllGather / All-to-All 时间。
- NVLink / PCIe / IB 带宽。
- p2p latency。
- KV transfer time。
- GPU idle gap before/after communication。
- 拓扑信息。

### 判断问题

如果多卡后性能变差，优先怀疑：

- TP 通信成本超过计算收益。
- 跨节点通信拓扑不合适。
- NCCL 配置或网络拥塞。
- PD KV 传输路径不亲和。
- EP All-to-All 负载不均衡。

## 6. Scheduler 资源竞争

### 竞争方

- 新 prefill 请求。
- running decode 请求。
- 长 prompt 请求。
- 短 prompt 请求。
- 接近完成的请求。
- prefix cache 命中请求。
- 高优先级或低优先级请求。

### 竞争资源

- batch token budget。
- max_num_seqs。
- KV Cache blocks。
- 单轮执行时间。
- decode latency budget。
- SLO budget。

### 可能症状

- 长 prompt 把短请求 TTFT 拖高。
- prefill 过多导致 decode TPOT 抖动。
- decode 请求占住 KV Cache，导致新请求进不来。
- P99 latency 明显差于 P50。
- GPU 有时满、有时空，吞吐抖动。
- 缓存命中请求被排到忙 worker，反而破坏 SLO。

### 排查指标

- waiting queue length。
- running sequence count。
- scheduled token count per step。
- prefill/decode token mix。
- scheduler step latency。
- preemption / eviction / swap 次数。
- TTFT/TPOT P50/P90/P99。
- cache hit rate 与 queueing delay 的关系。

### 判断问题

如果延迟长尾明显，优先怀疑：

- Scheduler 在吞吐和延迟之间取舍不合适。
- long prompt 与 decode 混排策略不合适。
- KV block 预算限制了新请求。
- 路由只看 cache hit，不看 worker queue 和 SLO。

## 7. 存储和模型加载竞争

### 竞争方

- 模型权重加载。
- tokenizer / config 读取。
- checkpoint shard 读取。
- container image / shared filesystem。
- KV Cache offload / reload。

### 竞争资源

- 本地 SSD。
- 网络盘。
- object storage。
- page cache。
- host memory。

### 可能症状

- 冷启动慢。
- 多实例同时拉起时启动时间暴涨。
- offload 场景下 TTFT 或 TPOT 抖动。
- worker 重启后恢复慢。

### 排查指标

- model load time。
- disk read throughput。
- page cache hit。
- offload read/write latency。
- startup timeline。

### 判断问题

如果问题主要发生在启动、扩容、恢复或 offload 时，优先怀疑：

- 存储带宽竞争。
- 网络盘延迟。
- 多实例同时加载模型。
- KV Cache offload 路径太慢。

## 排查时的固定问法

遇到 vLLM 性能或稳定性问题时，先问：

1. 哪个指标坏了：TTFT、TPOT、throughput、OOM、GPU utilization、P99？
2. 坏的是 prefill、decode、调度、通信、tokenizer 还是 client？
3. 哪些进程/组件在竞争同一个资源？
4. 竞争资源是 CPU、GPU compute、HBM capacity、HBM bandwidth、KV blocks、network、storage 还是 SLO budget？
5. Profile 里有没有等待、gap、长 kernel、通信 kernel 或 allocator 证据？
6. 限制并发、缩短输入、关闭 prefix cache、改 batch token budget 后，症状是否变化？

## 和指标的快速映射

| 症状 | 优先怀疑 |
|------|----------|
| TTFT 高 | 排队、Prefill、tokenizer、KV block 不足、长 prompt 混排 |
| TPOT 高 | Decode batch、HBM bandwidth、KV Cache 读取、通信 |
| GPU 利用率低 | client 压不满、CPU/tokenizer、Scheduler gap、通信等待 |
| GPU 利用率高但吞吐低 | kernel 低效、memory-bound、batch shape 不合适、通信占比高 |
| OOM | 模型权重、KV Cache、max_num_seqs、max_model_len、fragmentation |
| P99 很差 | 长请求、调度策略、队列堆积、cache routing 失衡、网络抖动 |
| 多卡扩展差 | TP/PP/EP 通信、拓扑、NCCL、跨节点带宽 |
| 冷启动慢 | 模型加载、网络盘、镜像、checkpoint shard、并发拉起 |

## 读源码时要标记的位置

后续读 vLLM 源码时，每看到一个模块，都标记它管理或竞争的资源：

- API Server：CPU、event loop、tokenizer、连接数。
- Scheduler：token budget、sequence budget、KV blocks、SLO budget。
- KV Cache Manager：block pool、HBM capacity、cache hit。
- Worker：GPU compute、HBM bandwidth、CUDA stream。
- Distributed Executor：GPU 拓扑、NCCL、P2P。
- Sampler：CPU/GPU sampling、logits memory。
- Metrics：观测指标是否足够反推出资源竞争。
