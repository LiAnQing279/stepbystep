# vLLM Scheduler：进程、KV Block、通信和瓶颈问题清单

日期：2026-07-03

## 范围

本笔记面向 vLLM V1。重点不是先逐行读源码，而是建立排查问题时要看的系统切面：

- 有哪些进程？
- 哪些地方是线程、事件循环或 busy loop？
- GPU KV block 如何被分配和释放？
- API Server、Engine Core、GPU Worker 之间如何通信？
- 常见瓶颈会出现在哪里？

## 1. vLLM V1 里有哪些主要进程

### API Server Process

职责：

- 接收 HTTP / OpenAI-compatible 请求。
- 做输入处理，例如 tokenization、多模态数据加载。
- 把请求发给 Engine Core。
- 接收输出并流式返回给客户端。

竞争资源：

- CPU core。
- event loop。
- tokenizer。
- HTTP 连接数。
- ZMQ 发送/接收队列。

典型问题：

- CPU 打满但 GPU 不忙。
- tokenization 慢导致 TTFT 高。
- API server 到 Engine Core 的队列堆积。

### Engine Core Process

职责：

- 运行 Scheduler。
- 管理 KV Cache。
- 协调 GPU Worker 执行。
- 在一个持续循环中不断 schedule 请求并 dispatch work。

竞争资源：

- Scheduler token budget。
- running request slots。
- KV Cache blocks。
- CPU 调度循环时间。
- 和 GPU Worker 的通信通道。

典型问题：

- Scheduler loop 慢，GPU 有 idle gap。
- waiting queue 长。
- KV block 不足导致 preemption / recompute。
- prefill / decode 混排策略导致 TTFT 或 TPOT 变差。

### GPU Worker Process

职责：

- 在 GPU 上执行模型 forward。
- 管理 GPU runner、CUDA stream、CUDA graph、kernel launch。
- 多卡时参与 TP / PP / EP / CP 等通信。

竞争资源：

- GPU SM / Tensor Core。
- HBM capacity。
- HBM bandwidth。
- CUDA stream。
- NCCL / P2P 通信资源。

典型问题：

- GPU busy 但 tokens/s 不高。
- Decode 阶段 HBM bandwidth 饱和。
- 多卡通信占比高。
- PP stage 或 TP rank 等待。

### DP Coordinator Process

使用 Data Parallel 时才会出现。

职责：

- 协调多个 DP rank / engine core。
- 处理 DP 组之间的请求分配或协调。

典型问题：

- DP rank 负载不均。
- KV Cache / Prefix Cache 命中被路由打散。
- 某些 DP 副本 queue 很长，另一些空闲。

## 2. 哪些地方是线程或循环

先粗略这样理解：

```text
API Server:
  async event loop + HTTP handling + tokenizer/detokenizer 相关执行

Engine Core:
  busy loop / scheduler loop
  每轮：收请求 -> schedule -> 发给 worker -> 收输出 -> 更新状态

GPU Worker:
  worker process 内部执行 model runner
  CUDA stream / kernel launch / NCCL communication
```

读源码时要特别标记：

- API Server 是否阻塞在输入处理。
- Engine Core loop 是否被 schedule 或通信阻塞。
- Worker 是否持续有 batch 可跑。
- GPU kernel 之间是否有空洞。

## 3. GPU KV Block 如何分配

### 核心概念

vLLM 把 KV Cache 切成 block，不给每个请求预留完整最大长度显存。

粗略流程：

```text
请求进入 waiting
  -> Scheduler 决定本轮给它多少 token budget
  -> KVCacheManager 检查已有 computed/cached blocks
  -> 为需要新计算的 token 分配新 KV blocks
  -> Scheduler 把 block ids / metadata 传给 Worker
  -> Worker forward 时把新 KV 写入对应 block
  -> 请求结束或被抢占时释放 block
```

### 分配时要看的约束

- block_size。
- max_num_batched_tokens。
- max_num_seqs / max_num_running_reqs。
- gpu_memory_utilization。
- max_model_len。
- prefix cache 是否命中。
- speculative decoding 是否需要 lookahead slots。
- DCP / CP 是否改变 KV 切分方式。

### block 不够时会怎样

常见结果：

- 请求继续 waiting。
- running 请求被 preempt。
- V1 默认倾向 recompute，而不是 swap。
- 被 preempt 的请求之后需要重新计算被释放的 KV。

排查信号：

- preemption count 增加。
- TTFT / end-to-end latency 上升。
- GPU 显存接近上限。
- waiting queue 变长。
- KV Cache usage 高。

## 4. 通信过程是怎么样的

### API Server 和 Engine Core

通信方式：

- V1 架构中 API Server 和 Engine Core 通过 ZMQ 通信。
- 多 API server 和多 engine core 可以形成 many-to-many 连接。

排查重点：

- API server 是否把请求及时送到 engine core。
- ZMQ 队列是否堆积。
- 多 DP 场景下请求是否路由均衡。

### Engine Core 和 GPU Worker

通信内容：

- scheduled request ids。
- 每个请求本轮要执行的 token 数。
- KV block ids / block table / slot mapping。
- sampling 参数。
- finished / output token / 状态更新。

排查重点：

- Engine Core 是否及时 dispatch。
- Worker 是否等调度结果。
- Worker 输出返回后 Scheduler 是否及时更新状态。

### GPU Worker 之间

取决于并行方式：

- TP：每层可能有 AllReduce / AllGather / ReduceScatter。
- PP：stage 之间 P2P 传激活。
- EP：MoE expert dispatch / combine，通常是 All-to-All。
- CP / DCP：围绕 context / KV 的跨卡通信。
- PD：Prefill worker 到 Decode worker 的 KV transfer。

排查重点：

- NCCL kernel 占比。
- NVLink / PCIe / IB 带宽。
- TPOT 是否因为通信变差。
- 某些 rank / stage 是否等待其他 rank。

## 5. 哪里最可能是瓶颈

### API / Tokenizer 瓶颈

症状：

- GPU 利用率低。
- TTFT 高。
- 服务端 QPS 上不去。

优先看：

- API Server CPU。
- tokenizer latency。
- event loop lag。
- client 是否压满。

### Scheduler 瓶颈

症状：

- GPU kernel 间有 gap。
- waiting queue 长。
- batch token 数不稳定。
- P99 明显差。

优先看：

- scheduler step latency。
- scheduled token count。
- prefill/decode mix。
- max_num_batched_tokens / max_num_seqs。

### KV Cache 瓶颈

症状：

- preemption 增加。
- 长上下文/高并发下 OOM。
- TTFT 和 end-to-end latency 上升。
- 新请求进不去 running。

优先看：

- KV block usage。
- gpu_memory_utilization。
- max_model_len。
- input/output length 分布。
- prefix cache 命中率。

### GPU Compute 瓶颈

症状：

- prefill TTFT 随输入长度上升明显。
- GPU SM 高。
- prompt tokens/s 到达上限。

优先看：

- Attention / MLP kernel 时间。
- batch shape。
- chunked prefill 是否开启。

### HBM Bandwidth 瓶颈

症状：

- Decode TPOT 高。
- output tokens/s 到平台上限。
- GPU busy 但不是纯 compute-bound。

优先看：

- HBM bandwidth。
- context length。
- decode batch size。
- KV Cache layout / dtype。

### 通信瓶颈

症状：

- 多卡扩展不线性。
- TPOT 抖动。
- GPU 等通信。
- 跨节点后性能明显变差。

优先看：

- NCCL time。
- AllReduce / All-to-All / P2P 时间。
- 拓扑：NVLink、PCIe、IB/RDMA。
- TP / PP / EP / CP 组合是否合适。

## 6. 读源码时的入口问题

### Scheduler

- `schedule()` 每轮输入是什么？
- `waiting` / `running` / `finished` 如何流转？
- token budget 如何分给 prefill 和 decode？
- 什么条件下调用 KVCacheManager 分配 block？
- block 不够时如何 preempt？

### KVCacheManager

- `allocate_slots()` 如何决定需要几个 block？
- computed blocks / cached blocks 如何识别？
- free block pool 如何维护？
- 请求结束后如何 free？
- prefix cache 命中后如何减少新分配？

### Worker / Model Runner

- block ids 如何变成 GPU 侧 block table？
- forward 时 KV 写入哪个位置？
- decode 每步如何追加 KV？
- sampling 输出如何回传给 Engine Core？

### Distributed

- TP/PP/EP/CP 各自在哪里建立通信组？
- worker 间通信发生在 forward 的哪一段？
- Scheduler 是否知道通信成本，还是只知道并行配置？

## 7. 一句话地图

```text
API Server 负责接请求和输入处理；
Engine Core 跑 Scheduler、管 KV Cache、发任务；
GPU Worker 跑模型和通信；
KV block 是 Scheduler 决策的硬资源；
瓶颈通常出在 CPU/tokenizer、scheduler loop、KV block、GPU compute、HBM bandwidth、NCCL/拓扑。
```
