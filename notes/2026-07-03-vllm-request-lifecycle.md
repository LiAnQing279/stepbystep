# vLLM 请求生命周期高层链路

日期：2026-07-03

## 目标

选择 vLLM 作为本周源码主线，先画出一次请求从进入系统到返回 token 的高层路径。

本笔记不追求逐行源码阅读，而是先建立一个可以继续追源码、做 Benchmark、看 Profile 的骨架。

后续读每个阶段时，都要额外标记：

- 这个阶段管理什么资源？
- 它会和哪些进程/组件竞争资源？
- 竞争失败会表现成什么指标异常？
- Profile 中应该用什么证据确认？

## 一句话链路

```text
HTTP request
  -> API Server
  -> AsyncLLMEngine
  -> add_request
  -> scheduler.schedule
  -> KV Cache block 分配
  -> worker.execute_model
  -> model forward
  -> sampler
  -> stream / return token
  -> 请求完成后释放资源
```

## 关键阶段

### 1. 请求进入 API Server

用户通过 OpenAI-compatible API 或内部调用提交请求。

这一层主要处理：

- 参数解析。
- tokenizer 相关准备。
- sampling 参数。
- 流式或非流式返回模式。
- 请求 ID 和上下文对象。

当前要关注的问题：

- 请求在进入引擎前已经携带哪些信息？
- input tokens、max tokens、temperature、stop 条件在哪里进入系统？
- 流式请求和普通请求在引擎层是否走同一条主链路？

### 2. 请求进入 AsyncLLMEngine

`AsyncLLMEngine` 负责把外部请求转换成引擎内部的请求对象，并交给核心执行循环。

这一层主要处理：

- 接收新请求。
- 放入等待队列。
- 推动后台 engine loop。
- 将生成结果异步返回给上层。

当前判断：

- API Server 是入口，AsyncLLMEngine 是异步服务化边界。
- 真正影响吞吐和延迟的核心逻辑在后面的调度和执行。

### 3. add_request：请求进入等待队列

`add_request` 的核心作用是把一个新请求注册到引擎状态中。

它通常要关联：

- request id。
- prompt token ids。
- sampling params。
- arrival time。
- 资源需求。
- 当前生成状态。

当前要关注的问题：

- 请求进入队列后是按什么策略等待调度？
- 是否有优先级、超时、取消或抢占机制？
- prompt 长度是否会影响排队和调度策略？

### 4. scheduler.schedule：决定本轮执行哪些请求

Scheduler 是本周和下周最值得深挖的部分。

这一层决定：

- 哪些等待请求进入本轮 batch。
- 哪些运行中请求继续 decode。
- prefill 和 decode 如何混排。
- KV Cache block 是否足够。
- 请求是否需要等待、抢占或释放。

调度的核心矛盾：

- 想提升吞吐，就要让 GPU batch 尽量饱满。
- 想降低 TTFT，就不能让新请求在队列里等太久。
- 想降低 TPOT，就不能让 decode 被长 prefill 反复打断。
- 想提升并发，就要更精细地管理 KV Cache。

### 5. KV Cache block 分配

请求被调度进入执行前，需要为它的 KV Cache 分配 block。

这一层要回答：

- prompt 的 KV Cache 如何写入？
- decode 每生成一个 token 后如何追加 KV？
- block 不够时如何处理？
- 请求结束后如何释放 block？

当前判断：

- KV Cache 是 vLLM 高吞吐的核心资源之一。
- Scheduler 的决策不能只看请求数量，还必须看 token 数和 KV Cache 余量。
- 长上下文和高并发场景下，KV Cache 往往比计算更早成为限制。

### 6. worker.execute_model：执行模型前向

调度结果会被发送到 Worker，Worker 负责真正调用模型执行。

这一层包括：

- 准备输入张量。
- 调用模型 forward。
- 调用 Attention / MLP / Sampling 相关 kernel。
- 多卡时处理 TP/PP 等并行通信。

当前要关注的问题：

- Prefill 和 Decode 的输入 shape 有什么差别？
- 哪些 kernel 是热点？
- GPU 利用率低时，是调度没喂饱，还是 kernel/通信拖慢？

### 7. sampler：从 logits 生成 token

模型 forward 输出 logits 后，采样器根据 sampling params 生成下一个 token。

这一层处理：

- temperature。
- top-p / top-k。
- stop token。
- EOS。
- 是否继续生成。

当前判断：

- 采样通常不是大头，但会影响请求是否完成。
- 对结构化输出、约束解码、投机解码等方向，采样层会变得更重要。

### 8. 返回 token 并更新请求状态

生成 token 后，系统会：

- 把 token 返回给上层。
- 更新请求的生成状态。
- 判断请求是否完成。
- 未完成则进入下一轮 decode。
- 完成则释放 KV Cache 和内部状态。

流式返回时，用户会逐 token 接收；非流式返回时，上层通常等完整结果。

## Prefill 与 Decode 的区别

### Prefill

特点：

- 一次性处理 prompt。
- 输入 token 多。
- 矩阵计算更重。
- 更偏 compute-bound。
- 直接影响 TTFT。

观察指标：

- TTFT。
- prompt tokens/s。
- GPU utilization。
- long prompt 下的排队延迟。

### Decode

特点：

- 每轮通常生成一个 token。
- 每步都要读取已有 KV Cache。
- 更偏 memory-bound。
- 直接影响 TPOT 和输出吞吐。

观察指标：

- TPOT。
- output tokens/s。
- KV Cache 占用。
- batch 内请求完成和退出是否顺畅。

## 请求生命周期草图

```text
┌─────────────┐
│ HTTP Request│
└──────┬──────┘
       │
       v
┌─────────────┐
│ API Server  │  参数解析 / tokenization / stream setup
└──────┬──────┘
       │
       v
┌────────────────┐
│ AsyncLLMEngine │  add_request / async result
└──────┬─────────┘
       │
       v
┌────────────────┐
│ Waiting Queue  │  新请求等待进入 batch
└──────┬─────────┘
       │
       v
┌────────────────┐
│ Scheduler      │  prefill/decode 混排 / KV Cache 预算
└──────┬─────────┘
       │
       v
┌────────────────┐
│ KV Cache Blocks│  分配 / 追加 / 释放
└──────┬─────────┘
       │
       v
┌────────────────┐
│ Worker Execute │  model forward / kernels / communication
└──────┬─────────┘
       │
       v
┌────────────────┐
│ Sampler        │  logits -> token
└──────┬─────────┘
       │
       v
┌────────────────┐
│ Return Token   │  stream / finish / continue decode
└────────────────┘
```

## 下一步源码阅读问题

下一步读 vLLM Scheduler 时，优先带着这些问题看：

1. Scheduler 内部如何区分 waiting、running、swapped、finished 请求？
2. Prefill 请求和 decode 请求发生冲突时，谁优先？
3. KV Cache block 不够时，请求是等待、抢占还是换出？
4. Chunked Prefill 如何改变 TTFT 和 TPOT 的权衡？
5. 调度结果如何传给 Worker？
6. Worker 执行结束后，Scheduler 如何更新请求状态？
7. Scheduler 管理哪些资源预算：token budget、sequence budget、KV blocks、单轮执行时间、SLO？
8. Prefill、Decode、长 prompt、短 prompt 分别会竞争什么资源？
9. 资源竞争会导致 waiting、preemption、swap、OOM、TPOT 抖动还是 GPU idle gap？

## 本周结论

vLLM 请求生命周期的核心不是 API Server，而是：

```text
Scheduler + KV Cache + Worker Execute
```

后续 Benchmark 和 Profile 也应该围绕这三块观察：

- Scheduler 是否让 GPU 吃满。
- KV Cache 是否限制并发。
- Worker 执行时间主要花在 kernel、通信还是等待。
