# 不规则超长上下文与短请求混合部署 Playbook

日期：2026-07-03

## 定位

这份文档专门处理一种复杂但常见的在线推理场景：

```text
不规则超长上下文请求
  + 短请求
  + 并发不高
  + 但延迟很大 / P99 很差 / GPU 利用率不稳定
```

核心目标不是单纯提高吞吐，而是在 SLO 下保护短请求延迟，同时让长上下文请求尽可能有效利用 GPU。

## 为什么这种场景难

长短请求混合会同时污染多个资源：

- Prefill compute：超长 prompt 会占用一轮很长的计算时间。
- KV Cache capacity：长上下文占用大量 KV blocks。
- Decode HBM bandwidth：长上下文 decode 每步读取更多 KV。
- Scheduler token budget：长请求消耗大量 `max_num_batched_tokens`。
- Queueing：短请求被长请求排在前面时，TTFT 被拖高。
- GPU 利用率：并发不高时，长请求单独跑不一定能填满 batch，短请求又容易被阻塞。

所以低并发也可能慢。慢不一定来自“请求太多”，也可能来自“单个请求太大、shape 太差、调度粒度太粗”。

## 常见现象

### 1. 并发不高但 TTFT 很大

可能原因：

- 一个超长 prompt prefill 占住本轮执行。
- Scheduler 把长 prefill 和短请求放在同一资源池里。
- tokenization / preprocessing 对长输入很慢。
- KV block 不够，短请求进入不了 running。

### 2. 短请求 P99 被长请求拖高

可能原因：

- FIFO 或近似 FIFO 下，短请求排在长请求后面。
- 没有按照 input length 分池。
- Chunked Prefill 没有开启或粒度不合适。
- 长请求占用 KV Cache，短请求无法获得 block。

### 3. GPU 利用率不稳定

可能原因：

- 低并发下 batch 不够饱满。
- 长 prompt prefill 时 GPU 忙，decode 阶段又 memory-bound。
- Scheduler 每轮 batch shape 波动大。
- CPU/tokenizer 对长请求处理导致 GPU idle gap。

### 4. TPOT 也变差

可能原因：

- 长上下文 decode 读取大量 KV Cache，HBM bandwidth 成为瓶颈。
- batch 内 context length 差异大，kernel shape 不稳定。
- CP/DCP 或多卡通信引入额外开销。

## 背后的原理

### Prefill 和 Decode 的资源类型不同

Prefill：

- 输入 token 多。
- 矩阵计算重。
- 更偏 compute-bound。
- 影响 TTFT。

Decode：

- 每步通常生成一个 token。
- 需要反复读取权重和已有 KV Cache。
- 更偏 memory-bound。
- 影响 TPOT。

长请求会在 prefill 阶段拖长单轮执行时间，在 decode 阶段放大 KV Cache 读取压力。

### KV Cache 是硬资源

长上下文请求占用 KV blocks 的时间很长。

结果：

- 新请求可能拿不到 block。
- running 请求可能被 preempt。
- 短请求虽然很小，但仍然被 KV Cache 容量限制。
- 显存接近上限时，GPU 可能并没有被 compute 打满。

### 低并发不等于低延迟

低并发下可能出现：

- batch 太小，GPU 不满。
- 单个超长 prefill 独占一轮执行。
- decode batch 小但 context 长，memory-bound 严重。
- 没有足够其他请求填补 GPU 空泡。

所以低并发慢的根因通常是：

```text
batch shape 不好 + 调度粒度粗 + KV/HBM 资源被长请求占住
```

## 部署设计原则

### 原则 1：不要让短请求和超长请求无边界混跑

最简单也最有效的策略是分池：

```text
short pool:
  input <= 2K 或 4K
  目标：低 TTFT / 低 P99

medium pool:
  input 4K-16K
  目标：平衡 TTFT 和吞吐

long pool:
  input >= 16K / 32K
  目标：保护系统，接受更宽松 SLO
```

分池方式：

- 不同 vLLM 实例。
- 不同 deployment。
- 不同 queue。
- 不同 routing policy。
- 不同参数配置。

### 原则 2：长请求要有独立 SLO

不要让长请求和短请求共享同一个 P99 目标。

示例：

```text
short request:
  TTFT P99 <= 1s
  TPOT P99 <= 50ms

long request:
  TTFT P99 <= 10s
  TPOT P99 <= 80ms
```

如果业务上必须共享 SLO，就必须对长请求做强限制：

- 限制最大输入长度。
- 限制并发。
- 限制输出长度。
- 限制进入高优先级池。

### 原则 3：长请求优先用调度隔离，而不是只靠加 GPU

加 GPU 可能只是放大成本，不一定解决短请求 P99。

优先考虑：

- 分池。
- priority scheduling。
- chunked prefill。
- max_num_batched_tokens 限制。
- max_num_seqs 限制。
- KV Cache 预算隔离。
- 长短请求不同路由。

### 原则 4：SLO 下看 goodput，不看平均吞吐

长请求混入后，平均 tokens/s 可能还不错，但短请求 P99 已经坏了。

所以要看：

```text
short goodput under short SLO
long goodput under long SLO
overall GPU cost
```

## 具体部署方案

### 方案 A：单池 + Chunked Prefill

适用：

- 长请求比例低。
- SLO 不极端严格。
- 资源有限，不想维护多套 deployment。

策略：

- 开启 Chunked Prefill。
- 控制 `max_num_batched_tokens`。
- 限制 `max_num_seqs`。
- 设置最大输入长度。
- 观察短请求 TTFT P99。

优点：

- 简单。
- 改动小。
- 可以缓解长 prefill 独占 GPU。

风险：

- 长请求仍然占 KV Cache。
- P99 可能仍然受长请求影响。
- 参数不好时可能牺牲吞吐。

### 方案 B：长短请求分池

适用：

- 短请求 SLO 明确。
- 长请求比例不稳定。
- 线上 P99 被长请求污染。

结构：

```text
Router
  -> short pool: 低 max_model_len / 小 batch token / 严格 SLO
  -> long pool: 高 max_model_len / chunked prefill / 更大 KV 预算
```

short pool：

- 低 `max_model_len`。
- 较小 `max_num_batched_tokens`，避免单轮过长。
- 更关注 TTFT 和 P99。
- 更高优先级。

long pool：

- 高 `max_model_len`。
- 开启 chunked prefill。
- 更关注 KV Cache 和 HBM bandwidth。
- 可以接受更长 TTFT。

优点：

- 保护短请求。
- 参数可以分别优化。
- 更容易定位瓶颈。

风险：

- 资源切分后，低流量池可能 GPU 利用率低。
- 需要动态路由和容量规划。
- 长请求突刺可能打爆 long pool。

### 方案 C：长请求专用异步队列

适用：

- 超长请求很少但非常重。
- 业务允许异步返回或更宽松 SLA。
- 短请求必须稳定。

策略：

- 超过阈值的请求进入 async long-context queue。
- 给用户返回任务 ID 或流式慢响应。
- 控制 long queue 并发。
- 必要时单独扩容 long pool。

优点：

- 最大限度保护在线短请求。
- 长请求可以用更激进的参数。

风险：

- 产品形态需要支持异步。
- 用户体验不同。

### 方案 D：PD 分离

适用：

- Prefill 和 Decode 干扰严重。
- 长 prompt 多，decode SLO 也重要。
- 有足够 GPU 和网络支撑 KV transfer。

结构：

```text
Prefill pool:
  处理长 prompt prefill
  compute-bound

Decode pool:
  处理 token generation
  memory-bound
```

优点：

- Prefill 和 Decode 可独立扩容。
- 更容易保护 decode TPOT。
- 长 prompt 不直接抢 decode GPU。

风险：

- KV transfer 成本。
- 拓扑亲和复杂。
- 容错复杂。
- worker 释放时间预测更重要。

### 方案 E：KV Cache / Prefix Cache / KV Router

适用：

- 长请求有大量重复 prefix。
- RAG / agent / 多轮对话存在上下文复用。
- 重复 prefill 成本高。

策略：

- 开 prefix cache。
- 用 KV-aware routing 提高命中。
- 路由时同时考虑 queue length 和 SLO。

关键：

```text
不是最高 cache hit，
而是 SLO 内有效 cache hit。
```

如果为了命中把请求打到一个忙 worker，短请求反而可能变慢。

## 如果并发不高但延迟大，怎么排查

### 第一步：判断慢的是 TTFT 还是 TPOT

TTFT 高：

- 排队。
- tokenization。
- long prefill。
- KV block 不足。
- scheduler batch 太大。

TPOT 高：

- decode memory-bound。
- KV Cache 太长。
- HBM bandwidth。
- TP/CP 通信。
- batch shape 不好。

### 第二步：看 GPU 是否真的忙

GPU utilization 低：

- client 没压满。
- API/tokenizer 慢。
- Scheduler loop 有 gap。
- 等通信。

GPU utilization 高但延迟大：

- 单个长 prefill 太重。
- decode HBM bandwidth 饱和。
- kernel shape 不好。
- KV Cache 读取压力大。

### 第三步：看 KV Cache

关注：

- KV block usage。
- preemption count。
- active sequences。
- total tokens in batch。
- input length 分布。
- 请求结束后 block 是否释放。

如果 KV block 接近上限：

- 降 `max_num_seqs`。
- 降 `max_model_len`。
- 长短分池。
- KV offload / prefix cache。
- CP / DCP。

### 第四步：看 batch shape

关注：

- 每轮 scheduled token 数。
- prefill token 数。
- decode token 数。
- batch 内 context length 分布。
- 是否一个长请求拖一轮。

如果 batch shape 不稳定：

- chunked prefill。
- 限制 long prompt 进入 short pool。
- 分长度 bucket。
- 分池。

## 参数策略

### short pool

目标：

- 低 TTFT。
- 稳 P99。

倾向：

- 较小 `max_model_len`。
- 较低 long prompt 阈值。
- 适中 `max_num_batched_tokens`。
- 限制长 prefill。
- 保留足够 KV block 给短请求。

### long pool

目标：

- 接受更高 TTFT。
- 尽量不 OOM。
- 稳定处理长上下文。

倾向：

- 较大 `max_model_len`。
- 开启 chunked prefill。
- 更谨慎的 `max_num_seqs`。
- 观察 KV block usage。
- 必要时 CP/DCP 或 KV offload。

### 混合单池

目标：

- 成本低。
- 运维简单。

倾向：

- 必须开启或评估 chunked prefill。
- 设输入长度 admission control。
- 对超长请求降优先级。
- 用 SLO violation rate 决定是否拆池。

## 实验设计

至少跑四组：

### 1. 短请求单独跑

目的：

- 得到 short pool 理想基线。
- 确认短请求本身没有 tokenizer / API / GPU 问题。

### 2. 长请求单独跑

目的：

- 得到 long pool 资源消耗。
- 看 prefill、KV Cache、decode HBM 瓶颈。

### 3. 长短混合单池

目的：

- 验证污染程度。
- 看短请求 P99 变差多少。

### 4. 长短分池

目的：

- 验证隔离收益。
- 比较整体 GPU 成本和 goodput。

指标：

- short TTFT/TPOT P99。
- long TTFT/TPOT P99。
- short goodput。
- long goodput。
- GPU utilization。
- KV block usage。
- preemption count。
- cache hit rate。

## 决策树

```text
短请求 P99 是否被长请求污染？
  是 -> 分池或 priority / chunked prefill
  否 -> 单池即可，继续优化 goodput

长请求是否经常 OOM 或 preempt？
  是 -> 降并发 / 限 max len / KV 优化 / CP-DCP
  否 -> 继续看 TTFT/TPOT

并发不高但 TTFT 高？
  GPU 空 -> 查 tokenizer / scheduler gap / client
  GPU 忙 -> 查 long prefill / chunked prefill / batch token

并发不高但 TPOT 高？
  查 HBM bandwidth / KV length / decode batch / communication

cache hit 高但 P99 差？
  路由加 queue penalty 和 SLO risk，不只看命中
```

## 一句话总结

```text
长短混合不是一个普通吞吐问题，
而是 prefill compute、KV capacity、decode bandwidth、scheduler budget 和 SLO priority 的综合调度问题。

低并发也可能慢，因为单个超长上下文就足以把 batch shape、KV Cache 和 HBM bandwidth 搞坏。
```
