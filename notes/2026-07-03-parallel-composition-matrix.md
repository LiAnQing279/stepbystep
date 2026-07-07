# 分布式推理并行组合矩阵

日期：2026-07-03

## 目标

回答一个更实际的问题：

```text
TP / DP / PP / EP / CP / SP / DCP 等并行方式，
哪些可以组合？
哪些组合没有意义或限制很强？
限制条件和使用范围是什么？
```

这份笔记面向推理服务，不是训练。训练里的 DP/FSDP/ZeRO/activation checkpoint 等问题先不展开。

## 先给结论

多数并行方式在数学上可以组合，但在线推理里不等于都值得组合。

推理服务里常见的组合顺序是：

```text
单机：TP
吞吐扩展：外层 DP
超大 Dense 模型：TP + PP，再外层 DP
MoE 模型：TP + EP，或 DP attention + EP expert
长上下文：TP + CP / DCP，必要时再外层 DP
大规模在线服务：DP × (TP/PP/EP/CP 组成的一个模型副本)
```

最重要的边界：

- DP 是复制模型副本，处理不同请求。
- TP / PP / CP 是把同一个请求拆到多卡上。
- EP 只对 MoE 的 expert 层有意义。
- CP / DCP 主要对长上下文和 KV Cache 压力有意义。
- SP 在推理里存在感弱，通常作为 TP 的附属优化理解。

## 组合公式

一个推理集群可以先粗略拆成：

```text
总 GPU 数
  = DP 组数
  × 每个模型副本使用的 GPU 数
```

每个模型副本内部再由多种模型并行组成：

```text
每个模型副本使用的 GPU 数
  ~= TP × PP × CP
```

MoE 场景特殊：

```text
EP 通常不是简单再乘一个独立维度，
而是 expert 层在已有 TP/DP/PP 组内重新切 expert。
```

在 vLLM 的一些 MoE 部署里，EP size 可以由 TP size 和 DP size 共同决定；attention 层和 expert 层可能使用不同的并行方式。这意味着不能只用一个简单乘法公式解释所有 MoE serving。

## 各并行方式的角色

| 并行 | 切分对象 | 主要解决 | 推理里最常见用途 |
|------|----------|----------|------------------|
| DP | 请求 / batch | 扩吞吐、扩副本 | 多实例服务、负载均衡 |
| TP | 单层权重 / 激活 | 单卡放不下、单请求加速 | 单机多卡部署大模型 |
| PP | 模型层 | 模型太深/太大、跨节点放置 | 超大 Dense 模型跨节点 |
| EP | MoE experts | expert 太多、MoE 计算稀疏 | MoE 模型 expert 分布 |
| CP | 序列 / 上下文 | 长上下文 prefill、KV 压力 | 长上下文请求 |
| DCP | Decode 阶段 context / KV | Decode KV Cache 过大 | 长上下文 decode |
| SP | 序列维度的非 attention 部分 | 激活显存 | 推理里一般不是主线 |

## 常见可组合关系

### DP + TP

结论：最常见、最实用。

结构：

```text
DP = 多个模型副本
每个副本内部用 TP
```

例子：

```text
8 卡机器：
TP=4, DP=2
=> 2 个副本，每个副本 4 卡
```

适用：

- 模型需要多卡才能放下。
- 单个 TP 组吞吐不够，需要多个副本扩 QPS。
- 在线服务需要横向扩展。

限制：

- 每个 DP 副本都要完整持有自己的模型分片和 KV Cache。
- DP 副本之间默认不共享 KV Cache。
- 负载均衡不好会导致有的副本排队，有的副本空闲。

排查重点：

- DP 之间请求是否均衡。
- 每个 TP 组内部通信是否成为瓶颈。
- KV Cache 命中是否被 DP 路由打散。

### TP + PP

结论：适合超大 Dense 模型，尤其跨节点。

结构：

```text
每个 pipeline stage 内部用 TP
不同 stage 放不同层
```

例子：

```text
2 机 × 8 卡：
TP=8, PP=2
=> 每个节点一个 pipeline stage，每个 stage 内 8 卡 TP
```

适用：

- 模型大到单节点 TP 仍然放不下。
- 跨节点 TP 通信太重，不如用 PP 只传层间激活。
- 离线推理或高吞吐场景可以忍受 pipeline bubble。

限制：

- 在线小 batch 推理中，PP 容易有 bubble，增加延迟。
- 流式 decode 每 step 都要经过所有 pipeline stage，调度复杂。
- stage 切分不均会导致慢 stage 拖全局。

排查重点：

- 每个 PP stage 是否负载均衡。
- stage 间 P2P 通信是否卡住。
- micro-batch 数是否足够隐藏 bubble。

### TP + EP

结论：MoE 模型常用，Dense 模型无意义。

结构：

```text
Attention / dense 部分：TP
Expert 部分：EP
```

适用：

- MoE 模型 expert 数量多。
- expert 权重无法全部放在每张卡。
- All-to-All 通信成本可以接受。

限制：

- 只适用于 MoE 层。
- expert 路由不均会导致负载不均。
- EP 需要 All-to-All，跨节点时网络压力很大。
- TP 和 EP 的通信模式不同，一个是每层 collective，一个是按 token dispatch。

排查重点：

- expert load balance。
- All-to-All 时间。
- token dispatch / combine 时间。
- 某些 expert 是否过热。

### DP + EP

结论：MoE serving 中很重要，但不是普通 DP。

结构：

```text
attention 层可能按 DP 复制
expert 层跨 DP ranks 做 EP
```

适用：

- MoE 模型，例如 DeepSeek 类模型。
- attention 部分复制更便宜，expert 部分需要分布。
- 希望扩大吞吐同时减少 expert 权重重复。

限制：

- DP rank 之间不再完全独立。
- expert 层 forward 需要跨 rank 同步。
- 请求数少于 DP rank 时，可能有 rank 空等。

排查重点：

- DP rank 是否真的独立。
- expert 层是否强制同步。
- 小并发下 GPU 是否空转。

### TP + CP

结论：长上下文场景可用，但通信复杂。

结构：

```text
TP 切 hidden / weight
CP 切 sequence / context
```

适用：

- 长上下文 prefill 很慢。
- KV Cache 随上下文长度爆炸。
- 单 TP 组内存不够存完整 KV。

限制：

- CP 会引入 Attention 相关通信。
- Prefill CP 和 Decode CP 的目标不同。
- 普通短上下文不值得上 CP。
- TP size、KV head 数、DCP size 之间可能有整除约束。

排查重点：

- 长上下文 TTFT 是否下降。
- Decode TPOT 是否因跨卡读 KV 变差。
- KV Cache 是否真正减少复制。
- CP 通信是否超过收益。

### TP + DCP

结论：Decode 长上下文专用组合。

结构：

```text
在 TP 组内复用 GPU，把 Decode 阶段的 KV Cache 沿 context 维度切开
```

适用：

- Decode 阶段 KV Cache 太大。
- 长上下文导致 batch size 上不去。
- 希望减少 KV Cache duplication，提高并发。

限制：

- 通常不额外增加 world size，而是在已有 TP 组内切。
- DCP size 通常受 TP size 和 KV head 数约束。
- 每个 decode step 可能引入额外通信。

排查重点：

- KV Cache 容量是否释放出来。
- output tokens/s 是否上升。
- TPOT 是否因通信变差。

### DP + TP + PP

结论：大模型在线服务的经典结构。

结构：

```text
外层 DP 扩吞吐
每个副本内部 TP + PP 承载超大模型
```

例子：

```text
4 机 × 8 卡：
TP=8, PP=2, DP=2
=> 2 个副本，每个副本 2 机 16 卡
```

适用：

- 模型很大。
- 单个 TP 组放不下。
- 需要多个副本承载线上流量。

限制：

- 资源颗粒度很粗，一个副本要占很多 GPU。
- DP 层路由会影响 KV Cache 命中。
- PP bubble 和 TP 通信会同时存在。

排查重点：

- 副本之间负载均衡。
- stage 之间负载均衡。
- TP collective 时间。
- KV Cache 命中是否被路由打散。

### TP + PP + EP

结论：超大 MoE 模型会遇到，但复杂度很高。

结构：

```text
PP 切层
每个 stage 内 TP
MoE 层内 EP
```

适用：

- 超大 MoE，单节点放不下。
- 需要跨节点部署。
- 网络和调度能力足够强。

限制：

- TP collective、PP P2P、EP All-to-All 会叠加。
- MoE 层和 Dense 层负载不同，stage balance 更难。
- 在线服务长尾可能明显。

排查重点：

- All-to-All 是否成为瓶颈。
- pipeline stage 是否被 MoE 层拖慢。
- expert imbalance 是否放大 PP bubble。

### DP + TP + EP

结论：MoE 在线服务常见高阶组合。

结构：

```text
外层看起来有 DP 扩吞吐
attention/dense 层用 DP 或 TP
expert 层用 EP
```

适用：

- DeepSeek / Mixtral 类 MoE。
- expert 层权重很大。
- 希望兼顾吞吐和 expert locality。

限制：

- DP 不再是完全独立副本。
- expert 同步和 All-to-All 对网络敏感。
- 小流量下可能有 GPU 空转。

排查重点：

- attention 和 expert 层是不是使用了不同并行组。
- EP size 如何由 TP/DP 组合得到。
- expert layer 是否需要所有 rank 同步。

### DP + TP + CP / DCP

结论：长上下文在线服务常见方向。

结构：

```text
外层 DP 扩吞吐
每个副本内部 TP
长上下文请求用 CP / DCP 降低 KV 压力
```

适用：

- 长文档、代码库、RAG、大上下文对话。
- KV Cache 容量限制 batch size。
- SLO 要求下希望提高 TPM。

限制：

- DP 路由会影响 KV Cache 命中。
- CP/DCP 会增加通信。
- 长短请求混跑时容易影响短请求延迟。

排查重点：

- 长上下文请求是否应该单独路由。
- KV Cache block 是否成为瓶颈。
- SLO 是否被 CP 通信破坏。

## 不推荐或限制很强的组合

### Dense 模型 + EP

不推荐。

原因：

- Dense 模型没有 expert，EP 没有切分对象。
- Dense 模型应该优先考虑 TP / PP / DP / CP。

### 普通短上下文 + CP / DCP

通常不推荐。

原因：

- 上下文不长时，KV Cache 不是主要瓶颈。
- CP/DCP 的通信开销可能大于收益。
- 会让调度和排查复杂化。

### 跨节点大 TP

限制很强。

原因：

- TP 每层都有高频通信。
- 跨节点带宽和延迟通常不如节点内 NVLink。
- 如果模型必须跨节点，通常优先考虑 `TP within node + PP across nodes`。

### 在线低并发 + 大 PP

限制很强。

原因：

- PP 需要足够 micro-batch 才能隐藏 bubble。
- 在线 decode 往往 batch 小、step 多。
- 低并发下 PP stage 容易空等。

### DP 只看负载均衡，不看 KV Cache 命中

不是真正不能组合，但线上容易出问题。

原因：

- KV Cache / Prefix Cache 命中依赖请求路由。
- 只按 queue length 做 DP 路由，可能打散 cache locality。
- 只按 cache hit 路由，又可能把请求打到忙副本，破坏 SLO。

## 选择策略

### 先判断模型类型

Dense：

```text
单卡能放下：DP
单卡放不下：TP
单节点 TP 还不够：TP + PP
长上下文：TP + CP/DCP
吞吐不够：外层 DP
```

MoE：

```text
expert 能复制：先 TP/DP 简化
expert 太大：EP
attention 和 expert 负载不同：DP attention + EP expert
跨节点：TP/EP/PP 组合，重点看 All-to-All 和拓扑
```

长上下文：

```text
短上下文不加 CP
Prefill 太慢：考虑 Prefill CP / Chunked Prefill
Decode KV 太大：考虑 DCP / KV offload / cache management
在线混合负载：长短请求分池或分路由
```

### 再判断瓶颈

显存不够：

```text
权重显存不够：TP / PP / EP
KV Cache 显存不够：CP / DCP / KV offload / 降 max_model_len / 限并发
```

吞吐不够：

```text
单副本 GPU 没吃满：调 batch / scheduler
单副本已经吃满：DP 扩副本
MoE expert 不均：EP load balance
```

延迟不满足：

```text
TTFT 高：prefill、排队、长 prompt、chunked prefill、CP
TPOT 高：decode、HBM bandwidth、KV Cache、TP/CP 通信
P99 高：调度、长短请求混跑、KV routing、网络抖动
```

通信太重：

```text
TP 通信太重：减 TP，增加 PP 或更强节点内互联
EP All-to-All 太重：优化 expert placement / load balance / 拓扑
CP 通信太重：只给长上下文开 CP，或改 KV cache 策略
```

## 常见配置模式

### 单机 8 卡 Dense 70B

```text
TP=8
DP=1
PP=1
```

特点：

- 简单。
- 适合延迟敏感。
- 依赖节点内高速互联。

### 单机 8 卡 Dense 7B，多副本

```text
TP=1
DP=8
```

或：

```text
TP=2
DP=4
```

特点：

- 模型小，优先扩副本。
- 注意 DP 路由和 KV Cache 命中。

### 2 机 16 卡 Dense 超大模型

```text
TP=8
PP=2
DP=1
```

特点：

- 节点内 TP，节点间 PP。
- 避免跨节点 TP。

### 单机 8 卡 MoE

```text
TP=1 或 2
DP=若干
EP=由 TP/DP 组合形成
```

特点：

- attention 和 expert 可能使用不同并行。
- 重点观察 All-to-All 和 expert load balance。

### 长上下文服务

```text
TP=N
DCP=M
外层 DP=K
```

特点：

- DCP 用来减少 Decode 阶段 KV Cache duplication。
- 只在长上下文场景下收益明显。
- 需要观察 TPOT 是否被通信拖慢。

## 排查时的固定问题

1. 这个请求是在一个 DP 副本内处理，还是跨多个并行组处理？
2. 当前模型副本内部用了哪些并行：TP、PP、EP、CP、DCP？
3. 这些并行分别切的是权重、层、expert、sequence、KV，还是请求？
4. 组合后最频繁的通信是什么：AllReduce、AllGather、All-to-All、P2P、KV transfer？
5. 通信发生在每层、每 MoE 层、每 pipeline boundary，还是每 decode step？
6. 这个组合解决的是权重显存、KV 显存、吞吐、TTFT、TPOT，还是长上下文？
7. 它引入了什么新问题：bubble、All-to-All、cache locality、拓扑依赖、SLO 长尾？

## 一句话总结

```text
DP 扩请求，TP 切层内，PP 切层间，EP 切 expert，CP/DCP 切上下文和 KV。

组合不是越多越好；
每加一种并行，都必须说清楚它解决哪个瓶颈，又新增哪种通信和调度风险。
```
