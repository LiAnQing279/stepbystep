# 部署优化与 SLO 下最大效能 Playbook

日期：2026-07-03

## 定位

这份文档回答实际部署里的核心问题：

```text
给定现有 GPU、模型和业务 SLO，
如何选择部署形态、优化参数，并在 SLO 限制下实现最大效能？
```

这里的“最大效能”不是裸吞吐最大，而是：

```text
满足 SLO 的 goodput 最大
```

也就是在 TTFT、TPOT、P99、错误率和成本约束下，尽可能提高：

- TPM：tokens per minute。
- QPS / RPS。
- GPU 利用率。
- KV Cache 命中带来的有效加速。
- 单卡或单集群成本效率。

## 总流程

```text
1. 明确约束
  -> 2. 建立 workload
  -> 3. 估算资源
  -> 4. 选择部署拓扑
  -> 5. 设置初始参数
  -> 6. Benchmark + Profile
  -> 7. 定位瓶颈
  -> 8. 调参和重测
  -> 9. 确认 SLO goodput
  -> 10. 固化容量规划
```

每次部署优化都必须形成闭环，不能只调一个参数看一次吞吐。

## 1. 明确约束

### 业务 SLO

先确定：

- TTFT P50 / P90 / P99。
- TPOT P50 / P90 / P99。
- end-to-end latency。
- error rate。
- timeout。
- steady QPS。
- peak QPS。
- 是否流式返回。
- 是否允许排队。

关键判断：

- 如果 TTFT 是核心 SLO，重点优化 prefill、排队、chunked prefill、prefix cache。
- 如果 TPOT 是核心 SLO，重点优化 decode、KV Cache、HBM bandwidth、通信。
- 如果 P99 是核心 SLO，重点优化长短请求隔离、路由、排队策略和资源预留。

### 硬件约束

记录：

- GPU 型号和数量。
- 单卡显存。
- GPU 间互联：NVLink / PCIe。
- 跨节点网络：IB / RoCE / Ethernet。
- CPU 核数。
- host memory。
- 本地 SSD / 网络盘。

关键判断：

- 单机 NVLink 适合 TP。
- 跨节点大 TP 风险高，通常优先 `TP within node + PP across nodes`。
- HBM 容量限制模型和 KV Cache；HBM 带宽限制 decode。
- CPU/tokenizer 也可能限制服务端 QPS。

### 模型约束

记录：

- Dense / MoE / MLA / Linear Attention。
- 参数量。
- dtype / quantization。
- max model length。
- KV head 数。
- 是否支持 MTP / speculative decoding。
- 是否需要 trust remote code。

关键判断：

- Dense 模型优先考虑 TP / PP / DP。
- MoE 模型必须考虑 EP 和 All-to-All。
- 长上下文模型必须考虑 KV Cache 和 CP / DCP。
- MTP / speculative decoding 会改变 decode step 和 worker 释放时间。

## 2. 建立 workload

部署优化不能只用固定长度 synthetic prompt。

至少记录：

- input length 分布。
- output length 分布。
- 并发分布。
- 到达过程：平稳、突刺、周期性。
- 是否存在 prefix 重复。
- 是否存在长短请求混合。
- 是否多租户。
- 是否有优先级。

推荐最小 workload：

```text
短请求：input 128-512, output 64-256
中请求：input 512-2048, output 128-512
长请求：input 4096-32768+, output 128-1024
混合请求：按线上比例采样
```

关键判断：

- 固定输入长度只能验证单点能力。
- 混合负载才能暴露 P99、调度、KV Cache 和路由问题。
- 如果 prefix 重复率高，KV Cache / prefix cache / KV router 是关键优化点。
- 如果存在不规则超长上下文和短请求混合，必须单独评估长短请求互相污染，详见 `docs/mixed-long-context-serving-playbook.md`。

## 3. 估算资源

### 权重显存

先判断模型权重是否能放下：

```text
weight_memory ~= params × bytes_per_param
```

再考虑：

- tensor parallel 后每卡权重分片。
- MoE expert 是否复制或分布。
- quantization 后权重变化。
- framework workspace。

### KV Cache 显存

KV Cache 粗略随这些变量增长：

```text
batch_size × sequence_length × num_layers × kv_heads × head_dim × dtype
```

关键判断：

- 权重显存决定模型能不能放下。
- KV Cache 决定能同时服务多少请求和多长上下文。
- 长上下文场景里，KV Cache 往往比权重更快成为硬约束。

### 计算与带宽

Prefill：

- 更偏 compute-bound。
- 主要影响 TTFT。
- 输入越长越明显。

Decode：

- 更偏 memory-bound。
- 主要影响 TPOT。
- 受 HBM bandwidth、KV Cache 读取、batch shape 影响。

## 4. 选择部署拓扑

### Dense 模型

优先级：

```text
单卡能放下：DP 多副本
单卡放不下：TP
单节点 TP 还不够：TP + PP
吞吐不够：外层 DP
长上下文：TP + CP/DCP 或 KV Cache 优化
```

### MoE 模型

优先级：

```text
先判断 expert 是否能复制
expert 太大或路由稀疏：EP
attention/dense 部分：TP 或 DP
跨节点：重点看 All-to-All 和拓扑
吞吐不够：外层 DP 或更多 decode worker
```

### 长上下文

优先级：

```text
先优化 KV Cache 管理和 max_model_len
再考虑 CP / DCP
如果 prefix 重复高，优先考虑 prefix cache / KV router
长短请求混合时，考虑分池或分路由
```

不规则长短混合场景优先看：

- 短请求 P99 是否被长请求污染。
- 低并发下是否仍然存在长 prefill 独占、KV block 占满或 decode HBM 带宽瓶颈。
- 是否需要 short / medium / long pool 分池。
- 是否需要异步 long-context queue。

### PD 分离

适用：

- Prefill 和 Decode 干扰严重。
- TTFT 和 TPOT 很难同时满足。
- 长 prompt 和长 decode 混合。
- 希望独立扩展 Prefill worker 和 Decode worker。

关键问题：

- Prefill / Decode worker 配比。
- KV transfer 成本。
- worker 释放时间预测。
- 拓扑亲和。
- 容错和重试策略。

## 5. 初始参数集合

### 影响并发和显存

重点参数类别：

- max model length。
- max number of sequences。
- max batched tokens。
- GPU memory utilization。
- KV Cache dtype。
- block size。
- swap / offload。

调参逻辑：

- OOM 或 preemption 多：降低并发、降低 max model length、降低 batched tokens、优化 KV Cache。
- GPU 空闲：增加 batched tokens、增加并发、检查 client/API/tokenizer。
- TTFT 高：限制长 prefill、使用 chunked prefill、分离长短请求。
- TPOT 高：看 decode batch、HBM bandwidth、通信和 KV Cache。

### 影响调度

重点参数类别：

- chunked prefill。
- scheduler policy。
- request priority。
- preemption mode。
- speculative decoding / MTP。
- prefix cache。

调参逻辑：

- TTFT 和 TPOT 冲突时，先观察 prefill/decode mix。
- 长 prompt 拖短请求时，考虑 chunked prefill 或分池。
- cache 命中高但 P99 变差时，说明路由只看命中不够，还要看 queue 和 SLO。

### 影响并行

重点参数类别：

- tensor parallel size。
- pipeline parallel size。
- data parallel size。
- expert parallel。
- context parallel / decode context parallel。

调参逻辑：

- TP 越大，单卡权重压力越小，但通信越重。
- PP 解决超大模型放置，但可能引入 bubble。
- DP 扩吞吐，但可能打散 KV Cache 命中。
- EP 只服务 MoE expert，但 All-to-All 可能成为瓶颈。
- CP/DCP 只在长上下文/KV 压力明显时值得上。

## 6. Benchmark 方法

### 指标

必须同时看：

- TTFT。
- TPOT。
- end-to-end latency。
- prompt tokens/s。
- output tokens/s。
- TPM。
- QPS。
- GPU utilization。
- GPU memory。
- HBM bandwidth。
- KV Cache usage。
- cache hit rate。
- preemption count。
- NCCL / communication time。

### Goodput

不要只看 throughput，要看满足 SLO 的 goodput：

```text
goodput = 满足 SLO 的请求或 token 数 / 时间
```

例如：

```text
TTFT P99 <= 2s
TPOT P99 <= 50ms
error rate <= 0.1%
```

在这个条件下比较：

- 哪个参数组合 TPM 最高？
- 哪个部署拓扑 GPU 利用率最高？
- 哪个方案成本最低？

## 7. 调参循环

每次只改一类参数：

```text
baseline
  -> 改并发参数
  -> 改调度参数
  -> 改并行拓扑
  -> 改 KV Cache 策略
  -> 改路由策略
  -> 改 speculative decoding / MTP
```

每次记录：

- 改了什么。
- 为什么改。
- 预期改善哪个指标。
- 实际改善哪个指标。
- 是否破坏其他 SLO。
- 瓶颈是否转移。

## 8. 常见症状到优化方向

| 症状 | 优先怀疑 | 优化方向 |
|------|----------|----------|
| TTFT 高 | 排队 / Prefill / tokenizer / 长 prompt | chunked prefill、分池、prefix cache、限制 long prompt |
| TPOT 高 | Decode / HBM bandwidth / KV Cache / 通信 | 调 decode batch、KV dtype/layout、减少通信、MTP |
| GPU 利用率低 | client/API/tokenizer/Scheduler gap | 压测工具、CPU、batch token、scheduler loop |
| GPU 满但 TPM 低 | memory-bound / 通信 / batch shape | profile kernel、HBM、NCCL、调并行 |
| OOM | 权重 / KV Cache / max_model_len | 降 max len、降并发、TP/PP、KV offload |
| P99 差 | 长短请求混跑 / 路由 / 网络抖动 | 分池、优先级、KV-aware routing、拓扑亲和 |
| 并发低但延迟高 | 单个长 prompt、batch shape 差、KV/HBM 被长请求占住 | 看 TTFT/TPOT、分池、chunked prefill、限 max len |
| cache hit 高但 SLO 差 | 只追命中，忽略排队 | 路由加 queue/SLO penalty |
| 多卡扩展差 | TP/EP/PP/CP 通信 | 改拓扑、减跨节点 TP、优化 placement |

## 9. SLO 下提高 KV 命中

目标不是最高 cache hit rate，而是：

```text
SLO 内的有效 cache hit
```

路由时至少考虑：

- 目标 worker 是否有对应 KV / prefix cache。
- 目标 worker queue length。
- 目标 worker decode saturation。
- KV 复用能节省多少 prefill。
- 因排队增加的延迟是否超过节省。
- 请求是否属于高优先级。

简单评分思路：

```text
score = cache_benefit
        - queue_penalty
        - transfer_cost
        - SLO_risk
```

## 10. 容量规划输出

一次部署优化完成后，至少输出：

- 当前硬件和模型配置。
- 选择的并行拓扑。
- 参数配置。
- workload 分布。
- SLO 定义。
- goodput 曲线。
- 最大安全 QPS / TPM。
- GPU 利用率区间。
- KV Cache 使用上限。
- 触发扩容的指标。
- 已知风险和回滚方案。

## 和现有学习路线的关系

这条线贯穿所有主题：

```text
Scheduler
  -> KV Cache
  -> Parallel Composition
  -> Speculative Decoding / MTP
  -> KV Router
  -> PD Scheduling
  -> Topology-aware Scheduling
```

最终都要回到一个问题：

```text
在现有 GPU 和业务 SLO 下，哪种部署方案的 goodput 最高？
```
