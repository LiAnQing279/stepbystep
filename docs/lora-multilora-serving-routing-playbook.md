# LoRA / 多 LoRA 部署与路由 Playbook

日期：2026-07-03

## 定位

LoRA / 多 LoRA serving 是一条独立工作线。

它不是简单“加载一个 adapter”，而是涉及：

- adapter 生命周期。
- 多 LoRA batching。
- LoRA 路由。
- adapter cache。
- 多租户隔离。
- SLO。
- GPU/CPU/存储资源竞争。
- 故障排查和版本管理。

核心问题：

```text
一个 base model 服务多个 LoRA adapter 时，
如何在保证 SLO 的前提下，提高 adapter 命中、batch 效率和 GPU 利用率？
```

## 系统视角

```text
Client Request
  -> 识别 tenant / task / adapter id
  -> LoRA Router
  -> 选择 base model replica
  -> 加载或命中 LoRA adapter
  -> Scheduler batch
  -> Worker 执行 base model + LoRA delta
  -> 返回结果
```

LoRA serving 不是只多了一个权重文件，而是在请求维度引入了新的调度变量：

```text
request = prompt + sampling params + adapter id + tenant + SLO
```

## 关键问题

### 1. Adapter 放在哪里

候选位置：

- GPU memory。
- CPU memory。
- 本地 SSD。
- 远端 object storage。
- 镜像内预置。

判断：

- 高频 adapter 放 GPU。
- 中频 adapter 放 CPU 或本地 SSD。
- 低频 adapter 按需加载。
- 超多 adapter 需要 cache / eviction 策略。

### 2. Adapter 什么时候加载

模式：

- 启动时加载固定 LoRA。
- 运行时动态加载。
- 请求到达时 lazy load。
- 预热热门 adapter。

权衡：

- 启动加载：延迟稳定，但显存占用高。
- 动态加载：灵活，但首次请求 TTFT 高。
- lazy load：节省资源，但 P99 容易差。

### 3. 多 LoRA 如何 batch

难点：

- 同一个 batch 里可能有不同 adapter。
- base model 计算可共享。
- LoRA delta 需要按 adapter 区分。
- adapter 数越多，batch shape 和 kernel 路径越复杂。

需要关注：

- 一个 batch 允许多少不同 LoRA？
- 是否按 adapter 分组 batch？
- 是否支持 Punica / S-LoRA 类 multi-adapter kernel。
- 多 LoRA 是否导致 kernel launch 增多。

### 4. LoRA 路由怎么设计

路由输入：

- adapter id。
- base model id。
- tenant。
- request priority。
- SLO。
- adapter 当前在哪些 worker 上。
- worker queue length。
- GPU memory。
- adapter load cost。

路由目标：

```text
score = adapter_hit_benefit
        - queue_penalty
        - load_cost
        - SLO_risk
        - tenant_isolation_penalty
```

不能只追 adapter 命中。

如果为了命中 adapter，把请求路由到一个排队很长的 worker，SLO 反而会变差。

## 多 LoRA 部署模式

### 模式 A：固定小数量 LoRA

适用：

- adapter 数少。
- 热度稳定。
- 每个 adapter 流量足够。

策略：

- 启动时加载所有 adapter。
- 固定路由。
- 简化 cache。

优点：

- 稳定。
- 首 token 延迟可控。

风险：

- GPU memory 被 adapter 占用。
- adapter 增长后不可扩展。

### 模式 B：热门 LoRA 常驻 + 冷门动态加载

适用：

- adapter 数量中等或较多。
- 热度符合长尾分布。

策略：

- top-K adapter 常驻 GPU。
- 中频 adapter 放 CPU / SSD。
- 冷门 adapter 远端拉取。
- 设置 eviction 策略。

优点：

- 资源利用率更好。
- 对热门请求稳定。

风险：

- 冷门 adapter 首次请求 P99 高。
- eviction 错误会导致抖动。

### 模式 C：按租户或业务分池

适用：

- 多租户。
- 不同租户 SLO 不同。
- adapter 数很多。

策略：

- 高优租户独立 pool。
- 普通租户共享 pool。
- 低频 adapter 走异步或低优队列。

优点：

- 隔离性好。
- SLO 更可控。

风险：

- 资源碎片。
- 低流量 pool GPU 利用率低。

### 模式 D：LoRA Router + Adapter Cache

适用：

- 大量 adapter。
- adapter 热度变化快。
- 需要动态调度。

策略：

- 路由器感知 adapter location。
- worker 维护 adapter cache。
- 路由同时考虑 cache hit、queue length、load cost 和 SLO。

优点：

- 更适合大规模多 LoRA serving。

风险：

- 路由复杂。
- cache 元数据同步复杂。
- 需要完善 metrics。

## 参数和资源关注点

### 关键参数类别

- max LoRA rank。
- max LoRA adapters per worker。
- max active LoRAs per batch。
- LoRA dtype。
- adapter cache size。
- adapter load timeout。
- CPU / GPU adapter cache 配额。
- eviction policy。

### 资源竞争

LoRA 会新增竞争：

- GPU memory：adapter 常驻占显存。
- CPU memory：adapter cache。
- storage：动态加载 adapter。
- PCIe / host-to-device copy：adapter 加载到 GPU。
- Scheduler budget：不同 adapter 的请求能否合 batch。
- SLO budget：adapter miss 会增加 TTFT。

## Benchmark 设计

### 基础场景

1. 单 LoRA。
2. 多 LoRA，adapter 数固定。
3. 热门 adapter 重复请求。
4. 长尾 adapter 随机请求。
5. adapter 冷启动。
6. 多租户混合。

### 指标

- TTFT / TPOT。
- adapter hit rate。
- adapter load time。
- adapter eviction count。
- active LoRA count。
- GPU memory used。
- CPU memory used。
- storage read latency。
- batch 中 distinct adapter 数。
- SLO violation rate。
- goodput。

### 关键实验

```text
固定总 QPS，增加 adapter 数：
  1 -> 4 -> 16 -> 64 -> 256

固定 adapter 数，改变热度分布：
  均匀分布 vs Zipf 分布

固定 adapter cache，改变命中率：
  观察 TTFT P99 和 goodput
```

## 故障模式

### Adapter 找不到

可能原因：

- adapter id 错误。
- registry 没同步。
- object storage 权限问题。
- 路径不一致。

### Adapter 加载失败

可能原因：

- rank 不匹配。
- base model 不匹配。
- dtype 不兼容。
- 权重文件损坏。

### 输出质量异常

可能原因：

- adapter 和 base model 不匹配。
- tokenizer / chat template 不匹配。
- LoRA scaling 配置错误。
- 多 adapter batch 中 adapter id 映射错误。

### P99 很差

可能原因：

- adapter miss。
- 动态加载慢。
- eviction 频繁。
- 只按 adapter hit 路由，忽略 queue。

### GPU OOM

可能原因：

- 常驻 adapter 太多。
- max active LoRA 太高。
- KV Cache 和 adapter 共同占显存。
- batch 中 distinct adapter 太多。

## 路由设计原则

### 不只看 adapter hit

错误路由：

```text
有 adapter 就路由过去
```

正确路由：

```text
adapter hit benefit
  vs queue delay
  vs load cost
  vs SLO risk
```

### 区分热门和冷门

- 热门 adapter：优先常驻和高命中。
- 冷门 adapter：允许更高 TTFT 或异步队列。
- 高优租户：独立容量或更高 cache priority。

### 防止 cache 抖动

如果 adapter 热度变化快，需要：

- eviction hysteresis。
- admission policy。
- 预热策略。
- load rate limit。

## 和 KV Router 的关系

LoRA router 和 KV router 很像，但对象不同：

```text
KV Router:
  路由到已有 prefix / KV Cache 的 worker

LoRA Router:
  路由到已有 adapter 的 worker
```

复杂场景下，两者要一起考虑：

```text
score = KV_cache_benefit
        + adapter_hit_benefit
        - queue_penalty
        - load_or_transfer_cost
        - SLO_risk
```

## 与现有路线的关系

LoRA / 多 LoRA 会横跨：

- Runtime Engine。
- Scheduler。
- KV / Adapter Cache。
- Deployment。
- Multi-tenancy。
- SLO Goodput。

它适合放在 KV Router 之后学习，因为两者路由思想相似。

建议路线：

```text
Scheduler
  -> KV Cache
  -> KV Router
  -> LoRA / Multi-LoRA Router
  -> Multi-tenant Serving
```

## 后续阅读材料

正式展开前建立：

- `readings/YYYY-WW-lora-multilora-serving-reading-list.md`

候选材料：

- vLLM LoRA serving 文档。
- vLLM dynamic LoRA serving 文档。
- SGLang LoRA serving 文档。
- S-LoRA 论文。
- Punica 论文。
- LoRA adapter cache / routing 相关系统文章。

## 一句话总结

```text
多 LoRA 部署的核心不是“能加载多个 adapter”，
而是 adapter 命中、batch 效率、cache 管理、路由策略和 SLO 之间的系统权衡。
```
