# 导师要点 Backlog

日期：2026-07-03

## 定位

这份文档用于收集导师补充的在线推理系统要点，并把它们放回现有学习路线。

这些主题暂时不全部展开。当前优先级仍然是：

```text
vLLM 请求生命周期
  -> Scheduler
  -> KV Cache
  -> PD 分离和在线调度
```

导师要点主要归为三条更高阶主线：

1. 长上下文和 Context Parallel。
2. KV Cache 系统化管理与路由。
3. Speculative Decoding / MTP 这类 Decode 加速技术。
4. PD 分离式服务的调度、容错、拓扑亲和和 SLO 优化。

## 主题归类

| 导师要点 | 系统地图层级 | 解决的核心问题 | 建议学习阶段 |
|----------|--------------|----------------|--------------|
| TP / DP / PP / EP / CP 组合关系 | Parallel / Communication / Scheduler | 一组并发中哪些并行可以组合，组合后有哪些限制和排查风险 | Scheduler 后、KV Cache 前 |
| context 并行 | Parallel / KV Cache / Communication | 超长上下文下 KV Cache 和 Attention 计算如何跨卡扩展 | KV Cache 后 |
| KV 缓存管理：HiCache、LMCache、Mooncake | KV Cache / Runtime Engine / Storage | KV Cache 如何复用、卸载、迁移、持久化和跨实例共享 | Scheduler 后 |
| KV router | Scheduler / Deployment / KV Cache | 请求如何路由到更可能命中 KV Cache 的实例 | KV Cache 管理后 |
| LoRA / 多 LoRA 部署与路由 | Runtime / Scheduler / Deployment | 多 adapter serving 下如何做 adapter cache、multi-LoRA batching、LoRA router 和多租户 SLO | KV Router 后 |
| MTP / speculative decoding / DSpark | Runtime / Scheduler / Model | 通过预测多个未来 token 减少 Decode step，提高 TPM 和降低 TPOT | Scheduler + KV Cache 后 |
| PD 调度 | Scheduler / Deployment | Prefill 和 Decode 分离后如何分配请求、迁移 KV、提高吞吐 | Scheduler 后 |
| 提前预判 worker 结束 | Scheduler / Deployment | 预测 Decode worker 释放时间，减少排队和空转 | PD 调度中 |
| PD 容错 | Deployment / Runtime Engine | Prefill、Decode 或 KV 传输失败时如何恢复 | PD 调度后 |
| PD 中拓扑亲和 | Scheduler / Communication / Hardware | 结合 NVLink、PCIe、IB/RDMA 拓扑降低 KV 传输成本 | PD 调度后 |
| 实际部署优化 | Deployment / Benchmark / Scheduler | 在现有 GPU 和业务 SLO 下选择部署拓扑、优化参数并最大化 goodput | 贯穿全程 |
| 不规则超长上下文 + 短请求混合 | Scheduler / KV Cache / Deployment | 保护短请求 P99，解释低并发高延迟，设计分池/路由/参数策略 | 贯穿全程 |
| 特定模型 vLLM 自定义实现 | Runtime / Kernel / Scheduler | 针对模型结构、workload 和 SLO，自定义 model adapter、op、attention、KV 或 scheduler 提升推理效率 | Scheduler + Profile 后 |
| 论文新特性接入 SGLang 镜像 | Runtime / Deployment / Benchmark | 将 DSpark、MTP 等新特性接入 SGLang runtime，构建自定义镜像并验证 SLO goodput | Speculative Decoding 后 |
| 异常推理失败排查 | Observability / Runtime / Deployment | 对启动失败、请求失败、OOM、hang、输出异常、性能退化和多卡通信故障做止血、定位、复现和复盘 | 贯穿全程 |
| 多机训练问题与排障 | Training Systems / Communication / Checkpoint | 理解训练中的数据、数值、并行、通信、checkpoint、容错问题，以及它们如何影响推理部署 | 旁线，按需学习 |
| 提高 GPU 利用率、提升 TPM | Scheduler / Benchmark / Runtime | 在 SLO 内提高 tokens per minute 和 GPU 饱和度 | 贯穿全程 |
| SLO 条件下提高 KV 缓存命中 | Scheduler / KV Cache / Router | 在延迟约束内最大化缓存复用 | KV router 后 |

## 推荐进入顺序

### 阶段 1：Scheduler 基础

目标：

- 看懂一次请求如何进入 batch。
- 看懂 waiting / running / finished 状态流转。
- 建立 TTFT、TPOT、吞吐、显存之间的关系。

对应已有主线：

- `readings/2026-W28-vllm-scheduler-reading-list.md`
- `notes/2026-07-03-vllm-request-lifecycle.md`

暂不展开：

- KV router。
- PD 容错。
- 拓扑亲和。

### 阶段 2：KV Cache 管理

进入 KV Cache 管理前，先补一层并行组合关系：

- 哪些并行方式是在切请求，哪些是在切模型，哪些是在切上下文？
- DP + TP、TP + PP、TP + EP、TP + CP/DCP 分别解决什么瓶颈？
- 哪些组合只适合 MoE，哪些只适合长上下文？
- 哪些组合会放大通信、拓扑和 SLO 长尾？

对应笔记：

- `notes/2026-07-03-parallel-composition-matrix.md`

目标：

- 理解 KV Cache 是推理系统的一等资源。
- 学习 GPU 显存、CPU DRAM、SSD、远端存储之间的缓存层次。
- 对比 HiCache、LMCache、Mooncake 的定位。

关键问题：

- KV Cache 什么时候应该留在 GPU？
- 什么时候应该 offload 到 CPU / SSD / remote storage？
- 缓存复用节省的是 Prefill 计算，还是 Decode 读取？
- 缓存命中和 SLO 冲突时，应该优先命中还是优先低延迟？

候选阅读材料：

- SGLang HiCache 文档。
- LMCache 文档和论文。
- Mooncake 论文。
- vLLM Prefix Cache / KV Cache 相关文档。

### 阶段 3：Context Parallel 和长上下文

目标：

- 理解超长上下文下，KV Cache 容量和 Attention 计算如何成为瓶颈。
- 区分 Context Parallel、Sequence Parallel、Decode Context Parallel。
- 判断 CP 什么时候值得用，什么时候通信成本过高。

关键问题：

- context 并行切的是序列维度的哪一部分？
- Prefill 和 Decode 中 context 并行的通信模式是否不同？
- CP 与 KV Cache offload / cache reuse 是互补还是替代？

### 阶段 4：KV Router

### 阶段 4a：Speculative Decoding / MTP

目标：

- 理解 MTP、draft-model speculative decoding、tree-based speculative decoding 的区别。
- 跟踪 DSpark / DeepSpec 等新技术。
- 判断 speculative decoding 如何影响 Scheduler、KV Cache、TPOT、TPM 和 SLO。

关键问题：

- drafter 成本和 acceptance rate 如何共同决定真实收益？
- MTP 是模型原生能力，还是服务侧推理策略？
- rejected token 的 KV Cache 如何处理？
- TP / PP / EP / CP 组合下 speculative decoding 会新增什么通信或调度问题？
- 它是否能帮助 PD 调度提前预测 worker 释放时间？

对应文档：

- `docs/speculative-decoding-watchlist.md`

### 阶段 5：KV Router

目标：

- 理解路由不只是负载均衡，还要感知缓存位置和缓存价值。
- 在 SLO 条件下最大化 KV Cache 命中。

关键问题：

- 路由器如何知道某个 worker 上有哪些 KV Cache？
- cache hit、queue length、GPU memory、decode saturation 之间如何权衡？
- 为了缓存命中把请求路由到忙 worker，是否会破坏 SLO？
- 如何定义“有效缓存命中”：命中了但排队太久是否仍然有价值？

### 阶段 6：PD 调度

目标：

- 理解 Prefill / Decode 分离后的资源池设计。
- 学习 PD 调度如何影响 TTFT、TPOT、TPM 和 GPU 利用率。
- 学习 KV Cache 在 Prefill worker 和 Decode worker 之间如何传输。

关键问题：

- Prefill worker 和 Decode worker 的配比如何决定？
- 如何提前预判 Decode worker 什么时候结束？
- KV 传输成本如何进入调度决策？
- 什么时候应该为了拓扑亲和牺牲局部负载均衡？

### 阶段 7：PD 容错和拓扑亲和

目标：

- 从“单次请求能跑通”进入“在线服务能稳定跑”。
- 处理 worker 失败、KV 传输失败、网络抖动和节点拓扑差异。

关键问题：

- Prefill 完成但 Decode worker 失败时，KV Cache 是否还能复用？
- Decode 中途失败，能否恢复还是必须重算？
- KV Cache 是否需要副本？
- 拓扑亲和如何影响 KV 传输延迟？
- 跨机、跨 rack、跨 IB 域的调度成本是否应该进入 scheduler score？

## 与当前月度路线的关系

这些要点不是新开八条线，而是重排成一条递进路线：

```text
vLLM Scheduler
  -> KV Cache 管理
  -> Context Parallel / 长上下文
  -> Speculative Decoding / MTP
  -> KV Router
  -> PD 调度
  -> PD 容错和拓扑亲和
  -> SLO 下提升 GPU 利用率和 TPM
```

实际部署优化是贯穿线，不单独等到最后再学。每个主题都要回到：

```text
给定现有 GPU、模型、workload 和 SLO，
当前方案是否提升了满足 SLO 的 goodput？
```

对应文档：

- `docs/deployment-optimization-slo-playbook.md`
- `docs/mixed-long-context-serving-playbook.md`
- `docs/model-specific-vllm-customization-playbook.md`
- `docs/sglang-feature-integration-image-playbook.md`
- `docs/inference-failure-troubleshooting-playbook.md`
- `docs/distributed-training-troubleshooting-playbook.md`
- `docs/lora-multilora-serving-routing-playbook.md`

## 后续产物

每个主题正式学习前，都先建立阅读材料清单：

- `readings/YYYY-WW-kv-cache-management-reading-list.md`
- `readings/YYYY-WW-context-parallel-reading-list.md`
- `readings/YYYY-WW-kv-router-reading-list.md`
- `readings/YYYY-WW-pd-scheduling-reading-list.md`
- `readings/YYYY-WW-pd-fault-tolerance-topology-reading-list.md`

每个主题学习后，至少产出：

- 一份阅读摘记。
- 一份核心问题清单。
- 一张系统图或调度流程图。
- 一个可验证的 Benchmark / Profile 问题。
