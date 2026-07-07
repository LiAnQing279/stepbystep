# 2026-W28 阅读材料：vLLM Scheduler

周期：2026-07-06 到 2026-07-12

## 阅读目标

在正式学习 vLLM Scheduler 之前，先建立材料池和问题意识。

本周阅读不追求一次性读完源码，而是先回答：

- vLLM 的请求调度解决什么问题？
- Scheduler 如何影响 TTFT、TPOT、吞吐和显存？
- Prefill、Decode、KV Cache、Continuous Batching 如何连在一起？
- 哪些进程/组件会竞争 CPU、GPU、显存、KV block、通信和调度预算？
- 不同资源竞争会表现成哪些排障症状？
- vLLM Scheduler 涉及哪些进程、线程或循环？
- GPU KV block 是在哪里、按什么约束被分配和释放的？
- API Server、Engine Core、GPU Worker 和多卡 Worker 之间的通信过程是什么？
- 哪些位置最容易成为瓶颈？
- Scheduler / KV Cache / Worker 异常失败时会留下哪些日志、metrics 和状态信号？
- 当前请求是在一个 DP 副本内处理，还是被 TP / PP / EP / CP / DCP 共同切分？
- 并行组合会如何改变 Scheduler 的资源约束和排查路径？
- Speculative Decoding / MTP 会不会改变 Scheduler 对 decode step、batch shape 和 worker 释放时间的判断？
- 后续读源码时应该从哪些文件进入？

## 必读材料

### 1. vLLM 官方文档：vLLM V1

链接：

- https://docs.vllm.ai/en/latest/usage/v1_guide.html

阅读目的：

- 了解当前 vLLM V1 的整体架构变化。
- 先确认后续学习应该以哪个引擎版本为主。
- 关注 Scheduler、KV Cache、Engine、Worker 之间的关系。

阅读问题：

- V1 相比旧架构，调度和执行链路有什么变化？
- V1 中哪些组件会直接影响请求生命周期？
- 如果后续读源码，应该优先读 V1 还是兼容旧路径？

### 2. vLLM 官方文档：Automatic Prefix Caching

链接：

- https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html

阅读目的：

- 理解 Prefix Cache 与 KV Cache 复用的关系。
- 为后续读 Scheduler 时理解缓存命中、block 管理和请求复用做准备。

阅读问题：

- Prefix Cache 解决的是哪个层级的瓶颈？
- 它主要影响 TTFT、TPOT、吞吐还是显存？
- 哪些业务负载最容易从 Prefix Cache 中受益？

### 3. vLLM 官方文档：Optimization and Tuning

链接：

- https://docs.vllm.ai/en/latest/configuration/optimization.html

阅读目的：

- 从使用者视角理解哪些参数会影响调度、吞吐、延迟和显存。
- 把配置参数和系统地图层级绑定起来。

阅读问题：

- 哪些参数影响 batch 构造？
- 哪些参数影响 KV Cache 使用？
- 哪些参数更偏吞吐优化，哪些更偏延迟优化？

### 4. PagedAttention 论文

链接：

- https://arxiv.org/abs/2309.06180

阅读目的：

- 理解 vLLM 的核心背景：为什么 KV Cache 管理会成为吞吐瓶颈。
- 建立 Scheduler 和 KV Cache 之间的联系。

阅读问题：

- 传统 KV Cache 管理为什么浪费显存？
- PagedAttention 如何把操作系统分页思想迁移到 KV Cache？
- 这个设计为什么能提升并发和吞吐？

## 源码入口

### 1. vLLM GitHub 主仓库

链接：

- https://github.com/vllm-project/vllm

用途：

- 确认当前文件路径。
- 对照官方文档读源码。
- 后续固定一个 commit 或 release，避免源码随 main 分支变化导致笔记漂移。

### 2. Scheduler 源码入口

优先搜索这些文件或目录：

```text
vllm/v1/core/scheduler.py
vllm/core/scheduler.py
vllm/v1/engine/
vllm/engine/
vllm/v1/core/kv_cache_manager.py
vllm/core/block_manager.py
```

阅读目的：

- 找到当前版本的 Scheduler 主入口。
- 看清请求状态流转。
- 看清 Scheduler 和 KV Cache manager 的边界。

阅读问题：

- scheduler 的输入是什么？
- scheduler 的输出是什么？
- 请求有哪些状态？
- 哪些条件会让请求从 waiting 进入 running？
- KV Cache 不够时如何处理？
- 每轮调度如何区分 prefill 和 decode？
- Scheduler 管理的是哪些资源预算：token budget、sequence budget、KV block、单轮执行时间还是 SLO？
- Prefill 请求、Decode 请求、长 prompt、短 prompt 分别在竞争什么？
- 如果资源竞争失败，会表现为 waiting 变长、preemption、swap、OOM，还是 TPOT 抖动？

## 选读材料

### 1. vLLM 博客或架构文章

链接：

- https://blog.vllm.ai/

阅读目的：

- 作为官方文档和论文之外的补充背景。
- 优先读与 vLLM V1、调度、KV Cache、吞吐优化相关的文章。

### 2. SGLang 文档：Runtime / Scheduler / Prefix Cache

链接：

- https://docs.sglang.ai/

阅读目的：

- 只作为对照，不展开主线。
- 观察另一个 Runtime 如何组织调度和缓存。

阅读边界：

- 本周不深入 SGLang 源码。
- 只记录和 vLLM Scheduler 对比时有用的概念。

## 阅读顺序

建议按这个顺序读：

1. vLLM V1 官方文档。
2. Optimization and Tuning。
3. Automatic Prefix Caching。
4. PagedAttention 论文。
5. GitHub 中定位 Scheduler 源码入口。
6. 先读 `notes/2026-07-03-vllm-resource-contention-checklist.md`，建立资源竞争视角。
7. 只读 Scheduler 文件的类名、函数名、注释和主流程，不急着逐行读。

## 阅读时必须记录的内容

每读一份材料，记录 5 到 10 行即可：

```text
材料：
所属层级：
解决的瓶颈：
我现在理解了什么：
我仍然不理解什么：
下一步要问源码/实验的问题：
```

## 读完后再进入学习的问题

读完材料后，正式学习从这些问题开始：

1. Scheduler 每轮调度的输入和输出是什么？
2. waiting 请求进入 running 的条件是什么？
3. Prefill 和 Decode 在同一轮调度里如何竞争资源？
4. KV Cache block 是 Scheduler 的硬约束，还是后置检查？
5. 哪些配置参数会改变调度行为？
6. 如果 TTFT 变差，应该先怀疑 Scheduler 哪一段？
7. 如果 TPOT 变差，应该先怀疑 Scheduler、KV Cache 还是 Kernel？
8. vLLM 中哪些组件可能同时竞争 GPU compute、HBM capacity、HBM bandwidth、CPU、网络和存储？
9. 这些资源竞争分别会导致哪些问题：TTFT 高、TPOT 高、P99 长尾、OOM、GPU 空泡、吞吐抖动还是多卡扩展差？
10. Profile 中哪些证据能证明问题来自资源竞争，而不是单纯代码慢？
11. TP / DP / PP / EP / CP / DCP 分别切分了请求、权重、层、expert、上下文还是 KV？
12. 如果一个服务同时使用 DP + TP + EP 或 TP + PP，Scheduler 看到的“一个请求”和底层实际执行的并行组之间是什么关系？
13. 如果启用 speculative decoding / MTP，一轮 decode 还只是生成一个 token 吗？Scheduler 和 KV Cache Manager 需要额外处理什么？
14. API Server、Engine Core、GPU Worker、DP Coordinator 分别运行在哪些进程里？
15. KVCacheManager 分配 block 的触发条件是什么？block 不够时是 waiting、preempt、recompute 还是 swap？
16. Engine Core 到 GPU Worker 传递了哪些调度结果和 KV metadata？
17. 如果请求 hang、OOM、preempt 过多、worker crash 或输出异常，Scheduler 侧能观察到什么证据？

## 本周产出要求

阅读完成后再开始正式学习，先产出：

- 一份阅读摘记：`readings/2026-W28-vllm-scheduler-notes.md`
- 一份问题清单：`notes/2026-07-vllm-scheduler-questions.md`
- 一份资源竞争标注：读源码时把 API Server、Scheduler、KV Cache Manager、Worker、Distributed Executor 分别管理和竞争的资源标出来。
- 一份并行组合标注：把当前阅读到的 TP / DP / PP / EP / CP / DCP 组合关系和限制条件标出来。
- 一条观察线：记录 vLLM Scheduler 中是否有 speculative decoding / MTP 相关入口，暂不展开实现。
- 一份进程/通信/KV block 标注：对照 `notes/2026-07-03-vllm-scheduler-process-kv-communication-bottlenecks.md`，逐项用源码确认。
- 一份失败模式标注：对照 `docs/inference-failure-troubleshooting-playbook.md`，标出 Scheduler / KV / Worker 各自可能导致什么故障。
- 然后再进入源码阅读笔记。
