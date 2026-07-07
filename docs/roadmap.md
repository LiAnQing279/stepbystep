# 长期路线图

这条路线的目标不是成为一个只会部署 vLLM 的工程师，而是成长为 AI Inference System / Agent System Engineer，并逐步走向 AI Infra Architect。

## 当前主线判断

下一阶段不再以“横向补更多技能”为主，而是围绕推理系统和 Agent 系统建立更深的系统能力。

已经具备的基础包括：vLLM、SGLang、多机部署、PD 分离、国产卡适配、推理优化和问题排查。它们足够支撑下一阶段的成长。现在最重要的不是继续扩展工具菜单，而是把真实工程经验抽象成系统认知。

### 最高优先级能力

1. **系统建模能力**：从单点问题继续追问“它属于哪一类问题”，并放入请求模式、调度策略、KV Cache、显存、通信、GPU 利用率、吞吐和 TTFT 的因果网络。
2. **提出问题的能力**：不满足于修复一个 Bug，而是追问它为什么反复出现、暴露了什么架构缺陷、别人是否还会踩、能否抽象成更一般的规律。
3. **Agent System 视角**：研究 Agent 负载对推理系统的新要求，包括长生命周期 Session、Memory、Tool Calling、多 Agent、状态恢复、Checkpoint、Scheduling、资源隔离、Cost 和 Reliability。
4. **公开输出能力**：把经验沉淀成别人可以复用的知识，固定采用“问题 → 原因 → 原理 → 排查 → 最佳实践”的结构。
5. **产品思维**：先理解企业、业务和用户为什么需要 PD、Prefix Cache、Disaggregation、Long Context 等能力，再反推系统设计。

### 暂时降低优先级

以下能力不是放弃，而是不作为未来一年的主线投入：

- Rust 新语法。
- C++ 新标准。
- 大量新的 Agent Framework。
- 各类 Prompt 技巧。

这些更新很快，且 AI 能提供较多辅助。未来五年的核心竞争力，应该是持续从真实工程中发现问题、抽象结构、形成系统认知，并把认知沉淀为可复用工程能力。

### 三个长期追问

每次技术学习、问题排查和实验复盘，都要带着三个问题：

1. 这个现象背后的结构是什么？
2. 这一类问题还能统一解释哪些现象？
3. 如果未来 Agent 成为默认负载，这个结构还成立吗？

## 第一阶段：成为优秀的推理工程师

时间：现在到半年。

关键词：把工作做到行业一流水平。

### 核心目标

把日常推理工作从“能跑起来”提升到“能解释、能优化、能排障、能复现、能比较”。

### 能力清单

系统建模：

- 能把单点问题放入完整链路，而不是停留在局部修复。
- 能建立“请求模式 → 调度策略 → KV Cache → 显存 → 通信 → GPU 利用率 → 吞吐 → TTFT / TPOT”的因果图。
- 能判断一个问题主要属于 Scheduler、KV Cache、Communication、Runtime、Hardware 还是业务负载。
- 能从一个真实 Case 抽象出一类可复用的问题模型。

模型部署：

- vLLM
- SGLang
- LMDeploy
- TensorRT-LLM

每个框架都要做到：

- 会部署。
- 会排查。
- 会 Benchmark。
- 会调参数。
- 会分析 Profile。
- 能解释性能变化背后的系统原因。

分布式推理：

- DP
- TP
- EP
- PP
- SP
- CP
- DCP
- PD Disaggregation
- KV-aware Routing
- Topology-aware Scheduling

每一种并行方式都要回答：

- 为什么要用？
- 什么时候用？
- 为什么不用？
- 它牺牲了什么？
- 它依赖什么硬件和网络条件？

在线推理调度：

- 如何提高 GPU 利用率和 TPM。
- 如何在 SLO 条件下提高 KV Cache 命中。
- 如何做 Prefill / Decode 分离调度。
- 如何提前预判 worker 释放时间。
- 如何处理 PD 容错。
- 如何结合 NVLink、PCIe、IB/RDMA 做拓扑亲和。

模型理解：

- Dense
- MoE
- MLA
- Linear Attention
- 未来的新结构

目标是看到一个新模型时，能初步判断：

- 计算瓶颈在哪里。
- 显存瓶颈在哪里。
- 通信瓶颈在哪里。
- KV Cache 压力在哪里。
- Prefill 和 Decode 哪个阶段更难。
- 当前主流框架是否适合部署它。

### 阶段产出

- 一份常用推理框架部署和排障手册。
- 一套标准 Benchmark 流程。
- 一份分布式推理并行策略对照表。
- 一张“请求模式到 SLO 指标”的推理系统因果图。
- 至少 3 篇模型部署瓶颈分析笔记。

## 第二阶段：建立自己的推理知识体系

时间：半年到一年。

关键词：从工具列表变成系统地图。

### 核心目标

不要把知识组织成：

```text
vLLM
FlashInfer
DeepEP
Mooncake
LMCache
```

而要组织成：

```text
模型
Kernel
Runtime
Scheduler
Cluster
Hardware
```

以后任何一个新技术，都先放进自己的知识地图。

### 归类示例

- DeepEP：Communication。
- FlashInfer：Kernel。
- Mooncake：KV Cache。
- LMCache：KV Cache / Cache Offloading。
- HiCache：KV Cache / Hierarchical Cache。
- KV Router：Scheduler / KV Cache / Deployment。
- TileRT：Compiler。
- vLLM：Runtime Engine / Scheduler。
- SGLang：Runtime Engine / Programming Interface / Scheduler。
- TensorRT-LLM：Runtime Engine / Compiler / Kernel Integration。

### 阶段产出

- 一张稳定维护的推理系统知识地图。
- 一份高频工程问题分类法：KV Cache、Scheduler、PD、RDMA、NCCL、NUMA、Prefill、Decode 等。
- 一套论文阅读分类模板。
- 一份推理技术名词索引。
- 每月至少 2 篇系统化调研笔记。

## 第二阶段补充：建立 Agent System 视角

时间：贯穿第二阶段到第四阶段。

关键词：把 Agent 作为未来默认负载来审视推理系统。

### 核心问题

不要只问“如何写 Agent”，而要问：

```text
如果同时跑一万个长生命周期 Agent，现在的推理框架哪里会坏？
```

### 重点观察对象

- 长生命周期 Session 如何改变 KV Cache、Prefix Cache 和调度策略。
- Memory 与 Tool Calling 如何改变请求模式和状态管理。
- 多 Agent 协作如何改变并发、隔离、可靠性和成本模型。
- 状态恢复、Checkpoint 和失败重试如何进入推理 runtime。
- Agent 负载下的 Scheduling 是否仍然能按传统 Prefill / Decode 方式优化。
- 资源隔离、租户隔离和成本控制如何影响系统设计。

### 阶段产出

- 一份 Agent 负载画像：请求长度、会话时长、工具调用频率、状态大小、失败模式。
- 一篇“Agent 对推理系统提出哪些新需求”的系统化笔记。
- 至少 2 个面向 Agent 负载的 Benchmark 场景设计。

## 第三阶段：深入源码，形成系统能力

时间：一年左右。

关键词：知道它为什么这么写。

### 阅读层次

第一层：

- vLLM
- SGLang

第二层：

- FlashInfer
- DeepEP
- LMCache

第三层：

- Triton
- Cutlass
- CUDA

第四层：

- NCCL
- NVSHMEM
- RDMA

### 源码阅读目标

源码阅读不是为了逐行记忆，而是回答：

- 核心抽象是什么？
- 数据流怎么走？
- 控制流怎么调度？
- 哪些地方决定性能上限？
- 哪些设计是为了兼容性？
- 哪些设计是为了工程稳定性？
- 如果换一种模型、硬件或并行策略，这段代码会遇到什么限制？

### 阶段产出

- vLLM 和 SGLang 的核心架构图。
- 至少 5 篇源码阅读笔记。
- 至少 2 个可复现的性能问题分析。
- 至少 1 个向开源项目提交的 Issue、PR 或实验复现报告。

## 第四阶段：开始做自己的东西

时间：一年以后开始持续推进。

关键词：从学习者变成建设者。

### 核心转变

不要停留在：

```text
看论文 -> 做实验 -> 结束
```

而要变成：

```text
看论文 -> 理解思想 -> 自己实现 -> 写博客 -> 开源 -> 别人使用
```

### 可以做的方向

推理优化实验：

- DeepEP 实验。
- PD 分离实验。
- PD 调度和容错实验。
- KV Router 实验。
- KV Cache 实验。
- HiCache / LMCache / Mooncake 对比实验。
- Context Parallel 长上下文实验。
- 国产卡实验。
- 多机多卡稳定性实验。

Benchmark 工具：

- 自动比较 vLLM、SGLang、TensorRT-LLM。
- 支持 Prefill、Decode、混合负载和长上下文场景。
- 支持吞吐、延迟、显存、GPU 利用率和网络指标采集。
- 支持 SLO 条件下的 TPM、TTFT、TPOT、KV Cache 命中率和 miss penalty 分析。
- 支持 PD 分离场景下的 KV 传输时间、worker 空转时间、拓扑亲和收益分析。
- 支持 Agent 负载场景：长 Session、多轮工具调用、状态恢复、资源隔离和成本统计。

产品化问题研究：

- 企业为什么需要 PD 分离，而不是继续堆单体推理实例。
- 用户为什么需要 Long Context、Prefix Cache 和 Disaggregation。
- 不同行业负载对 TTFT、TPOT、成本、可靠性和隔离性的真实优先级。
- 技术能力如何转化成可购买、可运维、可解释的产品能力。

国产卡适配记录：

- 昇腾。
- 寒武纪。
- 沐曦。
- 天数。

记录重点：

- 框架适配状态。
- 算子缺口。
- 通信能力。
- 编译链问题。
- 性能差距。
- 排障路径。

### 阶段产出

- 一个公开 Benchmark 项目。
- 一个推理优化实验仓库。
- 一组国产卡部署和适配笔记。
- 一套可以复用的实验报告模板。

## 第五阶段：形成行业影响力

时间：长期持续。

关键词：知识输出。

### Inference Weekly

每周稳定输出：

- 推理论文。
- 推理优化。
- 新模型部署。
- 国产卡适配。
- 源码分析。
- 行业动态和工程判断。
- 一个产品需求反推系统设计的案例。

目标不是追热点，而是形成连续的判断能力。坚持两年后，别人会开始引用这套内容。

### AI Inference Handbook

可以逐步整理成一本推理工程参考手册：

- 模型。
- Kernel。
- KV Cache。
- Scheduler。
- Distributed。
- Network。
- Compiler。
- Deployment。
- Benchmark。

### 阶段产出

- 稳定更新的 `Inference Weekly`。
- 持续演化的 `AI Inference Handbook`。
- 公开演讲、技术分享或系列文章。
- 能被团队或行业复用的方法论。

## 总体判断标准

三到五年后，真正重要的不是熟悉某一个框架，而是拥有一套能迁移到任何新模型、新硬件、新框架上的推理系统能力。

更具体地说，目标不是“会更多技术”，而是当别人想到 KV Cache、PD、Agent 负载下的推理调度、多机通信或国产卡适配时，会想到你建立过一套清晰、可验证、可复用的理解。
