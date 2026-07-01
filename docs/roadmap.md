# 长期路线图

这条路线的目标不是成为一个只会部署 vLLM 的工程师，而是成长为 AI Inference System Engineer，并逐步走向 AI Infra Architect。

## 第一阶段：成为优秀的推理工程师

时间：现在到半年。

关键词：把工作做到行业一流水平。

### 核心目标

把日常推理工作从“能跑起来”提升到“能解释、能优化、能排障、能复现、能比较”。

### 能力清单

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

每一种并行方式都要回答：

- 为什么要用？
- 什么时候用？
- 为什么不用？
- 它牺牲了什么？
- 它依赖什么硬件和网络条件？

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
- TileRT：Compiler。
- vLLM：Runtime Engine / Scheduler。
- SGLang：Runtime Engine / Programming Interface / Scheduler。
- TensorRT-LLM：Runtime Engine / Compiler / Kernel Integration。

### 阶段产出

- 一张稳定维护的推理系统知识地图。
- 一套论文阅读分类模板。
- 一份推理技术名词索引。
- 每月至少 2 篇系统化调研笔记。

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
- KV Cache 实验。
- 国产卡实验。
- 多机多卡稳定性实验。

Benchmark 工具：

- 自动比较 vLLM、SGLang、TensorRT-LLM。
- 支持 Prefill、Decode、混合负载和长上下文场景。
- 支持吞吐、延迟、显存、GPU 利用率和网络指标采集。

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
