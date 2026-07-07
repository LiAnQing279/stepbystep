# 个人知识路径

这份文档把长期路线图拆成日常可执行的学习系统。核心原则是：工作、论文、源码、实验和输出必须互相喂养，而不是五条互不相干的线。

## 总原则

每天做工程，不能只留下“问题解决了”。要留下：

- 问题属于哪一层？
- 根因是什么？
- 解决方案为什么有效？
- 有没有更一般的规律？
- 这个经验能否写成文档、实验或工具？

下一阶段的主线不是横向补更多技能，而是把真实工程经验加工成系统建模能力、提出问题的能力、Agent System 视角、公开输出能力和产品思维。

### 三个固定追问

每次遇到技术现象、Bug、性能波动或新框架特性，都先问：

1. 这个现象背后的结构是什么？
2. 这一类问题还能统一解释哪些现象？
3. 如果未来 Agent 成为默认负载，这个结构还成立吗？

### 暂时不追的方向

除非工作直接需要，否则未来一段时间降低以下方向的投入：

- Rust 新语法。
- C++ 新标准。
- 大量新的 Agent Framework。
- 各类 Prompt 技巧。

这些内容可以借助 AI 按需补齐。日常学习的主时间，应继续投向推理系统建模、真实问题抽象、Agent 负载理解和可复用输出。

## 每周节奏

一周建议分成七类活动。顺序很重要：先收集材料和形成问题，再进入阅读、源码、实验、输出和产品需求反推。

### 1. 阅读材料收集

每周正式学习之前，先建立一份阅读材料清单，放入 `readings/`。

材料清单至少包含：

- 官方文档。
- 关键论文或技术报告。
- 源码入口。
- 相关博客或架构文章。
- 版本或 commit 信息。
- 阅读顺序。
- 阅读前问题。
- 阅读后要产出的笔记。

材料收集的目标不是囤链接，而是先给学习设边界：

- 本周只读哪些材料？
- 哪些材料是必读？
- 哪些只是对照或选读？
- 读完后要回答什么问题？
- 什么时候才进入源码阅读或实验？

每份材料都要先判断：

- 它属于系统地图哪一层？
- 它可能解决什么瓶颈？
- 它和本周主线是什么关系？
- 它是否足够权威或接近一手来源？

### 2. 工作问题沉淀

记录本周遇到的部署、性能、稳定性和排障问题。

每个问题至少写清楚：

- 现象。
- 环境。
- 复现路径。
- 初步假设。
- 验证方法。
- 根因。
- 解决方案。
- 归属层级。
- 问题类别：KV Cache、Scheduler、PD、RDMA、NCCL、NUMA、Prefill、Decode、Runtime、Hardware、业务负载。
- 因果链路：请求模式 → 调度策略 → KV Cache → 显存 → 通信 → GPU 利用率 → 吞吐 → TTFT / TPOT。
- 是否暴露架构缺陷。
- 是否会在别人那里重复出现。
- 是否值得公开输出。

每周至少选择 1 个 Case，额外写成“一类问题”的形式：

```text
个案现象 → 共性模式 → 触发条件 → 影响范围 → 排查路径 → 系统性改进
```

### 3. 论文和技术调研

每周选择 1 到 3 个主题，不求多，求放进地图。

阅读后必须回答：

- 它解决什么瓶颈？
- 它改了模型、Kernel、Runtime、Scheduler、Cluster 还是 Hardware？
- 它为什么在当前阶段出现？
- 它依赖哪些硬件、数据规模或业务形态？
- 它的实验是否能复现？
- 它对应什么真实产品需求？
- 如果负载从 Chat Completion 变成长期运行的 Agent，它还成立吗？

### 4. 源码阅读

源码阅读按问题驱动，不做漫无目的的逐行阅读。

进入源码阅读前，必须先有对应的阅读材料清单，并且已经写下阅读问题。

推荐路径：

- 先从 vLLM / SGLang 的请求生命周期读起。
- 再读 Scheduler、KV Cache、Worker、Distributed Executor。
- 遇到性能热点，再下探到 FlashInfer、Triton、CUDA。
- 遇到跨节点瓶颈，再看 NCCL、DeepEP、RDMA。

### vLLM 源码入口指引

```text
请求入口：
  vllm/entrypoints/openai/api_server.py
    → AsyncLLMEngine.generate()

核心引擎：
  vllm/engine/async_llm_engine.py   # 异步引擎主循环
  vllm/engine/llm_engine.py         # 同步引擎，核心逻辑

调度器：
  vllm/core/scheduler.py            # Scheduler.schedule() — 决定哪些请求进入本次 batch
  vllm/core/block_manager.py        # KV Cache block 分配和释放

Worker 执行：
  vllm/worker/worker.py             # Worker.execute_model() — 单卡执行
  vllm/executor/                    # 分布式执行器（TP/PP 的组织）

关键链路：
  add_request() → step() → schedule() → execute_model() → sample() → 返回 token
```

### SGLang 源码入口指引

```text
请求入口：
  python/sglang/srt/server.py       # HTTP 服务入口

核心调度：
  python/sglang/srt/managers/scheduler.py         # 主调度循环
  python/sglang/srt/managers/tp_worker.py         # TP Worker

RadixAttention / KV Cache：
  python/sglang/srt/layers/radix_attention.py     # RadixAttention 实现
  python/sglang/srt/mem_cache/radix_cache.py      # 基于 Radix Tree 的 KV Cache 复用

关键链路：
  HTTP request → TokenizerManager → Scheduler → TpWorker → ModelRunner → 返回 token
```

每次源码阅读输出：

- 一张调用链。
- 一个核心抽象解释。
- 一个性能或稳定性判断。
- 一个未来可能验证的实验。

### 5. 实验和 Benchmark

实验不要只比较一个数字，要解释数字为什么变。

实验报告至少包含：

- 实验目标。
- 环境配置。
- 模型和数据。
- 参数设置。
- 指标定义。
- 结果表格。
- Profile 证据。
- 结论。
- 局限性。
- 下一步。

重点实验方向：

- vLLM vs SGLang vs TensorRT-LLM。
- Prefill / Decode 分离。
- Chunked Prefill。
- 长上下文 KV Cache。
- MoE EP / DeepEP。
- 多机多卡通信。
- 国产卡部署。
- Agent 负载：长生命周期 Session、多轮工具调用、状态恢复、资源隔离、成本和可靠性。

### 6. 对外输出

每周至少沉淀一个可以对外输出的主题，即使暂时不发布。

输出形式：

- 技术短文。
- 源码阅读笔记。
- Benchmark 报告。
- 部署踩坑记录。
- 论文解读。
- 架构图。
- Issue / PR。

推荐固定结构：

```text
问题 → 原因 → 原理 → 排查 → 最佳实践
```

公开输出的目标不是证明自己“知道很多”，而是让别人能复用你的判断、排查路径和系统模型。优先沉淀 PD 分离、KV Cache、国产卡适配、多机场景、NCCL、RDMA、vLLM 和 SGLang 等已经有真实经验的主题。

### 7. 产品需求反推

每周选择一个技术能力，从用户和业务需求反推它为什么重要。

可选主题：

- 为什么企业需要 PD 分离。
- 为什么企业需要 Prefix Cache。
- 为什么企业需要 Disaggregation。
- 为什么企业需要 Long Context。
- 为什么 Agent 负载需要状态恢复、资源隔离和成本控制。

每次至少回答：

- 谁需要这个能力？
- 它解决什么业务痛点？
- 没有它会出现什么成本、稳定性或体验问题？
- 这个需求如何改变 Scheduler、KV Cache、通信、部署或监控设计？

## 月度主题

每个月选择一个主线，避免所有方向同时展开。

示例：

- 第 1 月（7 月）：vLLM 请求生命周期和 Scheduler。← 当前
- 第 2 月（8 月）：KV Cache 管理：PagedAttention、Prefix Cache、HiCache、LMCache、Mooncake。
- 第 3 月（9 月）：Context Parallel、长上下文和 KV Cache 压力。
- 第 4 月（10 月）：Speculative Decoding / MTP：DSpark、DeepSpec、EAGLE、Medusa，以及 Decode 加速。
- 第 5 月（11 月）：KV Router：缓存感知路由，以及 SLO 条件下提高 KV 命中。
- 第 6 月（12 月）：LoRA / 多 LoRA 部署与路由：adapter cache、multi-LoRA batching、多租户 SLO。
- 第 7 月（1 月）：PD 分离和在线调度：Prefill / Decode 配比、worker 结束预测、TPM 和 GPU 利用率。
- 第 8 月（2 月）：PD 容错和拓扑亲和：KV 传输失败、worker 失败、NVLink/PCIe/IB/RDMA 亲和调度。
- 第 9 月（3 月）：MoE 推理与 EP / DeepEP。
- 第 10 月（4 月）：Agent 负载下的推理系统：Session、Memory、Tool Calling、状态恢复、资源隔离和成本。
- 第 11 月（5 月）：国产 GPU 部署适配。
- 第 12 月（6 月）：TensorRT-LLM、编译优化和必要的 Kernel 基础。

贯穿线：实际部署优化与 SLO 下 goodput 最大化，维护在 `docs/deployment-optimization-slo-playbook.md`。

旁线：多机训练问题与排障，维护在 `docs/distributed-training-troubleshooting-playbook.md`。这条线按需学习，重点理解训练产物、NCCL/RDMA、checkpoint 和多机故障如何影响推理部署。

导师补充主题统一维护在 `docs/mentor-topics-backlog.md`。Speculative Decoding / MTP 观察线维护在 `docs/speculative-decoding-watchlist.md`。正式学习前先为每个主题建立对应的 `readings/` 材料清单。

## 能力成长刻度

### L1：能跑

- 能部署模型。
- 能按文档启动服务。
- 能跑简单 Benchmark。

### L2：能排障

- 能定位环境、模型、框架、显存、通信和参数问题。
- 能读日志。
- 能复现线上问题。
- 能按层定位推理失败：请求、tokenizer、scheduler、KV Cache、model forward、kernel、NCCL、基础设施。
- 能理解多机训练中的数据、数值、并行、通信、checkpoint 问题，并判断它们是否影响推理部署。

### L3：能优化

- 能通过 Benchmark 和 Profile 找瓶颈。
- 能调 batch、并行策略、KV Cache、Prefill、Decode 参数。
- 能解释吞吐和延迟变化。
- 能在 SLO 条件下比较 goodput，而不是只看裸吞吐。

### L4：能设计

- 能根据模型结构、业务负载和硬件条件设计部署方案。
- 能判断用 vLLM、SGLang、TensorRT-LLM 还是其他方案。
- 能做容量规划和稳定性设计。
- 能为现有 GPU 选择并行拓扑、参数组合、路由策略和扩容阈值。
- 能设计监控、告警、runbook 和故障止血策略。
- 能从用户需求反推系统能力，而不是只从技术特性出发。
- 能判断 Agent 负载是否会打破现有推理系统假设。

### L5：能建设

- 能实现工具、实验框架、插件或优化方案。
- 能向开源项目贡献。
- 能让别人复用你的方法和代码。
- 能针对特定模型在 vLLM 中实现 model adapter、custom op、attention backend、KV 或 scheduler 优化。
- 能把论文新特性接入 SGLang / vLLM serving runtime，构建自定义镜像并完成正确性、性能和 SLO goodput 验证。
- 能把一个真实 Case 抽象成一类问题，并沉淀为模板、Playbook、Benchmark 或设计建议。

### L6：能影响

- 能持续输出行业级内容。
- 能提出框架选型、硬件选型和系统架构建议。
- 能形成自己的 Inference Weekly 和 AI Inference Handbook。
- 能让别人想到 KV Cache、PD、Agent 负载推理、多机通信或国产卡适配时，联想到你建立的系统理解。

## 近期 90 天路线

起始日期：2026-07-01。

### 第 1 到 30 天（07-01 至 07-30）：打牢当前工作闭环

目标：

- 建立标准 Benchmark 模板。
- 梳理 vLLM / SGLang 部署和排障清单。
- 记录至少 5 个真实问题。
- 为每个真实问题标注问题类别和因果链路。

产出：

- 一份部署排障手册草稿。
- 一份 Benchmark 指标表。
- 一篇“如何判断推理瓶颈”的笔记。
- 一张“请求模式 → 调度 → KV Cache → 显存 → 通信 → GPU 利用率 → SLO”的因果图。

### 第 31 到 60 天（07-31 至 08-29）：建立系统地图

目标：

- 把已知技术全部归类到知识地图。
- 重点补 KV Cache、Scheduler、Parallel、Communication。
- 开始源码阅读 vLLM 请求生命周期。
- 从至少 2 个真实 Case 抽象出一类问题。

产出：

- 一张推理系统地图。
- 两篇源码阅读笔记。
- 一篇 KV Cache 或 Scheduler 主题文章。
- 一篇“问题 → 原因 → 原理 → 排查 → 最佳实践”结构的公开输出草稿。

### 第 61 到 90 天（08-30 至 09-28）：做第一个可复现实验

目标：

- 选择一个明确主题，例如 PD 分离、长上下文 KV Cache 或 vLLM vs SGLang。
- 做完整实验。
- 输出报告。
- 额外加入一个 Agent 负载视角：这个实验结论在长 Session、多轮工具调用或状态恢复场景下是否仍成立。

产出：

- 一个实验仓库或实验目录。
- 一份可复现 Benchmark 报告。
- 一篇可公开发布的技术文章。
- 一份 Agent 负载场景假设和后续实验计划。

## 长期作品线

### Inference Weekly

定位：每周一份推理系统观察。

固定栏目：

- 本周论文。
- 本周框架更新。
- 本周模型部署。
- 本周源码片段。
- 本周实验。
- 一个系统判断。
- 一个产品需求反推。
- 一个 Agent 负载追问。

### AI Inference Handbook

定位：长期整理成推理工程参考手册。

目录方向：

- 模型结构。
- Attention 和 MLP。
- Kernel。
- KV Cache。
- Scheduler。
- Distributed Inference。
- Communication。
- Deployment。
- Benchmark。
- Hardware。
- 国产卡适配。

## 反复提醒

真正要培养的能力不是记住更多工具，而是建立一张推理系统地图。

有了这张地图，每一个新技术都不是孤立知识点，而是在补全系统能力。

更进一步，真正稀缺的是：能够不断从真实工程中发现问题、抽象结构、形成系统认知，并把这些认知沉淀为别人可以复用的工程能力。
