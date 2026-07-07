# 2026-W27 进度和过程

周期：2026-06-29 到 2026-07-05

本周主线：建立推理系统学习闭环。

## 当前状态

- 项目已经具备规划层、积累层和模板层。
- 本周 TODO 已建立：[2026-W27-todo.md](2026-W27-todo.md)。
- 当前最重要的动作：把第一批学习内容写进 `notes/`、`readings/` 或 `experiments/`，不要只停留在路线图。

## 过程记录规则

每次学习后，用 5 到 10 行记录即可：

- 学了什么。
- 它属于系统地图哪一层。
- 解决什么瓶颈。
- 当前理解到什么程度。
- 下一步要验证什么。

## 每日记录

### 2026-07-01

已完成：

- 建立本周 TODO 文档。
- 建立本周进度和过程文档。
- 在 README 增加本周入口。

过程判断：

- 当前目录结构已经够用，短期不需要继续细分太多目录。
- 现在更重要的是形成稳定写入习惯：`notes/` 记录理解，`readings/` 记录论文和源码，`experiments/` 记录实验，`weekly-reviews/` 记录计划和复盘。

下一步：

- 通读推理系统知识地图。
- 标出 3 个薄弱层级。
- 产出第一篇系统地图笔记。

### 2026-07-02

已完成：

- 建立分布式推理并行策略对照表：DP / TP / EP / PP / SP / CP / DCP。

过程判断：

- 并行策略不能只记全称，要和“切什么、解决什么、牺牲什么、通信模式”绑定。
- TP 是单机推理最常见入口；EP 主要服务 MoE；CP / DCP 主要在长上下文和 KV Cache 压力下变重要。

下一步：

- 后续 Benchmark 时，把并行策略和通信证据绑定，而不是只做概念对照。

### 2026-07-03

已完成：

- 通读推理系统知识地图，整理首轮笔记：`notes/2026-07-01-inference-system-map-first-pass.md`。
- 标出当前最薄弱的 3 个层级：Scheduler / Deployment、KV Cache、Communication。
- 选择 vLLM 作为本周源码主线。
- 画出 vLLM 请求生命周期高层链路：`notes/2026-07-03-vllm-request-lifecycle.md`。
- 补充 vLLM 资源竞争排查清单：`notes/2026-07-03-vllm-resource-contention-checklist.md`。
- 补充分布式推理并行组合矩阵：`notes/2026-07-03-parallel-composition-matrix.md`。
- 补充 Benchmark / Profile 检查清单：`notes/2026-07-03-benchmark-profile-checklist.md`。
- 补充部署优化与 SLO 下最大效能 playbook：`docs/deployment-optimization-slo-playbook.md`。
- 补充不规则超长上下文与短请求混合部署 playbook：`docs/mixed-long-context-serving-playbook.md`。
- 补充特定模型 vLLM 自定义实现 playbook：`docs/model-specific-vllm-customization-playbook.md`。
- 补充 SGLang 新特性镜像集成 playbook：`docs/sglang-feature-integration-image-playbook.md`。
- 补充推理失败故障排查 playbook：`docs/inference-failure-troubleshooting-playbook.md`。
- 补充多机训练问题与排障 playbook：`docs/distributed-training-troubleshooting-playbook.md`。
- 补充 LoRA / 多 LoRA 部署与路由 playbook：`docs/lora-multilora-serving-routing-playbook.md`。
- 记录一个学习问题：Benchmark 数字差异如何归因。
- 完成 W27 周复盘：`weekly-reviews/2026-W27-review.md`。

过程判断：

- vLLM 请求生命周期的核心不是 API Server，而是 Scheduler、KV Cache 和 Worker Execute。
- 后续看 vLLM 不能只看调用链，还要看资源竞争链：CPU、GPU compute、HBM capacity、HBM bandwidth、KV blocks、通信、存储和 SLO budget。
- 后续看并行不能只看单个名词，还要看组合关系：DP 是请求副本，TP/PP/CP 切同一个请求，EP 只对 MoE expert 有意义。
- Benchmark 不能只看 total tokens/s，必须拆成 prompt tokens/s、output tokens/s、TTFT、TPOT、显存和 GPU 利用率。
- 实际部署优化不能只追求裸吞吐，应该看满足 SLO 的 goodput。
- 长短混合不是普通吞吐问题，低并发也可能因为长 prefill、KV block、HBM bandwidth 和 batch shape 变差而高延迟。
- 特定模型优化要先有 Profile 归因，再选择最小自定义实现点，不能一上来就 fork 或写 kernel。
- 论文新特性工程化不是把代码装进镜像，而是接入 runtime 的请求、调度、KV、模型执行、metrics 和回退体系。
- 推理失败排查要先止血和保留证据，再按层定位到请求、API、Scheduler、KV、model forward、kernel、通信或基础设施。
- 训练系统是旁线但必须理解，尤其是 tokenizer、checkpoint、多机通信和数值问题如何传导到推理部署。
- 多 LoRA serving 不是简单加载多个 adapter，而是 adapter cache、batching、router、多租户和 SLO 的系统问题。
- 没有 Profile 证据的性能对比，很难沉淀成系统判断。

下一步：

- 下周聚焦 vLLM Scheduler。
- 学习前先阅读 `readings/2026-W28-vllm-scheduler-reading-list.md`。
- 读完材料后再回答：一个请求从 waiting 进入 running，需要满足哪些调度条件？

## 本周决策

- 每周使用两个文档推进：TODO 管目标，progress 管过程。
- 学习记录优先按“系统地图层级”组织，而不是按工具名称堆叠。
- 本周只选一个源码主线：vLLM，避免同时展开。
- Benchmark / Profile 检查清单先面向单机单模型，后续再扩展到多卡和多机。
- 下周主线确定为 vLLM Scheduler，但进入正式学习前先做阅读材料收集和阅读摘记。

## 卡点

- 暂无。

## 待确认问题

- vLLM Scheduler 源码阅读从哪个版本开始，以实际工作环境版本为准。
- 是否有可用 A100/A800 环境跑最小 Benchmark。

## 完成项

- [x] 建立本周 TODO。
- [x] 建立本周进度和过程记录。
- [x] README 增加本周入口。
- [x] 系统地图首轮通读笔记。
- [x] 分布式推理并行策略对照表。
- [x] vLLM 请求生命周期高层链路。
- [x] vLLM 资源竞争排查清单。
- [x] 分布式推理并行组合矩阵。
- [x] Benchmark / Profile 检查清单草稿。
- [x] 部署优化与 SLO playbook。
- [x] 长短混合上下文部署 playbook。
- [x] 特定模型 vLLM 自定义实现 playbook。
- [x] SGLang 新特性镜像集成 playbook。
- [x] 推理失败故障排查 playbook。
- [x] 多机训练问题与排障 playbook。
- [x] LoRA / 多 LoRA 部署与路由 playbook。
- [x] Benchmark 归因学习问题沉淀。
- [x] W27 周复盘。
- [x] 为下周 vLLM Scheduler 建立阅读材料清单。
