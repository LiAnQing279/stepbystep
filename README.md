# AI Inference System Engineer Roadmap

这个项目用于沉淀一条长期成长路径：从会部署模型的工程师，成长为具备系统判断、源码能力、实验能力、产品理解和行业影响力的 AI Inference System / Agent System Engineer，并进一步走向 AI Infra Architect。

这里不追求收集更多工具名，而是建立一张可以迁移的推理系统地图。以后每看到一个新模型、新框架、新论文、新硬件或新优化方案，都先回答三个问题：

1. 它属于推理系统地图的哪一层？
2. 它解决了这一层的什么瓶颈？
3. 为什么现在这个瓶颈变得重要了？

## 文档结构

### 规划层

- [长期路线图](docs/roadmap.md)：五个阶段的成长目标、能力标准和阶段产出。
- [推理系统知识地图](docs/inference-system-map.md)：把模型、Kernel、Runtime、Scheduler、Cluster、Hardware 放进同一张地图。
- [个人知识路径](docs/learning-path.md)：把每天的工作、论文、源码、实验和输出组织成可持续的学习系统。
- [导师要点 Backlog](docs/mentor-topics-backlog.md)：收集导师补充的 Context Parallel、KV Cache、KV Router、PD 调度/容错/拓扑亲和等主题。
- [Speculative Decoding / MTP 观察线](docs/speculative-decoding-watchlist.md)：跟踪 MTP、DSpark、DeepSpec、EAGLE、Medusa 等 Decode 加速方向。
- [部署优化与 SLO Playbook](docs/deployment-optimization-slo-playbook.md)：在现有 GPU 和业务 SLO 下选择部署形态、调参数并最大化 goodput。
- [长短混合上下文部署 Playbook](docs/mixed-long-context-serving-playbook.md)：处理不规则超长上下文和短请求混合、低并发高延迟、P99 污染等复杂场景。
- [特定模型 vLLM 自定义实现 Playbook](docs/model-specific-vllm-customization-playbook.md)：面向特定模型，在 vLLM 中做 model adapter、attention backend、custom op、KV、scheduler 等实现优化。
- [SGLang 新特性镜像集成 Playbook](docs/sglang-feature-integration-image-playbook.md)：把论文新特性如 DSpark/DeepSpec/MTP 接入 SGLang runtime 并构建自定义镜像。
- [推理失败故障排查 Playbook](docs/inference-failure-troubleshooting-playbook.md)：处理启动失败、请求失败、OOM、hang、输出异常、性能退化和多卡通信故障。
- [多机训练问题与排障 Playbook](docs/distributed-training-troubleshooting-playbook.md)：了解训练中的数据、数值、并行、通信、checkpoint、容错问题，以及它们如何影响推理部署。
- [LoRA / 多 LoRA 部署与路由 Playbook](docs/lora-multilora-serving-routing-playbook.md)：处理 adapter 加载、multi-LoRA batching、LoRA router、adapter cache、多租户和 SLO。

### 积累层

- `notes/`：日常学习笔记，按主题组织。
- `readings/`：论文和源码阅读笔记。
- `experiments/`：动手实验的记录和代码。
- `weekly-reviews/`：每周复盘的实际记录。

### 模板和工具

- [阅读材料清单模板](templates/reading-list.md)：每周学习前先收集材料、设定阅读问题和阅读边界。
- [周复盘模板（精简版）](templates/weekly-review-lite.md)：日常使用，3 个核心问题。
- [周复盘模板（完整版）](templates/weekly-review.md)：需要深入记录时使用。
- [CHANGELOG](CHANGELOG.md)：关键学习节点和里程碑。

## 本周入口

- [本周学习 TODO](weekly-reviews/2026-W27-todo.md)：本周要完成什么、优先级是什么、完成标准是什么。
- [本周进度和过程](weekly-reviews/2026-W27-progress.md)：记录每天推进、判断变化、卡点和下一步。

## 当前定位

当前阶段的主线不是“能跑 vLLM”，而是把推理工作做到行业一流水平：

- 会部署、排查、Benchmark、调参和分析 Profile。
- 真正理解 DP、TP、EP、PP、SP、CP、DCP 的适用场景和代价。
- 能从 Dense、MoE、MLA、Linear Attention 等模型结构推断部署瓶颈。
- 能从 Agent 负载反推推理系统的新需求：长 Session、Memory、Tool Calling、状态恢复、资源隔离、成本和可靠性。
- 把新技术放回系统地图，而不是只记住工具名字。

## 使用方式

每周至少更新一次：

- 先建立本周阅读材料清单。
- 新增学到的技术点。
- 记录一次源码阅读或实验。
- 把论文、框架或问题归类到系统地图。
- 写下一个可以公开输出的主题。

长期目标是形成自己的 `Inference Weekly` 和 `AI Inference Handbook`，让积累从私人笔记变成可以被复用、被引用、被验证的工程知识。
