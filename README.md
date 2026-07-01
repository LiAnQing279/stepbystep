# AI Inference System Engineer Roadmap

这个项目用于沉淀一条长期成长路径：从会部署模型的工程师，成长为具备系统判断、源码能力、实验能力和行业影响力的 AI Inference System Engineer，并进一步走向 AI Infra Architect。

这里不追求收集更多工具名，而是建立一张可以迁移的推理系统地图。以后每看到一个新模型、新框架、新论文、新硬件或新优化方案，都先回答三个问题：

1. 它属于推理系统地图的哪一层？
2. 它解决了这一层的什么瓶颈？
3. 为什么现在这个瓶颈变得重要了？

## 文档结构

### 规划层

- [长期路线图](docs/roadmap.md)：五个阶段的成长目标、能力标准和阶段产出。
- [推理系统知识地图](docs/inference-system-map.md)：把模型、Kernel、Runtime、Scheduler、Cluster、Hardware 放进同一张地图。
- [个人知识路径](docs/learning-path.md)：把每天的工作、论文、源码、实验和输出组织成可持续的学习系统。

### 积累层

- `notes/`：日常学习笔记，按主题组织。
- `readings/`：论文和源码阅读笔记。
- `experiments/`：动手实验的记录和代码。
- `weekly-reviews/`：每周复盘的实际记录。

### 模板和工具

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
- 把新技术放回系统地图，而不是只记住工具名字。

## 使用方式

每周至少更新一次：

- 新增学到的技术点。
- 记录一次源码阅读或实验。
- 把论文、框架或问题归类到系统地图。
- 写下一个可以公开输出的主题。

长期目标是形成自己的 `Inference Weekly` 和 `AI Inference Handbook`，让积累从私人笔记变成可以被复用、被引用、被验证的工程知识。
