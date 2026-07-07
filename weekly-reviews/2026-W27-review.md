# 2026-W27 周复盘

周期：2026-06-29 到 2026-07-05

## 本周最重要的一件事

把项目从“长期规划”推进到了“可执行学习闭环”。

本周已经不只是有路线图，而是形成了几类可持续写入的产物：

- 系统地图首轮通读笔记。
- 分布式推理并行策略对照表。
- vLLM 请求生命周期高层链路。
- Benchmark / Profile 检查清单。
- 学习问题沉淀。
- vLLM vs SGLang 实验目录。

## 新理解

### 系统地图不是目录，而是归因工具

所属层级：全局方法论。

新的理解：

- 新技术不应该只按名字记忆。
- 每个技术点都要放回模型结构、Kernel、KV Cache、Parallel、Communication、Runtime、Scheduler、Hardware 这些层级里。
- 真正有价值的问题是：它解决哪个瓶颈，又把瓶颈转移到哪里？

### vLLM 主线应该从 Scheduler 进入

所属层级：Runtime Engine / Scheduler。

新的理解：

- API Server 只是入口。
- 一次请求的核心链路是 `Scheduler + KV Cache + Worker Execute`。
- TTFT、TPOT、吞吐和并发，本质上都绕不开 Scheduler 的资源决策。

### Benchmark 不能只看总吞吐

所属层级：Scheduler / KV Cache / Kernel / Communication。

新的理解：

- total tokens/s 不能单独作为结论。
- prompt tokens/s、output tokens/s、TTFT、TPOT、显存、GPU 利用率必须一起看。
- 没有 Profile 证据，就很难判断差异来自调度、KV Cache、Kernel 还是通信。

## 本周输出

已完成：

- `notes/2026-07-01-inference-system-map-first-pass.md`
- `notes/2026-07-02-parallel-strategies.md`
- `notes/2026-07-03-vllm-request-lifecycle.md`
- `notes/2026-07-03-benchmark-profile-checklist.md`
- `notes/2026-07-03-benchmark-attribution-question.md`
- `experiments/2026-07-vllm-vs-sglang-qwen7b/README.md`

可继续打磨：

- vLLM 请求生命周期可以进一步变成源码阅读笔记。
- Benchmark 检查清单可以补具体命令和结果表。
- 并行策略对照表后续可以补通信量估算。

## 下周主线

建议下周主线：vLLM Scheduler。

学习前置步骤：

- 先阅读 `readings/2026-W28-vllm-scheduler-reading-list.md`。
- 读完后整理阅读摘记和问题清单。
- 再进入 Scheduler 源码阅读和学习整理。

必做：

- 完成 vLLM Scheduler 阅读材料的阅读和摘记。
- 阅读 Scheduler 的核心入口。
- 画出 waiting / running / finished 请求状态流转。
- 解释 prefill 和 decode 如何进入同一轮调度。
- 标出 KV Cache block 分配和释放的位置。
- 把 Scheduler 逻辑和 TTFT / TPOT / throughput 指标关联起来。

可选：

- 跑一组最小 Benchmark。
- 抓一次短 prompt 和长 prompt 的 Profile。
- 写一篇短文草稿：《为什么推理 Benchmark 不能只看 tokens/s》。

## 风险

- 过早展开 vLLM、SGLang、TensorRT-LLM 三条线会分散注意力。
- 只读源码但不做实验，容易缺少性能直觉。
- 只做 Benchmark 但不做 Profile，容易无法归因。

## 下周第一步

从阅读材料开始，先回答一个问题：

```text
读 Scheduler 源码前，我需要先理解哪些概念和背景？
```
