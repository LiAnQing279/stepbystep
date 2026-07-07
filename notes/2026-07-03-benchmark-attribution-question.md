# 学习问题沉淀：Benchmark 数字差异如何归因

日期：2026-07-03

## 现象

准备做 vLLM vs SGLang Benchmark 时，容易只比较 `tokens/s`，但单个吞吐数字不能解释差异来源。

同样是吞吐下降，原因可能完全不同：

- 请求排队变长。
- Prefill 太重导致 TTFT 上升。
- Decode 阶段 KV Cache 读取成为瓶颈。
- Scheduler 没有把 batch 喂满。
- GPU kernel 效率低。
- 多卡通信拖慢。
- 请求生成器或 tokenizer 成为瓶颈。

## 环境

当前是学习和实验设计阶段，目标实验为：

- 框架：vLLM vs SGLang。
- 模型：Qwen2-7B / Qwen2.5-7B。
- 目标环境：单卡 A100/A800 80G。
- 实验目录：`experiments/2026-07-vllm-vs-sglang-qwen7b/`。

## 复现路径

在实验设计中，如果只记录：

```text
framework A: 多少 tokens/s
framework B: 多少 tokens/s
```

就无法判断性能差异来自：

- Scheduler。
- KV Cache。
- Kernel。
- Communication。
- API / tokenizer / client overhead。

## 初步假设

推理 Benchmark 不能只看总吞吐，至少要拆成：

- prompt tokens/s。
- output tokens/s。
- TTFT。
- TPOT。
- peak memory。
- GPU utilization。

如果要做归因，还需要 Profile 证据。

## 验证方法

### 第一层：指标拆分

固定输入长度、输出长度和并发，分别记录：

- prompt tokens/s。
- output tokens/s。
- TTFT P50/P90/P99。
- TPOT P50/P90/P99。
- end-to-end latency。
- 峰值显存。

### 第二层：负载变化

用三组负载观察瓶颈迁移：

1. 固定输入长度，变并发：看调度和并发上限。
2. 固定并发，变输入长度：看 Prefill 和 KV Cache 压力。
3. 混合负载：看长尾和线上相似度。

### 第三层：Profile 证据

至少抓一次：

- 单请求短 prompt。
- 单请求长 prompt。
- 高并发混合负载。

观察 CPU/GPU 时间线、kernel gap、KV Cache 行为和显存变化。

## 当前根因判断

当前最大问题不是还没有跑实验，而是实验设计还没有强制要求“数字必须能解释”。

如果缺少指标拆分和 Profile，Benchmark 很容易退化成排行榜，无法沉淀为推理系统判断。

## 解决方案

已补充 `notes/2026-07-03-benchmark-profile-checklist.md`，把 Benchmark 固定成以下结构：

- 先固定变量。
- 再设计负载。
- 同时记录吞吐、延迟和资源。
- 用 Profile 支撑归因。
- 最后把差异归到系统地图的一层。

## 归属层级

- Scheduler / Deployment：TTFT、排队、batch 构造。
- KV Cache：显存占用、长上下文、并发上限。
- Kernel：Attention / MLP / Sampling 热点。
- Communication：多卡 TP/EP/PP 场景。
- Runtime Engine：框架整体执行路径差异。

## 可复用经验

以后所有推理 Benchmark 都遵守一个规则：

```text
没有归因的数字，不作为结论。
```

每个性能结论至少要回答：

1. 差异表现在哪个指标？
2. 差异属于系统地图哪一层？
3. 哪个实验或 Profile 证据支持这个判断？
4. 是否还有其他可能解释？
