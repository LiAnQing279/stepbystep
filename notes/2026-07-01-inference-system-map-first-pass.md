# 推理系统知识地图首轮通读

日期：2026-07-03

## 目标

把 `docs/inference-system-map.md` 从一份长期地图，转换成接下来几周可以执行的学习入口。

这次不追求把每一层都学深，而是先回答三个问题：

- 哪些层级已经有初步判断？
- 哪些层级现在最薄弱？
- 本周和下周应该先补哪条链路？

## 当前理解

推理系统不是单一框架问题，而是一条从模型结构到硬件网络的链路：

```text
模型结构
  -> Kernel
  -> KV Cache
  -> Parallel / Communication
  -> Runtime Engine
  -> Scheduler / Deployment
  -> Hardware / Network
```

每次遇到新论文、新框架或新优化方案时，不先记名字，而是先放回这张地图：

1. 它属于哪一层？
2. 它解决了什么瓶颈？
3. 它把瓶颈转移到了哪里？
4. 它依赖什么硬件、负载或业务条件？

## 当前相对清楚的部分

### 模型结构

已经知道 Dense、MoE、MLA、Linear Attention 会影响不同瓶颈：

- Dense 模型常见 Decode memory-bound。
- MoE 会引入 Expert 路由和 All-to-All 通信。
- 长上下文会放大 KV Cache 压力。
- 新 Attention 结构可能改变 Runtime 的调度假设。

还需要补的是：看到具体模型配置时，能不能快速估算显存、计算和通信压力。

### Parallel

已经有第一版并行策略对照表：DP / TP / EP / PP / SP / CP / DCP。

当前基本判断：

- 单机大模型优先考虑 TP。
- 跨节点后要警惕 TP 通信成本，可能转向 PP 或 TP+PP。
- MoE 模型需要单独考虑 EP 和 All-to-All。
- 超长上下文才会明显牵引 CP / DCP。

还需要补的是：把这些判断和真实 Benchmark / Profile 结果连起来。

### Runtime Engine

已经明确本月不同时展开所有框架，先聚焦 vLLM 请求生命周期和 Scheduler。

当前理解：

- Runtime 不只是启动模型。
- 它负责请求接入、batch 构造、调度、KV Cache 管理、Worker 执行和 token 返回。
- 性能差异往往不是单点，而是 Scheduler、KV Cache、Kernel、并行策略共同作用。

## 当前最薄弱的 3 个层级

### 1. Scheduler / Deployment

薄弱原因：

- 现在只知道 Continuous Batching、Chunked Prefill、PD Disaggregation 等概念，还没有把它们放进一次真实请求链路里。
- 对 TTFT、TPOT、吞吐、排队延迟之间的关系还不够熟。
- 还没有形成“线上容量规划”视角。

下一步：

- 先画出 vLLM 高层请求生命周期。
- 单独拆 Scheduler：请求状态、batch 构造、prefill/decode 混排、资源回收。
- Benchmark 时必须同时记录吞吐、TTFT、TPOT、显存、GPU 利用率。

### 2. KV Cache

薄弱原因：

- 知道 PagedAttention / block manager 的方向，但还没有把 KV Cache 分配、复用、释放和调度关系讲清楚。
- 长上下文、并发、batch size 如何共同放大 KV Cache 压力，目前还缺少量化感。

下一步：

- 在 vLLM 请求生命周期里标出 KV Cache 介入的位置。
- 做一个最小估算：不同 batch、输入长度、输出长度下 KV Cache 大小如何变化。
- 后续单独开一周学习 KV Cache。

### 3. Communication

薄弱原因：

- 对 AllReduce、AllGather、ReduceScatter、All-to-All 的直觉还不够强。
- 知道 TP/EP 会通信，但还不能快速判断通信是否已经成为瓶颈。
- 对 NCCL、RDMA、NVLink、PCIe、IB/RoCE 的边界理解还需要实验支撑。

下一步：

- 先把并行策略对照表和通信模式绑定。
- Benchmark 中记录单机/多卡场景时，要把通信时间作为单独观察对象。
- 后续读 DeepEP 或 MoE EP 时再深入 All-to-All。

## 本周优先级

本周主线应该收敛成一条：

```text
vLLM 请求生命周期
  -> Scheduler 如何构造 batch
  -> KV Cache 在哪里分配和释放
  -> Benchmark/Profile 应该观察哪些指标
```

这条线能把本周 TODO 的几个产物串起来：

- 系统地图首轮笔记。
- 并行策略对照表。
- vLLM 请求生命周期链路。
- Benchmark / Profile 检查清单。
- 周复盘和下周主线。

## 下周候选主线

建议下周选 `Scheduler`，理由：

- 它直接承接 vLLM 请求生命周期。
- 它能解释 TTFT、TPOT、吞吐和显存之间的权衡。
- 它会自然引出 KV Cache、Chunked Prefill、Continuous Batching 和 PD 分离。

如果本周请求生命周期还没完全讲清楚，下周就继续 `vLLM Scheduler`；如果已经讲清楚，再进入 `KV Cache 管理和长上下文`。
