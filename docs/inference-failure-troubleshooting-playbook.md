# 推理失败故障排查 Playbook

日期：2026-07-03

## 定位

推理系统最难的工作之一不是正常跑通，而是异常失败时快速定位：

```text
是请求问题、模型问题、参数问题、KV Cache 问题、调度问题、
GPU / CUDA / NCCL 问题、框架 bug，还是基础设施问题？
```

这份文档用于建立一套排障流程。目标是：

- 先止血。
- 再定位。
- 再复现。
- 最后沉淀成可复用经验。

## 总原则

### 1. 先分层，不要直接猜

每次故障先归到一层：

```text
Client / Request
  -> API / Tokenizer
  -> Scheduler
  -> KV Cache
  -> Model Forward
  -> Kernel / CUDA
  -> Distributed / NCCL / Network
  -> Storage / Image / Runtime Env
  -> Deployment / Autoscaling
```

### 2. 先判断影响面

先问：

- 是单个请求失败，还是所有请求失败？
- 是某个模型失败，还是所有模型失败？
- 是单卡失败，还是多卡/多机才失败？
- 是启动失败，还是运行一段时间后失败？
- 是必现，还是偶发？
- 是失败，还是结果错误/性能退化？

影响面决定排查方向。

### 3. 先止血，再深挖

线上故障时优先：

- 降级到已知稳定版本。
- 关闭新特性 flag。
- 降低并发。
- 降低 max model len。
- 移除长请求。
- 切换到备用 deployment。
- 回滚镜像。

止血后再做根因分析。

## 故障分类

| 故障类型 | 常见表现 | 优先怀疑 |
|----------|----------|----------|
| 启动失败 | server 起不来、worker crash、load model 失败 | 镜像、依赖、权重、config、CUDA |
| 请求失败 | 4xx/5xx、timeout、stream 中断 | 请求参数、API、scheduler、worker |
| OOM | CUDA OOM、进程被 kill、显存爆 | 权重、KV Cache、batch、max len |
| Hang | 请求卡住、GPU 空等、NCCL 卡住 | 通信、死锁、worker 异常、队列阻塞 |
| 输出错误 | 乱码、重复、提前 EOS、质量异常 | tokenizer、sampling、模型适配、KV 污染 |
| 性能退化 | TTFT/TPOT/P99 变差、GPU 利用率异常 | 调度、KV、通信、CPU、workload 变化 |
| 多卡失败 | 单卡正常，多卡失败 | TP/PP/EP/CP、NCCL、拓扑、rank 配置 |
| 偶发失败 | 少量请求失败或长尾 | 特定输入、长上下文、资源竞争、网络抖动 |

## 第一阶段：止血

### 线上立即动作

```text
1. 确认影响范围
2. 保留日志和 metrics
3. 降低流量或切走流量
4. 关闭最近新增 feature flag
5. 回滚镜像或配置
6. 降低 max concurrency / max model len
7. 禁止超长请求进入主池
8. 确认服务恢复
```

不要在没有保留证据前重启所有实例，否则会丢失根因线索。

### 必须保留的证据

- 请求样例和 request id。
- 模型名和版本。
- 镜像 tag。
- vLLM / SGLang commit。
- CUDA / driver / PyTorch / NCCL 版本。
- 完整启动参数。
- 最近配置变更。
- 关键日志。
- GPU memory / utilization。
- TTFT / TPOT / P99。
- OOM / traceback / NCCL error。

## 第二阶段：快速定位路径

### A. 启动失败

排查顺序：

1. 镜像是否正确。
2. CUDA / driver / PyTorch 是否匹配。
3. 模型权重是否完整。
4. config / tokenizer 是否匹配。
5. trust remote code / custom model 是否需要。
6. 单卡能否启动。
7. 多卡启动是否卡在 distributed init。

常见根因：

- 权重 shard 缺失。
- tokenizer 文件不完整。
- 模型 architecture 未被 serving 框架支持。
- CUDA 版本和 wheel 不匹配。
- NCCL 初始化失败。
- 显存不足以加载权重。

### B. 请求立即失败

排查顺序：

1. 请求参数是否非法。
2. input length 是否超过 max model len。
3. sampling 参数是否异常。
4. API server 日志是否报错。
5. worker 是否收到请求。
6. scheduler 是否拒绝请求。

常见根因：

- prompt 太长。
- max tokens 过大。
- 请求 payload 格式不兼容。
- tokenizer 报错。
- 多模态输入缺文件或格式错误。

### C. 请求 timeout 或 hang

排查顺序：

1. API server 是否仍在收请求。
2. Scheduler loop 是否仍在推进。
3. waiting queue 是否增长。
4. GPU worker 是否持续执行 kernel。
5. GPU 是否空闲。
6. NCCL / P2P 是否卡住。
7. 是否只有多卡场景发生。

判断：

- GPU 空闲 + queue 增长：可能是 scheduler / worker 通信 / CPU 阻塞。
- GPU 忙 + timeout：可能是超长 prefill / decode 太慢 / kernel 卡住。
- 多卡才 hang：优先怀疑 NCCL、rank、拓扑、通信组。

### D. OOM

排查顺序：

1. 是启动 OOM，还是运行时 OOM？
2. 单卡权重是否放得下？
3. max model len 是否过大？
4. 并发和 batch token 是否过大？
5. KV Cache 是否接近上限？
6. 是否有长上下文突刺？
7. CUDA graph / workspace 是否额外占显存？

止血：

- 降 max model len。
- 降 max num seqs。
- 降 max batched tokens。
- 降 GPU memory utilization。
- 加 TP / PP。
- 开量化。
- 分离长请求。

注意：

- 运行时 OOM 常常不是权重问题，而是 KV Cache / activation / temporary buffer 问题。

### E. 输出异常

表现：

- 乱码。
- 重复 token。
- 过早 EOS。
- stop 条件异常。
- 同样输入输出与 baseline 差异大。

排查顺序：

1. tokenizer / chat template 是否一致。
2. sampling 参数是否一致。
3. greedy decode 是否正常。
4. 单 batch 是否正常，多 batch 是否异常。
5. TP=1 是否正常，TP>1 是否异常。
6. KV Cache 是否被错误复用。
7. custom model / custom op 是否改过。

常见根因：

- tokenizer 不匹配。
- position id / attention mask 错误。
- KV Cache layout 错误。
- rejected speculative token 污染 KV。
- weight loader 映射错误。
- 量化精度问题。

### F. 性能突然退化

排查顺序：

1. workload 是否变了。
2. input/output length 分布是否变了。
3. 长请求比例是否上升。
4. cache hit rate 是否下降。
5. GPU utilization 是否变化。
6. HBM bandwidth / NCCL time 是否变化。
7. scheduler step latency 是否变化。
8. 最近是否改镜像、参数、模型版本。

典型根因：

- 长短请求混跑污染 P99。
- prefix cache 命中下降。
- DP 路由打散 KV locality。
- batch token 设置不合理。
- NCCL / 网络抖动。
- CPU/tokenizer 成为瓶颈。

## 第三阶段：按层排查

### Client / Request

看：

- 请求体大小。
- prompt 长度。
- output length。
- 超时配置。
- client 是否断连。
- 重试是否放大流量。

典型问题：

- 客户端 timeout 太短。
- 大请求上传慢。
- 重试风暴导致服务端被打爆。

### API / Tokenizer

看：

- CPU utilization。
- tokenizer latency。
- event loop lag。
- request validation 时间。

典型问题：

- 长 prompt tokenization 很慢。
- chat template 不兼容。
- API server CPU 打满。

### Scheduler

看：

- waiting queue。
- running seqs。
- scheduled tokens per step。
- prefill/decode mix。
- scheduler step latency。
- preemption count。

典型问题：

- 长 prefill 抢占 decode。
- KV block 不足。
- max batched tokens 不合理。
- priority / routing 策略不合理。

### KV Cache

看：

- KV block usage。
- prefix cache hit。
- block allocation failure。
- request finish 后 block 是否释放。
- preemption / recompute。

典型问题：

- 长上下文占满 KV。
- cache 命中策略导致排队。
- rejected token 污染 KV。
- PD KV transfer 失败。

### Model / Kernel

看：

- Attention / MLP / MoE kernel。
- custom op。
- dtype。
- CUDA graph。
- kernel gap。

典型问题：

- 新模型 forward 不适配。
- custom op 数值错。
- kernel 对某些 shape 性能差。
- MoE expert imbalance。

### Distributed / Communication

看：

- rank 是否全部 alive。
- NCCL error。
- collective 是否卡住。
- NVLink / PCIe / IB 带宽。
- TP/PP/EP/CP 组合。

典型问题：

- rank 配置不一致。
- 跨节点 TP 通信太重。
- expert All-to-All 负载不均。
- topology 不亲和。

### Infrastructure

看：

- node health。
- GPU Xid。
- driver reset。
- container restart。
- disk / network storage。
- image pull。

典型问题：

- GPU 硬件错误。
- 节点网络异常。
- 共享存储抖动。
- 镜像或依赖漂移。

## 第四阶段：复现

复现要从最小 case 开始：

```text
1. 单请求
2. 单 batch
3. 固定 seed
4. 固定输入长度
5. 单卡
6. 再加 TP/PP/EP/CP
7. 再加混合 workload
8. 再加线上流量分布
```

复现时每次只改一个变量：

- 模型版本。
- 镜像版本。
- 并发。
- input length。
- output length。
- max model len。
- max batched tokens。
- TP / PP / EP / CP。
- feature flag。

## 第五阶段：根因复盘模板

每次事故后写：

```text
故障标题：
时间：
影响范围：
用户影响：
触发条件：
直接原因：
根本原因：
为什么监控没有提前发现：
为什么测试没有覆盖：
止血动作：
长期修复：
需要新增的指标：
需要新增的测试：
是否需要更新 runbook：
```

## 必备监控

### 服务指标

- QPS。
- error rate。
- timeout rate。
- TTFT / TPOT P50/P90/P99。
- goodput。
- request queue length。

### Scheduler / KV

- waiting requests。
- running requests。
- scheduled tokens per step。
- prefill/decode token ratio。
- KV block usage。
- prefix cache hit rate。
- preemption count。

### GPU / Kernel

- GPU utilization。
- GPU memory used。
- HBM bandwidth。
- kernel time。
- GPU idle gap。

### Distributed

- NCCL time。
- AllReduce / All-to-All / P2P time。
- rank health。
- network bandwidth。

### Infra

- container restart count。
- node health。
- GPU Xid。
- disk / object storage latency。
- image version。

## 常用止血策略

| 问题 | 止血策略 |
|------|----------|
| OOM | 降 max model len、降并发、禁长请求、回滚参数 |
| timeout | 降流量、分离长请求、回滚新特性、扩容 |
| NCCL hang | 切单机/单卡、替换节点、回滚并行配置 |
| 输出异常 | 关闭 custom op / speculative decoding / 新模型适配 |
| P99 暴涨 | 分池、限长请求、开启/调整 chunked prefill |
| GPU 空闲 | 查 API/tokenizer/client/scheduler gap |
| GPU 满但慢 | 查 HBM、kernel、通信、batch shape |

## 和学习路线的关系

故障排查不是独立主题，它要求把所有主线串起来：

```text
Scheduler
  -> KV Cache
  -> Parallel / Communication
  -> Deployment / SLO
  -> Custom Model / Custom Op
  -> SGLang/vLLM Feature Integration
  -> Observability / Incident Response
```

最终目标：

```text
看到异常现象后，
能快速缩小到系统地图的一层，
用证据验证，
用最小动作止血，
再把根因沉淀成测试、监控和 runbook。
```
