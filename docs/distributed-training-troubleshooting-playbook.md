# 多机训练问题与故障排查 Playbook

日期：2026-07-03

## 定位

这个项目的主线仍然是推理系统，但实际应用中，训练阶段的问题也必须理解，尤其是跨机或多机训练。

原因：

- 训练产出的模型、checkpoint、tokenizer、config 会直接影响推理部署。
- 训练中的并行方式、通信和数值问题，和推理中的多卡通信、模型切分、显存压力有很多共通点。
- 多机训练排障能力能帮助理解 NCCL、RDMA、拓扑、checkpoint 和故障恢复。

这份文档目标不是成为训练算法路线，而是建立：

```text
推理工程师需要掌握的训练系统问题地图
```

## 训练问题总图

```text
Data
  -> Tokenizer / Dataset / Dataloader
  -> Model / Loss / Optimizer
  -> Parallelism
  -> Communication
  -> Checkpoint
  -> Fault Tolerance
  -> Evaluation
  -> Export / Serving
```

每一层出问题，最后都可能表现为：

- loss 不收敛。
- loss spike。
- NaN / Inf。
- throughput 低。
- GPU 利用率低。
- 多机 hang。
- checkpoint 无法恢复。
- 导出的模型无法推理。
- 推理输出质量异常。

## 1. 数据与 tokenizer 问题

### 常见现象

- loss 不下降。
- loss 周期性 spike。
- 训练速度抖动。
- 某些 step dataloader 卡住。
- 推理时 tokenizer/chat template 不匹配。

### 常见根因

- 数据重复、脏数据、空样本。
- tokenization 规则和推理不一致。
- special token 配置错误。
- sequence packing bug。
- attention mask / loss mask 错误。
- 多机 dataloader shard 不均。
- 数据存储或网络盘抖动。

### 排查证据

- 每 step token 数。
- sample length 分布。
- dataloader time。
- 每 rank 读到的数据量。
- tokenizer config。
- special token id。
- loss mask 可视化。

### 和推理的关系

训练 tokenizer / chat template 如果和 serving 不一致，会导致：

- 推理输出风格异常。
- EOS / stop token 不对。
- 多轮对话模板错位。
- benchmark 结果不可比。

## 2. 数值稳定性问题

### 常见现象

- loss NaN / Inf。
- grad norm 爆炸。
- loss spike 后无法恢复。
- 某些 rank 先 NaN，随后全局扩散。
- 训练继续但模型质量明显下降。

### 常见根因

- learning rate 太大。
- warmup 不够。
- loss scale 问题。
- bf16/fp16 混精配置不当。
- attention mask 错误导致非法值。
- optimizer state 损坏。
- checkpoint 恢复不完整。
- 某些 kernel 数值不稳定。

### 排查证据

- loss / grad norm 曲线。
- overflow 计数。
- activation / weight norm。
- 每 rank 是否同时 NaN。
- 最近是否换 kernel、dtype、optimizer。
- 从上一个 checkpoint 复现。

### 和推理的关系

训练数值问题可能导致：

- 模型能加载但质量差。
- 某些输入输出异常。
- logits 分布异常。
- 推理时重复、乱码、提前 EOS。

## 3. 并行策略问题

### 常见并行

训练里常见：

- DP / DDP。
- FSDP / ZeRO。
- TP。
- PP。
- EP。
- CP / SP。
- sequence packing / context parallel。

### 常见现象

- 单卡正常，多卡不收敛。
- 单机正常，多机 hang。
- 扩卡后吞吐不线性。
- 某些 rank OOM。
- PP bubble 严重。
- expert load imbalance。

### 常见根因

- global batch size 变化但学习率没调。
- gradient accumulation 配置错误。
- TP / PP / DP group 配置错误。
- rank 到 GPU 映射错误。
- pipeline stage 不均。
- MoE expert 路由不均。
- sequence parallel / context parallel mask 错。

### 和推理的关系

训练并行理解有助于推理：

- TP / PP / EP 的通信模式相通。
- MoE expert imbalance 在训练和推理都会出现。
- checkpoint shard 和推理权重加载强相关。
- 训练用的模型切分会影响 serving 转换。

## 4. 通信问题

### 常见现象

- NCCL init hang。
- 训练中途 hang。
- 某个 step 时间突然变长。
- 多机吞吐很差。
- 某些 rank timeout。
- AllReduce / All-to-All 占比异常。

### 常见根因

- rank 配置错误。
- master addr / port 错误。
- 防火墙或安全组阻断。
- IB / RoCE 配置不一致。
- 网卡选择错误。
- NCCL 环境变量不匹配。
- 跨机拓扑不亲和。
- 某个 rank OOM 或 crash，其他 rank 等待。
- All-to-All 负载不均。

### 排查证据

- NCCL log。
- rank 日志。
- 每 rank step time。
- network bandwidth。
- IB/RDMA counters。
- topology。
- 是否所有 rank 都进入同一 collective。

### 关键判断

如果多机 hang，先判断：

```text
是通信本身卡住，
还是某个 rank 在通信前已经 OOM / crash / dataloader 卡住？
```

很多 NCCL hang 的根因不在 NCCL，而是某个 rank 没有走到 collective。

### 和推理的关系

推理中的 TP、EP、PP、PD KV transfer 也会遇到类似问题：

- rank 不一致。
- 拓扑不亲和。
- 跨节点通信慢。
- 某个 worker crash 后其他 worker 等待。

## 5. Checkpoint 问题

### 常见现象

- checkpoint 保存很慢。
- 保存时训练停顿很久。
- 恢复失败。
- 恢复后 loss 跳变。
- 推理加载权重失败。
- 权重 shard 缺失。

### 常见根因

- checkpoint shard 不完整。
- optimizer / scheduler / RNG state 没保存。
- 保存和训练并发争用存储。
- 网络盘吞吐不足。
- 模型结构变了但 checkpoint 未转换。
- TP/PP/EP/FSDP shard 和 serving 格式不兼容。

### 排查证据

- checkpoint manifest。
- 每个 shard 大小。
- 保存耗时。
- restore 日志。
- global step。
- optimizer state 是否存在。
- tokenizer/config 是否一起保存。

### 和推理的关系

推理部署常见问题：

- 模型权重能下载但不能加载。
- config 和权重不匹配。
- safetensors shard 缺失。
- tokenizer 没随模型发布。
- 训练 checkpoint 没转换成 serving 格式。

## 6. 性能问题

### 常见现象

- GPU 利用率低。
- MFU 低。
- step time 抖动。
- dataloader 占比高。
- 通信占比高。
- pipeline bubble 高。

### 常见根因

- batch size 太小。
- sequence length 分布不均。
- dataloader 慢。
- CPU preprocessing 慢。
- storage 慢。
- communication overlap 不好。
- kernel 不匹配 shape。
- PP stage 不均。
- MoE expert imbalance。

### 排查指标

- step time breakdown。
- tokens/s。
- samples/s。
- MFU。
- GPU utilization。
- HBM bandwidth。
- dataloader time。
- communication time。
- bubble time。

### 和推理的关系

训练性能排查和推理性能排查共用很多方法：

- 分解时间。
- 看 GPU idle gap。
- 看通信。
- 看 HBM。
- 看 batch shape。
- 看数据输入。

## 7. 容错和恢复

### 常见问题

- 单个 worker 掉线导致全局失败。
- preemption 后恢复慢。
- elastic training 配置不稳定。
- checkpoint 间隔太长，损失训练进度。
- 恢复后数据位置重复或跳过。

### 关键问题

- checkpoint 是否包含完整训练状态？
- dataloader cursor 是否保存？
- RNG state 是否保存？
- optimizer / lr scheduler 是否保存？
- 多机 rank 数变化后能否恢复？
- 是否支持自动重试？

### 和推理的关系

推理中的 PD 容错、worker failover、KV Cache 恢复，也需要类似思维：

- 状态在哪里？
- 能否重建？
- 重建成本多高？
- 是否需要副本？

## 8. 从训练到推理的交付问题

训练完成不代表能推理。

交付清单：

- model weights。
- config。
- tokenizer。
- chat template。
- generation config。
- quantization config。
- model architecture code。
- license / model card。
- eval 结果。
- checkpoint 转换脚本。
- serving 启动参数。

常见问题：

- 训练用 remote code，推理镜像没有。
- tokenizer 版本漂移。
- chat template 没发布。
- checkpoint 需要 merge / convert。
- MoE expert 权重命名不匹配。
- MLA / MTP 等新结构 serving 不支持。

## 9. 多机训练故障排查流程

```text
1. 判断影响面
  -> 单卡 / 单机 / 多机？

2. 判断类型
  -> NaN / OOM / hang / 慢 / checkpoint / 输出质量？

3. 保留证据
  -> 每 rank log、NCCL log、配置、checkpoint、数据样例

4. 缩小复现
  -> 单卡 -> 单机多卡 -> 两机 -> 全规模

5. 每次只改一个变量
  -> batch、lr、dtype、parallel、network、data、kernel

6. 找到层级
  -> data / numeric / parallel / communication / checkpoint / infra

7. 修复和回归
  -> 加测试、加监控、更新 runbook
```

## 10. 必备监控

训练至少需要：

- loss。
- grad norm。
- learning rate。
- overflow / NaN count。
- tokens/s。
- step time。
- dataloader time。
- communication time。
- checkpoint save time。
- GPU utilization。
- GPU memory。
- network bandwidth。
- per-rank health。
- retry / failure count。

## 11. 和推理路线的关系

训练是旁线，但必须理解：

```text
Training data / tokenizer / checkpoint
  -> Model artifact
  -> Serving runtime
  -> Inference behavior
```

训练问题会在推理里表现为：

- 模型加载失败。
- 输出质量异常。
- tokenizer 不匹配。
- 新结构不被 serving 支持。
- 多卡通信和拓扑问题复现。
- checkpoint 转换错误。

## 12. 后续阅读材料

正式学习前建立：

- `readings/YYYY-WW-distributed-training-troubleshooting-reading-list.md`

候选主题：

- PyTorch Distributed / DDP。
- FSDP / ZeRO。
- Megatron-LM / Megatron-Core。
- DeepSpeed。
- NCCL troubleshooting。
- RDMA / IB / RoCE basics。
- checkpoint / fault tolerance。
- MoE training expert parallel。

## 一句话总结

```text
多机训练排障的核心不是背训练框架参数，
而是能把异常现象定位到 data、numeric、parallel、communication、checkpoint 或 infra 层，
并理解这些问题如何影响最终推理部署。
```
