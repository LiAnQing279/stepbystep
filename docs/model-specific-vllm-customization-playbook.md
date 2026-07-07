# 特定模型 vLLM 自定义实现 Playbook

日期：2026-07-03

## 定位

这份文档回答最有挑战的一类问题：

```text
面对一个特定模型，
除了调参数和换部署拓扑，
我们还能在 vLLM 里自定义哪些代码实现，
来提升模型推理效率？
```

目标不是为了改源码而改源码，而是建立一个分层判断：

```text
先用配置解决
  -> 再用框架已有后端
  -> 再做模型适配
  -> 再做 kernel / op / attention 优化
  -> 最后才改 scheduler / worker / KV 管理
```

## 基本原则

### 1. 先归因，再实现

不要看到模型慢就直接写 kernel。

先确认瓶颈属于哪一层：

- Tokenizer / preprocessing。
- Model forward。
- Attention。
- MLP / MoE。
- Sampling。
- KV Cache。
- Communication。
- Scheduler。
- Python overhead。

不同瓶颈对应完全不同的实现入口。

### 2. 先配置，再代码

优先尝试：

- dtype / quantization。
- tensor parallel / pipeline parallel / expert parallel。
- max model length。
- max batched tokens。
- chunked prefill。
- prefix cache。
- speculative decoding / MTP。
- attention backend。
- CUDA graph。

配置仍无法解决时，再考虑自定义实现。

### 3. 自定义代码必须可回退

每个自定义实现都要能回答：

- 和原始 vLLM 实现相比快在哪里？
- 是否影响正确性？
- 是否影响模型质量？
- 是否影响多卡并行？
- 是否影响 KV Cache？
- 是否影响升级 vLLM？
- 是否有开关可以回退？

## 自定义层级总览

| 层级 | 可以自定义什么 | 解决什么问题 | 风险 |
|------|----------------|--------------|------|
| Model Adapter | 模型结构、config、weight loader | 新模型跑不起来或结构不匹配 | 中 |
| Forward Path | 模型 forward、position embedding、attention mask | 去掉冗余逻辑，适配特殊结构 | 中 |
| Attention Backend | attention 实现、KV layout、backend 选择 | Prefill / Decode attention 慢 | 高 |
| Custom Op | RMSNorm、Rotary、MoE dispatch、sampling 等算子 | Python / torch op overhead 或 kernel 不佳 | 高 |
| Quantization | 权重量化、KV dtype、FP8/INT4 等 | 降显存、提带宽效率 | 中高 |
| KV Cache | block layout、prefix cache、cache reuse | 长上下文、高并发、重复 prefix | 高 |
| MoE Runtime | router、expert placement、dispatch/combine | Expert 不均、All-to-All 慢 | 高 |
| Scheduler | 调度策略、优先级、admission control | SLO、长短混合、低并发高延迟 | 很高 |
| Worker / Distributed | 通信组、pipeline、rank placement | 多卡扩展差、拓扑不亲和 | 很高 |

## 1. Model Adapter：让模型正确接入 vLLM

适用场景：

- 新模型结构 vLLM 还不支持。
- Hugging Face config 和 vLLM 预期不一致。
- 权重名称不匹配。
- 模型有自定义 layer、position embedding、attention mask。

可做实现：

- 新增模型类。
- 实现 `forward`。
- 实现 weight loading。
- 处理 config 映射。
- 注册模型 architecture。
- 适配特殊 tokenizer 或 chat template。

优化机会：

- 去掉训练态才需要的分支。
- 合并不必要的 tensor reshape。
- 避免每 step 重复构造 mask。
- 避免 CPU/GPU 来回拷贝。

风险：

- 权重加载错误但不一定立刻报错。
- logits 对齐错误导致质量下降。
- 多卡 TP/PP 切分不正确。

验证：

- 单条 prompt logits 对齐。
- greedy decode 输出对齐。
- 多 batch 输出对齐。
- TP=1 和 TP>1 输出一致。

## 2. Forward Path：针对模型结构剪掉浪费

适用场景：

- 模型有特殊 attention、MLA、GQA/MQA、MoE、linear attention。
- 通用 Transformers forward 太慢。
- 有训练态逻辑、debug 逻辑、动态 shape 逻辑影响推理。

可做实现：

- 写 vLLM native model forward。
- 固化推理路径。
- 优化 position id / rotary embedding。
- 优化 attention mask 构造。
- 合并小 op。
- 避免 Python 层循环。

优化目标：

- 降低 CPU overhead。
- 降低 kernel launch 数量。
- 提高 CUDA graph 复用机会。
- 稳定 batch shape。

## 3. Attention Backend：优化 Prefill / Decode 核心路径

适用场景：

- Attention 是 Profile 热点。
- 长上下文 prefill 慢。
- Decode TPOT 高。
- KV Cache layout 与模型结构不匹配。
- 默认 backend 对某模型结构不友好。

可做实现：

- 选择或接入新的 attention backend。
- 适配 FlashAttention / FlashInfer / Triton kernel。
- 优化 MLA / GQA / MQA 的 KV layout。
- 对特定 head_dim / block_size 做专门 kernel。
- 优化 causal mask / sliding window / sparse attention。

关键判断：

- Prefill 是 compute-bound 还是 memory-bound？
- Decode 是 HBM bandwidth 还是 kernel launch 问题？
- head_dim、num_heads、num_kv_heads 是否适合现有 kernel？
- 长上下文是否需要 CP / DCP 配合？

风险：

- attention 数值不一致。
- KV layout 和 block table 不匹配。
- 多卡 TP/CP 下通信和 layout 冲突。
- 升级 vLLM 后 backend API 变化。

## 4. Custom Op：替换低效算子

适用场景：

- Profile 发现大量小 op。
- RMSNorm / LayerNorm / Rotary / activation / sampling 占比异常。
- MoE dispatch / combine 慢。
- Torch eager op 太碎，kernel launch overhead 高。

可做实现：

- 自定义 RMSNorm。
- 自定义 Rotary Embedding。
- 自定义 fused activation。
- 自定义 MoE dispatch/combine。
- 自定义 sampling / logits processor。
- 用 vLLM CustomOp 接口接入特定 op。

优化目标：

- 减少 kernel launch。
- 减少 HBM 读写。
- 融合相邻 op。
- 避免不必要的 dtype cast。

验证：

- 数值误差。
- 不同 dtype。
- 不同 batch shape。
- CUDA graph 兼容。
- 多 GPU rank 一致。

## 5. Quantization / dtype：降低显存和带宽压力

适用场景：

- 权重显存限制部署。
- Decode HBM bandwidth 限制 TPOT。
- 需要提高 batch size 或 max model len。

可做实现：

- 接入或适配模型特定 quantization。
- 优化 FP8 / INT8 / INT4 权重加载。
- KV Cache dtype 调整。
- 特定 layer 保持高精度，其他 layer 量化。

关键判断：

- 量化是降低权重带宽，还是降低 KV Cache 带宽？
- 对 Prefill 和 Decode 的收益是否不同？
- 质量损失是否可接受？
- 是否影响 MoE router 或 logits 稳定性？

## 6. KV Cache：为模型和业务优化缓存

适用场景：

- 长上下文 OOM。
- KV block 使用率高。
- prefix 重复多。
- cache hit 高但 SLO 变差。
- PD 分离后 KV transfer 成本高。

可做实现：

- 自定义 prefix cache 策略。
- 自定义 KV Cache eviction。
- 调整 block size / layout。
- 接入 LMCache / HiCache / Mooncake 类缓存系统。
- KV-aware routing。
- 长短请求分池和 KV 预算隔离。

优化目标：

- 提高有效 cache hit。
- 降低 prefill 重算。
- 降低 KV transfer。
- 在 SLO 内提高 goodput。

风险：

- 只追 cache hit，导致排队变长。
- KV 生命周期错误导致输出错误。
- cache 元数据同步成本过高。

## 7. MoE Runtime：针对 Expert 做优化

适用场景：

- MoE 模型 expert 层慢。
- Expert load imbalance。
- EP All-to-All 成为瓶颈。
- 某些 expert 热点明显。

可做实现：

- expert placement。
- router 输出后 token regroup。
- dispatch / combine kernel 优化。
- expert load balance 策略。
- TP/EP/DP 组合调整。
- DeepEP / 通信库适配。

关键判断：

- 瓶颈在 router、dispatch、expert MLP 还是 combine？
- All-to-All 是否超过计算收益？
- expert 热点是否和业务输入分布相关？

## 8. Scheduler：针对业务 SLO 改调度

适用场景：

- 长短请求混合。
- 不规则超长上下文。
- 低并发但延迟大。
- 短请求 P99 被长请求污染。
- cache hit 和 SLO 冲突。
- PD 分离下 worker 释放时间预测重要。

可做实现：

- admission control。
- length-aware scheduling。
- priority scheduling。
- short / medium / long pool。
- SLO-aware routing。
- cache-aware routing。
- prefill/decode token budget 策略。
- worker finish time prediction。

风险：

- 改 Scheduler 很容易牺牲整体吞吐。
- 局部优化短请求可能饿死长请求。
- 路由依赖指标不准会导致震荡。

验证：

- short goodput。
- long goodput。
- SLO violation rate。
- GPU utilization。
- starvation rate。
- cache hit 与 queue delay 的关系。

## 9. Worker / Distributed：针对拓扑和通信优化

适用场景：

- 多卡扩展差。
- TP / PP / EP / CP 通信占比高。
- 跨节点后 TPOT 变差。
- PD KV transfer 延迟高。

可做实现：

- rank placement。
- topology-aware scheduling。
- 通信组调整。
- TP within node + PP across nodes。
- expert placement 贴近拓扑。
- PD prefill/decode worker 拓扑亲和。

关键判断：

- 通信发生在每层、每 MoE 层、每 decode step，还是 KV transfer？
- 当前拓扑是 NVLink、PCIe、IB/RDMA 哪个瓶颈？
- Scheduler 是否知道拓扑成本？

## 针对特定模型的实现流程

```text
1. 模型画像
  -> Dense / MoE / MLA / GQA / long context / MTP

2. workload 画像
  -> input/output 分布、prefix 重复、SLO

3. baseline
  -> 官方 vLLM 支持 + 默认参数

4. Profile
  -> 找 CPU / GPU compute / HBM / KV / comm / scheduler 瓶颈

5. 选择最小自定义点
  -> model adapter / op / attention / KV / scheduler

6. 正确性验证
  -> logits / decode / 多卡 / dtype

7. 性能验证
  -> SLO goodput / TTFT / TPOT / GPU / memory

8. 回退和维护
  -> feature flag / upstream diff / 文档
```

## 常见模型类型与优化入口

### Dense + 标准 Attention

优先：

- TP / DP / PP。
- CUDA graph。
- attention backend。
- quantization。
- batch / scheduler 参数。

少改：

- Scheduler。
- KV Cache 内部结构。

### GQA / MQA 模型

优先：

- KV head 数和 KV layout。
- attention backend 是否充分利用少量 KV heads。
- KV Cache 显存估算。

### MLA 模型

优先：

- MLA attention 实现。
- KV representation。
- decode 读取路径。
- 模型特定 attention backend。

### MoE 模型

优先：

- EP。
- expert placement。
- dispatch/combine。
- DeepEP / All-to-All。
- router 负载均衡。

### 超长上下文模型

优先：

- KV Cache。
- Chunked Prefill。
- CP / DCP。
- prefix cache。
- length-aware scheduling。
- long-context pool。

### 支持 MTP 的模型

优先：

- speculative decoding / MTP 接入。
- draft / verify 路径。
- accepted tokens 对 KV Cache 的影响。
- TPOT 和 P99 是否真正改善。

## 什么时候不该自定义代码

不要自定义代码的情况：

- 没有 profile 证据。
- 只是 QPS 不够，但 DP 扩副本即可。
- 只是显存不够，但 TP/quantization 可以解决。
- 只是长短请求混跑导致 P99 差，分池即可解决。
- 自定义实现无法维护或无法回退。

## 后续阅读材料

正式展开这个主题前，建立：

- `readings/YYYY-WW-vllm-custom-model-implementation-reading-list.md`

候选材料：

- vLLM Adding a New Model 文档。
- vLLM Models 文档。
- vLLM CustomOp 文档。
- vLLM Attention Backend 文档。
- vLLM Quantization 文档。
- vLLM Developer Guide。
- 目标模型的 model card / config / modeling 源码。
- 目标模型相关 inference optimization paper / issue / PR。

## 一句话总结

```text
针对特定模型优化 vLLM，不是“哪里慢改哪里”，
而是先把模型结构、workload、SLO、Profile 瓶颈放到同一张图里，
再选择最小但有效的自定义实现点。
```
