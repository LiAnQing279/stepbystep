# Speculative Decoding / MTP 观察线

日期：2026-07-03

## 定位

这份文档用于跟踪 MTP、speculative decoding、DSpark、DeepSpec、EAGLE、Medusa 等 Decode 加速方向。

它暂时不替代当前主线：

```text
vLLM Scheduler
  -> KV Cache 管理
  -> Context Parallel / 长上下文
  -> KV Router
  -> PD 调度
```

但它必须纳入长期路线，因为 Decode 阶段通常是在线推理的核心瓶颈之一，尤其会直接影响：

- TPOT。
- output tokens/s。
- TPM。
- GPU 利用率。
- SLO 下的吞吐上限。

## 核心问题

Speculative decoding 的基本思想：

```text
先用更便宜的 drafter 猜多个 token
再用 target model 并行验证
如果接受率高，就减少 target model 的 decode step 数
```

关注它时，不只看“加速百分比”，要看：

- drafter 成本是多少？
- acceptance rate 有多高？
- 一次能稳定接受几个 token？
- verification 是否引入额外显存、通信或调度开销？
- 对长上下文、MoE、PD 分离、多卡并行是否友好？
- 是否会影响输出分布、质量或确定性？

## 主题归类

| 技术方向 | 系统地图层级 | 解决的问题 | 主要风险 |
|----------|--------------|------------|----------|
| MTP | Model / Runtime / Scheduler | 模型原生预测多个未来 token，减少 decode step | 需要模型原生支持或额外 MTP head |
| Draft-model speculative decoding | Runtime / Scheduler | 小模型草稿，大模型验证 | drafter 成本和接受率不匹配会亏 |
| DSpark / DeepSpec | Runtime / Scheduler / Training | 训练和评估 speculative decoding drafter | 新技术变化快，需要持续跟踪 |
| EAGLE / Medusa 类方法 | Runtime / Model | 通过 head/tree/drafter 提高草稿质量 | 框架集成和验证开销复杂 |
| Tree-based speculative decoding | Scheduler / Kernel / Runtime | 一次验证多分支候选 | attention mask、batch shape、显存开销复杂 |

## 与现有路线的关系

### 与 Scheduler

Scheduler 需要知道：

- drafter 和 verifier 是否在同一 worker。
- speculative step 如何进入 batch。
- verify 阶段是否改变 decode batch shape。
- rejected token 会不会导致额外重算。
- speculative decoding 是否改变 TPOT 和 P99。

### 与 KV Cache

需要关注：

- draft tokens 是否写入 KV Cache？
- rejected tokens 的 KV 如何处理？
- verification 是否复用 KV？
- speculative decoding 会不会增加临时 KV 或 block 压力？

### 与并行组合

需要关注：

- TP / PP / EP 下 drafter 和 verifier 是否使用相同并行组。
- MoE 模型中 drafter 是否也需要 expert routing。
- PP 下 speculative verification 是否放大 bubble。
- CP / DCP 下一次验证多个 token 是否增加上下文通信。

### 与 PD 分离

需要关注：

- speculative decoding 主要优化 Decode worker，还是也影响 Prefill？
- PD 架构下 drafter 放在 Decode worker、Prefill worker，还是独立 worker？
- accepted tokens 增多后，Decode worker 释放时间是否更容易预测？
- KV transfer 和 speculative verification 是否会互相干扰？

## 阅读材料候选

正式学习前，为这个主题单独建立阅读材料清单。候选材料：

- vLLM MTP speculative decoding 文档。
- vLLM speculative decoding 总文档。
- DeepSeek DeepSpec / DSpark paper 和代码库。
- NVIDIA speculative decoding 技术博客。
- TensorRT-LLM DeepSeek R1 MTP 实现与优化文章。
- AMD / SGLang MTP serving 实践。
- Google Gemma MTP 文档。
- EAGLE、Medusa、FastMTP、Entropy-guided MTP 等近期论文。

## 进入顺序

建议放在 `Scheduler + KV Cache` 之后、`PD 调度` 之前：

```text
vLLM Scheduler
  -> KV Cache 管理
  -> Speculative Decoding / MTP
  -> KV Router
  -> PD 调度
```

理由：

- speculative decoding 会改变 decode step 数量和 batch shape。
- 它需要 Scheduler 理解，也会影响 KV Cache 生命周期。
- 在 PD 调度前理解它，可以更好判断 Decode worker 的利用率和释放时间。

## 必须回答的问题

1. MTP 和普通 draft-model speculative decoding 的区别是什么？
2. acceptance rate 如何影响真实加速？
3. drafter 成本什么时候会抵消收益？
4. speculative decoding 对 TTFT、TPOT、TPM 分别有什么影响？
5. rejected token 的 KV Cache 如何处理？
6. 多卡 TP / PP / EP / CP 下 speculative decoding 的通信会变多还是变少？
7. 它和 KV Cache 命中、PD 调度、worker 结束预测之间有什么关系？
8. 对线上 SLO 来说，平均加速和 P99 长尾是否一致？

## 产出要求

后续正式学习时至少产出：

- 一份阅读材料清单。
- 一份 MTP / speculative decoding 概念图。
- 一份 DSpark / DeepSpec 摘记。
- 一份和 vLLM Scheduler / KV Cache 的交互分析。
- 一个 Benchmark 问题：在什么 acceptance rate、output length、batch size 下才真正加速？
- 如果要进入 SGLang 工程集成，使用 `docs/sglang-feature-integration-image-playbook.md`，先做单卡 prototype，再构建自定义镜像和 SLO goodput 验证。
