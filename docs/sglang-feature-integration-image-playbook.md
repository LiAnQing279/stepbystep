# 从论文新特性到 SGLang 镜像支持 Playbook

日期：2026-07-03

## 定位

这份文档回答一个工程落地问题：

```text
论文或新项目出现一个新特性，比如 DSpark / DeepSpec / MTP，
我们如何把它加入到 SGLang 镜像中，实现自定义支持？
```

目标不是简单把代码 copy 进镜像，而是形成一条可复用路径：

```text
读论文 / 代码
  -> 判断特性属于哪一层
  -> 找 SGLang 扩展点
  -> 最小实现
  -> 构建镜像
  -> 正确性验证
  -> 性能验证
  -> SLO goodput 验证
  -> 回退和维护
```

## 先判断特性属于哪一层

每个新特性先归类，决定要改哪里。

| 特性类型 | 例子 | 可能改动位置 |
|----------|------|--------------|
| Model feature | MTP head、MLA、特殊 RoPE、GQA/MQA | model loader、forward、attention |
| Runtime feature | speculative decoding、draft/verify、tree verification | scheduler、engine、decode loop |
| KV feature | prefix reuse、KV offload、KV router | KV cache、router、scheduler |
| Kernel feature | custom attention、MoE dispatch、fused op | CUDA/Triton/custom op |
| Communication feature | EP、PD KV transfer、topology aware | distributed、worker、scheduler |
| Deployment feature | 新依赖、新服务进程、新配置 | Dockerfile、entrypoint、config |

DSpark / DeepSpec 这类 speculative decoding 特性通常横跨：

- Model：drafter / MTP。
- Runtime：draft + verify 流程。
- Scheduler：一次 decode step 可能不再只生成 1 个 token。
- KV Cache：accepted / rejected token 如何处理。
- Benchmark：acceptance rate 和真实加速。

## 总流程

```text
1. 读论文和代码
2. 定义集成边界
3. 选择 SGLang 扩展点
4. 做最小 prototype
5. 加配置开关
6. 构建自定义 Docker 镜像
7. 正确性验证
8. 性能和 SLO 验证
9. 文档化和回退
```

## 1. 读论文和代码

先回答：

- 这个特性解决哪个瓶颈？
- 它依赖模型结构，还是只是 serving 策略？
- 它是否需要训练额外 head / drafter？
- 它是否需要改 attention mask / KV Cache？
- 它是否要求特定 CUDA / Triton / PyTorch 版本？
- 它和现有 SGLang speculative decoding 是否重复？

以 DSpark / DeepSpec 为例，重点看：

- drafter 是如何训练或选择的。
- draft tokens 如何生成。
- target model 如何 verify。
- acceptance rate 如何计算。
- benchmark workload 是什么。
- 是否有现成 SGLang/vLLM/TensorRT-LLM 集成路径。

## 2. 定义集成边界

不要一开始就做完整集成。先选最小边界：

### Level 0：镜像中带上代码和依赖

目标：

- 能在 SGLang 镜像里 import 新特性代码。
- 能跑论文仓库自带 demo / unit test。

不改 SGLang serving 主链路。

### Level 1：离线验证

目标：

- 用同一模型和输入，对比 baseline decode 和新特性输出。
- 验证 logits / token / acceptance rate。

仍不接入线上 API。

### Level 2：接入 SGLang runtime 实验开关

目标：

- 加一个 feature flag。
- 请求级或服务级开启新特性。
- 支持回退到 baseline。

### Level 3：生产化支持

目标：

- 完整 metrics。
- 错误处理。
- 多卡支持。
- SLO goodput 验证。
- 镜像版本和部署文档。

## 3. 找 SGLang 扩展点

读 SGLang 时重点找这些位置：

```text
请求入口：
  server / OpenAI API / request parser

调度：
  scheduler / batch construction / decode loop

模型执行：
  model runner / forward batch / sampling

Speculative decoding：
  draft model / verify / acceptance / tree decode

KV Cache：
  radix cache / KV pool / prefix cache / accepted token update

配置：
  launch args / server args / model config

镜像：
  Dockerfile / requirements / entrypoint / build scripts
```

判断：

- 如果新特性只影响 draft/verify，优先接 speculative decoding 模块。
- 如果新特性需要模型结构，先接 model loader / forward。
- 如果新特性需要特殊 kernel，先离线验证 custom op。
- 如果新特性改变 decode step，需要同步 scheduler 和 KV Cache。

## 4. Prototype 实现策略

### 最小实现原则

先实现：

- 单机。
- 单卡。
- 单模型。
- 固定 batch。
- 不开 PD。
- 不开 CP/DCP。
- 不考虑复杂路由。

目标是验证特性本身是否有效。

### Feature flag

所有新特性必须带开关：

```text
--enable-dspark
--dspark-drafter-path
--dspark-verify-mode
--dspark-max-draft-tokens
```

命名只是示例，实际以 SGLang 参数风格为准。

### 回退路径

必须能做到：

```text
flag off -> 原始 SGLang 行为
flag on but error -> 自动 fallback 或 fail fast
```

不要让新特性污染默认路径。

## 5. 构建自定义 SGLang 镜像

### 镜像策略

推荐三层：

```text
base image:
  CUDA / PyTorch / SGLang 官方环境

feature layer:
  DSpark / DeepSpec 代码和依赖

patch layer:
  SGLang 自定义改动
```

这样便于升级 SGLang 或替换特性代码。

### 构建内容

镜像里至少固化：

- SGLang commit。
- DSpark / DeepSpec commit。
- CUDA 版本。
- PyTorch 版本。
- FlashInfer / Triton / custom kernel 版本。
- build args。
- feature flag 默认值。

### 镜像命名

建议：

```text
sglang:<sglang-version>-dspark-<feature-version>-cuda<version>
```

或者内部 registry：

```text
registry/sglang-dspark:<date>-<sglang-sha>-<dspark-sha>
```

## 6. 正确性验证

### baseline 对齐

先跑：

- greedy decode。
- fixed seed sampling。
- 单请求。
- 多请求 batch。
- 长 prompt。
- 不同输出长度。

验证：

- token 是否一致或在预期范围内。
- acceptance / rejection 逻辑是否正确。
- rejected token 是否不会污染 KV Cache。
- stop token / EOS 是否正确。
- streaming 输出顺序是否正确。

### 模型质量

Speculative decoding 理论上应保持 target model 分布。如果实现不严格，可能引入质量漂移。

至少检查：

- deterministic case。
- 常见 benchmark。
- 业务样例。
- 长上下文样例。

## 7. 性能验证

不要只看平均加速。

必须记录：

- acceptance rate。
- average accepted tokens per step。
- drafter latency。
- verifier latency。
- TTFT。
- TPOT。
- output tokens/s。
- TPM。
- GPU utilization。
- KV Cache usage。
- memory overhead。
- P99 latency。

关键判断：

```text
真实收益 = target decode step 减少带来的收益
         - drafter 成本
         - verify 成本
         - KV / memory / scheduler 额外成本
```

如果 acceptance rate 不够高，或者 drafter 太重，可能整体更慢。

## 8. SLO Goodput 验证

线上最终看：

```text
满足 SLO 的 goodput 是否提高
```

实验组合：

- baseline SGLang。
- SGLang + DSpark。
- 短请求。
- 长请求。
- 长短混合。
- 不同 batch / concurrency。
- 不同 output length。

判断：

- 平均 TPOT 是否降低。
- P99 是否恶化。
- GPU 利用率是否提高。
- SLO violation rate 是否下降。
- 如果 P50 变好但 P99 变差，不能直接上线。

## 9. 多卡和复杂场景

单卡有效后，再看：

- TP 下 drafter 和 verifier 是否使用同一并行组。
- PP 下 verify 是否放大 bubble。
- EP/MoE 下 drafter 是否也需要 expert routing。
- CP/DCP 下验证多个 token 是否增加 KV/context 通信。
- PD 下 drafter 放在 Decode worker 还是独立 worker。

不要直接从单卡 prototype 跳到生产多卡。

## 10. Metrics 和观测

新特性必须加 metrics：

- feature enabled。
- draft tokens。
- accepted tokens。
- rejected tokens。
- acceptance rate。
- drafter time。
- verifier time。
- fallback count。
- error count。
- SLO violation。

没有这些指标，就无法判断新特性是否真的有效。

## 11. 代码维护策略

### Patch 管理

不要长期维护一坨不可解释的 fork。

建议：

- 单独 feature branch。
- patch 文件清晰。
- 每个改动对应 issue / design note。
- 保留 upstream commit。
- 定期 rebase。

### Upstream 可能性

如果实现通用：

- 尝试向 SGLang 提 issue / PR。
- 或至少保持接口风格接近 upstream。

如果实现业务特定：

- 用 feature flag 隔离。
- 文档明确使用条件。

## 12. DSpark 类特性的集成检查表

```text
论文 / 代码：
- [ ] 明确 drafter / verifier 架构
- [ ] 明确 acceptance 逻辑
- [ ] 明确依赖版本

SGLang 接入：
- [ ] 找到 speculative decoding 入口
- [ ] 找到 decode loop
- [ ] 找到 KV Cache update 位置
- [ ] 找到 sampling / streaming 输出位置

镜像：
- [ ] 固定 SGLang commit
- [ ] 固定 DSpark / DeepSpec commit
- [ ] 固定 CUDA / PyTorch / Triton / FlashInfer
- [ ] 构建 feature image

验证：
- [ ] 单请求正确性
- [ ] batch 正确性
- [ ] rejected token 不污染 KV
- [ ] streaming 正确
- [ ] 多卡兼容性

性能：
- [ ] acceptance rate
- [ ] drafter overhead
- [ ] verifier overhead
- [ ] TPOT
- [ ] TPM
- [ ] SLO goodput
- [ ] P99

上线：
- [ ] feature flag
- [ ] metrics
- [ ] fallback
- [ ] rollback image
```

## 一句话总结

```text
把论文特性加入 SGLang 镜像，本质不是“装进镜像”，
而是把一个新算法接入 serving runtime 的请求、调度、KV、模型执行、metrics 和回退体系。
```
