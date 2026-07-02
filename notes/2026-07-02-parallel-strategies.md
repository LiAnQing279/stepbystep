# 分布式推理并行策略对照表

## 总览

| 策略 | 全称 | 切什么 | 解决什么 | 通信模式 | 通信频率 |
|------|------|--------|----------|----------|----------|
| DP | Data Parallel | 请求/batch | 吞吐扩展 | AllReduce（梯度同步，推理时少用） | 低 |
| TP | Tensor Parallel | 权重矩阵（列/行切） | 单请求延迟、显存不够放一张卡 | AllReduce / AllGather | 每层 2 次 |
| PP | Pipeline Parallel | 层 | 显存不够、卡数多 | P2P（层间传激活） | 每 micro-batch 每层边界 1 次 |
| EP | Expert Parallel | MoE 的 Expert | Expert 太多放不下一张卡 | All-to-All | 每 MoE 层 2 次 |
| SP | Sequence Parallel | 序列维度（非 Attention 部分） | 长序列的 LayerNorm/Dropout 显存 | 与 TP 共用，无额外通信 | 跟 TP 一致 |
| CP | Context Parallel | 序列维度（Attention 部分） | 超长上下文的 KV Cache 显存 | Ring AllGather / P2P | 每 Attention 层 |
| DCP | Decode Context Parallel | Decode 阶段的 KV Cache | Decode 阶段超长 KV Cache | P2P / AllGather | 每 Decode step |

## 逐一详解

### DP（Data Parallel）

- **什么时候用**：模型能放进单卡/单组 TP，需要扩吞吐。
- **什么时候不用**：模型太大放不进一张卡。
- **牺牲什么**：每张卡都要放完整模型副本，显存利用率低。
- **推理场景特点**：推理的 DP 通常不需要梯度同步，本质是多实例独立服务 + 前端负载均衡。

### TP（Tensor Parallel）

- **什么时候用**：模型放不进单卡，且对延迟敏感（推理最常用）。
- **什么时候不用**：卡间带宽不够（跨节点 TP 通常不划算）。
- **牺牲什么**：每层都要 AllReduce/AllGather，对 NVLink 带宽要求高。
- **推理场景特点**：TP=2/4/8 是单机推理的标准配置。TP 越大，通信占比越高，收益递减。
- **硬件要求**：同节点 NVLink 连接（>= 300 GB/s 双向）。

### PP（Pipeline Parallel）

- **什么时候用**：卡数太多不适合全 TP，或跨节点带宽有限。
- **什么时候不用**：在线推理对延迟敏感（pipeline bubble 增加延迟）。
- **牺牲什么**：引入 pipeline bubble；micro-batch 少时 GPU 利用率低。
- **推理场景特点**：推理通常只有 1 个 micro-batch（每个请求），bubble 严重。适合离线批处理或配合 TP 混合使用。
- **常见组合**：TP within node + PP across nodes。

### EP（Expert Parallel）

- **什么时候用**：MoE 模型，Expert 数量多，单卡放不下所有 Expert。
- **什么时候不用**：Dense 模型没有 Expert。
- **牺牲什么**：All-to-All 通信，且通信量取决于路由结果（不可预测）。
- **推理场景特点**：Expert 负载不均衡时部分卡空等，需要 load balancing 策略。
- **硬件要求**：高带宽互联（IB/NVLink），否则 All-to-All 成为瓶颈。

### SP（Sequence Parallel）

- **什么时候用**：已经用了 TP，想进一步节省 LayerNorm 和 Dropout 的激活显存。
- **什么时候不用**：序列不长、显存不紧张时没必要。
- **牺牲什么**：几乎无额外开销（复用 TP 的通信）。
- **推理场景特点**：推理没有 Dropout，SP 的收益主要在 LayerNorm 激活的显存节省，长序列 Prefill 时有意义。

### CP（Context Parallel）

- **什么时候用**：超长上下文（128K+），单卡/TP 组放不下 KV Cache。
- **什么时候不用**：上下文长度正常（<= 32K）。
- **牺牲什么**：Attention 计算需要跨卡 gather KV，增加通信。
- **推理场景特点**：把序列切成多段分布到不同卡，每张卡只存部分 KV Cache。Ring Attention 是典型实现。

### DCP（Decode Context Parallel）

- **什么时候用**：Decode 阶段 KV Cache 太大，单 TP 组放不下。
- **什么时候不用**：序列不长或 Prefill 阶段（Prefill 用 CP 更合适）。
- **牺牲什么**：每个 Decode step 都要跨卡通信 KV。
- **推理场景特点**：和 CP 类似但专门针对 Decode 阶段优化，减少不必要的重复计算。

## 组合策略示例

```text
场景：Llama-70B，8 卡单机
→ TP=8，无需 PP/EP

场景：Mixtral 8x22B，2 机 16 卡
→ TP=8（机内），EP=2（跨机），每机放 4 个 Expert

场景：Dense 405B，4 机 32 卡
→ TP=8（机内），PP=4（跨机）

场景：超长上下文 128K+，Llama-70B，8 卡
→ TP=4 + CP=2，或 TP=8 + 对 KV Cache 分段管理
```

## 决策树

1. 模型能放进单卡？→ DP 扩吞吐即可。
2. 放不进单卡？→ TP 优先（同节点 NVLink）。
3. TP=8 还不够？→ 加 PP（跨节点）。
4. 是 MoE 模型？→ EP 处理 Expert，TP 处理 non-Expert 部分。
5. 超长上下文 KV Cache 放不下？→ CP/DCP。
6. 要扩吞吐？→ 外层套 DP。
