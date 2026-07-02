# 实验：vLLM vs SGLang 部署 Qwen2-7B 性能对比

## 实验目标

在相同硬件上对比 vLLM 和 SGLang 部署 Qwen2-7B 的推理性能，建立第一个可复现的 Benchmark 基线。

重点回答：
1. 两者在 Prefill 和 Decode 阶段的吞吐和延迟差异有多大？
2. 差异背后的系统原因是什么？（调度策略？Kernel？KV Cache 管理？）
3. 在什么负载模式下差异最显著？

## 环境要求

- GPU：单卡 A100-80G（或 A800-80G）
- 模型：Qwen2-7B（或 Qwen2.5-7B）
- 框架版本：vLLM latest, SGLang latest
- Python：3.10+
- CUDA：12.x

## 测试维度

### 维度 1：固定输入长度，变 batch size

| 参数 | 值 |
|------|-----|
| 输入长度 | 512 tokens |
| 输出长度 | 256 tokens |
| 并发数 | 1, 4, 8, 16, 32, 64 |

### 维度 2：固定并发，变输入长度

| 参数 | 值 |
|------|-----|
| 并发数 | 16 |
| 输入长度 | 128, 512, 2048, 8192 |
| 输出长度 | 256 tokens |

### 维度 3：混合负载

| 参数 | 值 |
|------|-----|
| 输入长度 | 随机 128-4096 |
| 输出长度 | 随机 64-512 |
| 并发数 | 32 |
| 持续时间 | 5 分钟 |

## 指标采集

- **Throughput**：tokens/s（区分 Prefill 和 Decode）
- **Latency**：P50/P90/P99 TTFT（Time to First Token）和 TPOT（Time per Output Token）
- **GPU 利用率**：nvidia-smi 采样
- **显存使用**：峰值和稳态

## 工具

- Benchmark 客户端：框架自带 benchmark 脚本 或 自定义异步请求客户端
- Profile：`torch.profiler` 或 `nsys`（抓一次完整请求生命周期）
- 监控：`nvidia-smi dmon` 或 `dcgm`

## 预期产出

1. 一张对比表格（Throughput + Latency × 3 个维度）
2. 一段分析：差异在哪一层产生的？
3. 一次 Profile 截图或火焰图，标注关键热点
4. 结论：什么场景下选哪个框架

## 执行步骤

```bash
# 1. 环境准备
pip install vllm sglang[all]

# 2. 启动 vLLM
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-7B \
  --tensor-parallel-size 1 \
  --max-model-len 8192

# 3. 启动 SGLang
python -m sglang.launch_server \
  --model-path Qwen/Qwen2-7B \
  --tp 1

# 4. 运行 Benchmark（分别对两个端口）
# TODO: 补充具体 benchmark 脚本

# 5. 采集 Profile
# TODO: 补充 nsys 或 torch.profiler 命令
```

## 状态

- [ ] 环境准备
- [ ] vLLM Benchmark 完成
- [ ] SGLang Benchmark 完成
- [ ] Profile 采集
- [ ] 结果分析
- [ ] 写入 notes/
