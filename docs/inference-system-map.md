# 推理系统知识地图

这张地图用于组织所有推理相关知识。以后看到论文、框架、模型、硬件、优化方案，都先把它放到对应层级，再分析它解决的瓶颈。

## 总图

```text
                 Transformer
                      |
        +-------------+-------------+
        |             |             |
      Dense          MoE         Linear
        |             |             |
        +-------------+-------------+
                      |
                推理执行 Runtime
                      |
        +-------------+-------------+
        |             |             |
    Attention        MLP        Sampling
        |             |             |
        +-------------+-------------+
                      |
                  Kernel 层
           CUDA / Triton / Cutlass
                      |
        +-------------+-------------+
        |             |             |
    KV Cache      Parallel    Communication
        |             |             |
     LMCache      TP/EP/PP    NCCL/DeepEP
                      |
                Runtime Engine
          vLLM / SGLang / TRT-LLM
                      |
             Scheduler / Deployment
          PD / K8s / Autoscaling
                      |
                  GPU / Network
    Hopper / Blackwell / 国产 GPU / RDMA
```

## 第一层：模型结构

关注问题：

- 模型结构决定了哪些计算路径？
- Prefill 和 Decode 的瓶颈是否不同？
- Attention、MLP、MoE Router、Expert 计算分别占多少？
- KV Cache 规模如何随上下文长度增长？
- 新结构是否改变现有 Runtime 的调度假设？

主要类别：

- Dense Transformer。
- MoE。
- MLA。
- Linear Attention。
- Hybrid Architecture。

锚点案例：

> **MoE vs Dense 的部署差异**：Dense 模型（如 Llama-70B）的计算量均匀分布在每一层，瓶颈通常在显存带宽（Decode 阶段 memory-bound）。MoE 模型（如 Mixtral 8x7B）虽然总参数量大，但每个 token 只激活部分 Expert，计算量接近同规模 Dense 的 1/N，但带来了 Expert 路由不均匀和 All-to-All 通信问题。看到新模型时，先判断它是 compute-bound 还是 memory-bound，再决定并行策略。

## 第二层：Kernel

关注问题：

- 热点算子是什么？
- 算子瓶颈是 Compute Bound 还是 Memory Bound？
- 是否受限于显存带宽、寄存器、共享内存、Tensor Core 或访存模式？
- 是否需要融合算子？
- 是否需要按硬件代际重写？

技术归类：

- FlashAttention / FlashInfer：Attention Kernel。
- Triton：Kernel DSL / Compiler。
- Cutlass：CUDA Kernel Template Library。
- CUDA：底层编程模型。

锚点案例：

> **FlashAttention 为什么有效**：标准 Attention 需要把 N×N 的注意力矩阵写入 HBM 再读回，显存带宽成为瓶颈。FlashAttention 把计算分块（tiling），在 SRAM 中完成 softmax 的在线归约，避免中间矩阵落地到 HBM。本质上是用多次计算换少次访存——在 memory-bound 场景下，减少访存比减少计算更有效。

## 第三层：KV Cache

关注问题：

- KV Cache 如何分配？
- 如何分页、复用、压缩、迁移或卸载？
- 长上下文和高并发下显存压力如何变化？
- Prefill 和 Decode 是否需要不同缓存策略？
- PD 分离后 KV Cache 如何跨节点管理？

技术归类：

- PagedAttention：KV Cache 分页管理。
- LMCache：KV Cache 复用和卸载。
- Mooncake：KV Cache / 分布式缓存相关方向。

锚点案例：

> **PagedAttention 解决了什么**：传统实现为每个请求预分配最大长度的连续显存，导致大量碎片和浪费（实测浪费率可达 60-80%）。PagedAttention 借鉴操作系统虚拟内存的分页思想，把 KV Cache 切成固定大小的 block，按需分配、非连续存储。这让 vLLM 能同时服务更多请求（更高并发），是 vLLM 相比朴素实现吞吐提升 2-4x 的核心原因之一。

## 第四层：Parallel

关注问题：

- 并行策略解决的是显存、计算还是吞吐问题？
- 通信量如何随模型结构和 batch size 变化？
- 哪些并行方式会增加延迟？
- 哪些并行方式依赖高带宽网络？
- 多种并行方式组合时，瓶颈转移到哪里？

并行方式：

- DP：Data Parallel。
- TP：Tensor Parallel。
- EP：Expert Parallel。
- PP：Pipeline Parallel。
- SP：Sequence Parallel。
- CP：Context Parallel。
- DCP：Decode Context Parallel 或相关变体，需要结合具体框架定义。

锚点案例：

> **TP vs PP 的取舍**：TP 把每一层的权重切分到多卡，每次前向都需要 AllReduce/AllGather，通信频率高但延迟不增加（每个 token 经过所有层只需一次前向）。PP 把不同层放到不同卡，通信只在层边界发生（频率低），但引入 pipeline bubble——除非 micro-batch 足够多否则有卡在空等。推理场景下通常优先 TP（延迟敏感），PP 在卡数很多或跨节点带宽不足时才考虑。

## 第五层：Communication

关注问题：

- 通信发生在推理的哪个阶段？
- 通信模式是 AllReduce、AllGather、ReduceScatter、AllToAll 还是 P2P？
- 网络带宽、延迟、拓扑和拥塞如何影响性能？
- MoE 的 Expert Dispatch 是否成为瓶颈？
- 跨节点场景下是否需要 RDMA、NVSHMEM 或专用通信库？

技术归类：

- NCCL：GPU Collective Communication。
- NVSHMEM：GPU 侧共享内存通信模型。
- RDMA：低延迟网络能力。
- DeepEP：MoE Expert Parallel 通信优化。

锚点案例：

> **MoE 的 All-to-All 瓶颈**：MoE 模型在 EP 模式下，每个 token 的 Expert 分配结果不同，需要 All-to-All 通信把 token 发送到对应 Expert 所在的卡。这与 TP 的 AllReduce 不同——AllReduce 通信量固定且可预测，All-to-All 的通信量取决于路由结果，负载可能不均。DeepEP 通过优化通信调度和利用 RDMA/NVSHMEM 来降低这个开销。

## 第六层：Runtime Engine

关注问题：

- 请求如何进入系统？
- Batch 如何构造？
- Prefill 和 Decode 如何调度？
- KV Cache 如何管理？
- Kernel 如何被调用？
- 分布式并行如何组织？
- 失败、超时、回收和限流如何处理？

技术归类：

- vLLM：高吞吐推理 Runtime。
- SGLang：面向结构化生成和高性能推理的 Runtime。
- TensorRT-LLM：偏编译和深度优化的 NVIDIA 推理栈。
- LMDeploy：模型部署和推理工具链。

锚点案例：

> **vLLM 的 Continuous Batching**：传统 static batching 必须等一个 batch 内所有请求都完成才能开始下一个 batch，短请求被长请求拖住。vLLM 的 Continuous Batching 允许请求在任意时刻加入和退出正在执行的 batch——一个请求生成完 EOS 就立刻释放资源，新请求立刻填入。这让 GPU 利用率从 static batching 的 30-50% 提升到 70-90%+。

## 第七层：Scheduler / Deployment

关注问题：

- 请求级、token 级和实例级调度如何协同？
- Prefill 和 Decode 是否分离？
- Autoscaling 指标是什么？
- 多租户如何隔离？
- 如何处理流量突刺、长尾请求和优先级？
- Benchmark 结果如何转化为线上容量规划？

技术归类：

- Continuous Batching。
- Chunked Prefill。
- Speculative Decoding。
- PD Disaggregation。
- Kubernetes。
- Autoscaling。
- Load Balancing。

锚点案例：

> **PD 分离的动机**：Prefill 是 compute-bound（大量矩阵乘法），Decode 是 memory-bound（每步只生成一个 token，瓶颈在读取 KV Cache 和权重）。两者混合在同一 GPU 上时互相干扰——Prefill 占据计算资源导致 Decode 延迟抖动，Decode 请求占据显存导致能并发的 Prefill 变少。PD 分离把两个阶段放到不同的 GPU 池，各自按最优配置运行，代价是中间需要传输 KV Cache。

## 第八层：Hardware / Network

关注问题：

- GPU 架构决定了哪些 Kernel 上限？
- 显存容量和带宽如何约束模型部署？
- NVLink、PCIe、IB、RoCE 如何影响并行策略？
- 国产 GPU 的编译链、算子库、通信库有哪些差异？
- 新硬件出现后，旧 Runtime 的哪些假设会失效？

硬件范围：

- NVIDIA Hopper。
- NVIDIA Blackwell。
- 国产 GPU。
- InfiniBand。
- RoCE。
- RDMA 网络。

锚点案例：

> **Hopper vs Blackwell 对推理的影响**：Hopper（H100）引入了 FP8 Tensor Core 和 TMA（Tensor Memory Accelerator），让 Kernel 可以用硬件直接管理 shared memory 的异步拷贝。Blackwell（B200）进一步提升了 HBM 带宽和卡间 NVLink 带宽。对推理来说，HBM 带宽提升直接改善 Decode 阶段的 memory-bound 瓶颈，NVLink 带宽提升降低 TP 的 AllReduce 开销。判断硬件升级对推理的收益时，先看当前瓶颈是带宽还是算力。

## 三个固定问题

每次调研新技术时，必须回答：

1. 它属于这张地图的哪一层？
2. 它解决了这一层的什么瓶颈？
3. 为什么现在这个瓶颈变得重要了？

## 记录模板

```text
技术名称：

所属层级：

解决的瓶颈：

为什么现在重要：

依赖条件：

适用场景：

不适用场景：

和已有技术的关系：

我需要补的基础：

下一步实验：
```
