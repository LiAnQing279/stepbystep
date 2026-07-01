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
