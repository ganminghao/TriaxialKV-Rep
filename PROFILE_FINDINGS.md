# 混合精度 KV cache 的吞吐瓶颈分析（单卡 H100）

日期：2026-09-17　　硬件：单张 NVIDIA H100 80GB HBM3（HBM 峰值 3.35 TB/s）

这份报告是"做一个通用、高效、可落地的混合精度 KV cache 推理系统"的第一步：
先把本仓库里已有的 TriAxialKV（INT2/INT4 混合精度）原型 profile 一遍，
搞清楚这类系统的瓶颈在哪、离能上线的 serving 系统差多远。

分析框架参考 *Rethinking Key-Value Cache Compression Techniques for Large Language
Model Serving*（MLSys 2025, arXiv 2503.24000）。那篇文章的核心论点是：
**省显存不等于提吞吐**，现有 KV 压缩方法没有针对生产级 serving 做优化。
本报告用实测数据把这个论点在我们自己的实现上量化了一遍。

**一句话结论：这套 INT2/INT4 方案在吞吐上不但没有赢，而且比 bf16 基线慢 2.5 到 2.9 倍；
问题不是"kernel 还没调优"，而是四个层面的结构性缺陷。**

---

## 目录

1. [实验设置](#1-实验设置)
2. [瓶颈一：解码 kernel 是发射瓶颈，不是带宽瓶颈](#2-瓶颈一解码-kernel-是发射瓶颈不是带宽瓶颈)
3. [瓶颈二：容量收益有天花板，超过 2 倍压缩基本没有回报](#3-瓶颈二容量收益有天花板超过-2-倍压缩基本没有回报)
4. [瓶颈三：静态分池让有效容量变成 min 而不是 sum](#4-瓶颈三静态分池让有效容量变成-min-而不是-sum)
5. [瓶颈四：prefill 路径把省下来的又还回去了](#5-瓶颈四prefill-路径把省下来的又还回去了)
6. [端到端汇总](#6-端到端汇总)
7. [对新系统的设计启示](#7-对新系统的设计启示)
8. [如何复现](#8-如何复现)

---

## 1. 实验设置

| 项目 | 值 |
| --- | --- |
| GPU | 单张 H100 80GB HBM3，驱动 590.48.01 |
| 引擎 | 本仓库（SGLang v0.5.10 分支），python 环境 `/tmp/envs/sglang0510` |
| Kernel 微基准几何 | `H_kv=8, D=128, GQA group=8`（Qwen3-32B 的解码几何，也是论文的目标模型） |
| 端到端模型 | Qwen3-8B（36 层，8 个 KV head，D=128，GQA group=4） |
| 端到端配置 | `--disable-radix-cache --chunked-prefill-size 8192 --mem-fraction-static 0.85~0.88` |
| 压测客户端 | `sglang.bench_serving --dataset-name random-ids`（纯合成，不需要联网） |

### 对照组的选择

**基准线必须是 fp8，不是 bf16。** 本仓库原来的微基准拿 SGLang 自带的 Triton bf16 kernel
当基线，这会低估真实差距：

- SGLang 在 Hopper 上默认的解码后端是 **FA3**（`sgl_kernel.flash_attn.flash_attn_with_kvcache`），
  不是 Triton kernel。实测 Triton bf16 只有 FA3 的 0.82~0.85 倍。
- SGLang 自带的 `--kv-cache-dtype fp8_e4m3` 是一个已经上线的 2 倍压缩方案，
  它在**每一个维度上都严格优于 bf16**（见第 6 节）。任何新方案要落地，要赢的是它。

所以本报告的对照组是：`fa3-bf16`、`fa3-fp8`、`triton-bf16`、TriAxialKV（mixed / 纯 INT4 / 纯 INT2）。

### 关于 profiling 工具

这台机器上 `ncu` 取不到 GPU 性能计数器（`ERR_NVGPUCTRPERM`，容器内没有 `modprobe` 可以解除限制），
`nsys` 没有安装。替代方案：把 `TRITON_CACHE_DIR` 指到一个干净目录，跑一次 kernel，
然后对落盘的 `.cubin` 跑 `cuobjdump -res-usage` 和 `nvdisasm -c`，
就能拿到寄存器数、shared memory、以及完整的 SASS 指令流。
这条路拿到的信息对本问题反而更直接（见第 2 节）。

FlashInfer 的 JIT 在这个环境里编译不了（它从 `sys.prefix` 而不是 `$CUDA_HOME` 找 `nvcc`，
而 `nvcc` 只存在于另一个 conda 环境里）。FA3 是 AOT 编译好的，不需要工具链，所以用 FA3。
fp8 KV 走 FA3 时 q 也必须转成 fp8 并传 `k_descale` / `v_descale`。

---

## 2. 瓶颈一：解码 kernel 是发射瓶颈，不是带宽瓶颈

### 2.1 Roofline

`bs=32, seq_len=8192`（每次调用扫 262,144 个 KV token）：

| kernel | 字节/token | us/call | 有效带宽 | **占 HBM 峰值** | vs fa3-bf16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fa3-bf16` | 512 | 355.8 | 3018 GB/s | **90.1%** | 1.00x |
| `fa3-fp8` | 256 | 258.0 | 2081 GB/s | 62.1% | 1.38x |
| `triton-bf16` | 512 | 436.4 | 2460 GB/s | 73.4% | 0.82x |
| `tri-int4` | 320 | 1401.6 | 239 GB/s | 7.1% | 0.25x |
| **`tri-mixed`** | 216 | 1331.2 | **167 GB/s** | **5.0%** | **0.27x** |
| `tri-int2` | 192 | 1316.9 | 153 GB/s | 4.6% | 0.27x |

FA3 bf16 在大 batch 下能吃到 **90% 的 HBM 峰值**，是一个真正的带宽受限 kernel。
混合精度 kernel 卡在 **5%**。

完整的 bs × seq 扫描（bs ∈ {1,2,4,8,16,32,64} × seq ∈ {2048, 8192, 32768}）见
`benchmark/mixed_precision_profile/roofline.json`。`tri-*` 三条曲线在整个范围内
都稳定在 4.3% ~ 7.2%，**不随 batch 或序列长度改善**。

### 2.2 决定性证据：每个 KV token 的成本与位宽无关

把时间除以扫过的 token 数（`us / (bs * seq)`）：

```
variant       B/token    bs8   bs16   bs32   bs64   mean ns   vs bf16
fa3-bf16          512   1.52   1.40   1.36   1.33      1.40     1.00x
fa3-fp8           256   1.12   1.02   0.98   0.95      1.02     0.73x
triton-bf16       512   1.79   1.73   1.66   1.63      1.70     1.22x
tri-int4          320   5.55   5.44   5.35   5.31      5.41     3.86x
tri-mixed         216   5.38   5.20   5.08   5.07      5.18     3.70x
tri-int2          192   5.25   5.13   5.02   5.03      5.11     3.64x
```

INT2 比 INT4 少读 **1.67 倍**的字节（192 vs 320 B/token），时间只差 **6%**。

> **这是本报告最重要的一条观察：时间和读了多少字节完全脱钩。
> 压缩率再提高也不会让这个 kernel 变快一点。**

而 `fa3-bf16` 多读 4.7 倍字节，每个 token 只要 1.40 ns。

### 2.3 SASS 层面的解剖：为什么

用 `cuobjdump -res-usage` 和 `nvdisasm` 对比两个 stage-1 kernel：

| | SGLang bf16 `_fwd_grouped_kernel_stage1` | TriAxialKV `_fwd_grouped_kernel_stage1_mixed` |
| --- | ---: | ---: |
| **寄存器 / 线程** | **88** | **250** |
| shared memory | 21,504 B | 22,784 B |
| SASS 指令总数 | 746 | **4,093（5.5 倍）** |
| 位运算占比 | 11.0% | **33.1%** |
| —— | LOP3 51, SHF 26 | **PRMT 556, LOP3 421, SHF 276** |
| 类型转换 (I2F/F2F) | 0.8% | 7.1%（I2FP 200） |
| 张量核指令 (HMMA) | 16 | **8** |
| global load 指令数 | 20 | **321** |
| load 指令宽度 | 16×`LDG.E.64` + 1×`LDG.E.128` | **188×`LDG.E.U16` + 128×`LDG.E.U8`** |
| **平均每条 load 取的字节** | ~8 B | **1.6 B** |
| 寄存器限制的 CTA/SM | 5（20 warps） | **2（8 warps）** |
| **occupancy** | **31.2%** | **12.5%** |

三个问题同时发生：

1. **访存指令粒度塌了 5 倍。** 压缩本应让你读更少的字节，结果 bit-packing 的 layout
   逼着 kernel 用字节级 / 半字级的标量 load。**每交付一个字节要发 5 倍的访存指令**，
   LSU 的发射率成了真正的瓶颈，HBM 根本没吃饱。
   最典型的是 INT2 K 的 meta：per-(page, head, channel) 的 scale/min，
   取一个 tile 要发 8 条标量 `tl.load` 加 `tl.where` 选择。
2. **寄存器压力把 occupancy 砍到 12.5%**（每个 SM 只有 8 个 warp）。
   这点 warp 数掩盖不住 HBM 的访存延迟，于是延迟直接暴露在关键路径上。
   这也解释了为什么增大 batch 完全救不了它 —— 问题在 SM 内部，不在并行度不够。
3. **张量核基本没用上**（8 条 HMMA vs bf16 的 16 条）。QK^T 退化成了 FFMA 标量乘加。

补充：混合块（一个 `BLOCK_N` 里同时有 INT2 和 INT4 的 slot）会同时加载两条路径再做 blend，
在这种块上做 2 倍的无效工作。这解释了为什么 `tri-mixed` 并不比 `tri-int4` 或 `tri-int2` 快。

### 2.4 追平基线需要多少加速

| 目标 | 需要的 kernel 加速 |
| --- | ---: |
| 追平 `fa3-bf16` 的延迟 | **3.7 倍** |
| 追平 `fa3-fp8`（真实生产基线） | **5.1 倍** |

注意这只是"追平延迟"，还不算把省下的显存换成更大 batch 之后的净收益。

---

## 3. 瓶颈二：容量收益有天花板，超过 2 倍压缩基本没有回报

这一条对是否立项最关键。

做法：启动一个服务器，在它上面扫并发（输入 4096 / 输出 1024，即每个请求 5120 个 token），
看吞吐随并发的变化曲线。

| 并发 | bf16（容量 1.0×）| fp8（2.0×）| TriAxial（3.3×）|
| ---: | ---: | ---: | ---: |
| 8   | 788  | 807  | 465 |
| 16  | 1272 | 1349 | 611 |
| 32  | 1789 | 1994 | 743 |
| 64  | **2226**（run 44）| 2588（run 44）| 813 |
| 128 | 2189（run **56**）| **3036**（run 80）| 869（run 80）|
| 256 | 2190（run 56）| 3026（run **127**）| **879**（run 104）|

单位：输出 tok/s。`run` = 服务端实际同时在跑的请求数（从 `#running-req` 日志取平均）。
KV 容量：bf16 374,562 tok（约 73 个请求）／fp8 749,125（146）／TriAxial 1,223,072（238）。

三个读数：

1. **吞吐在 run_req ≈ 80 就饱和了。** fp8 从 run 80 涨到 run 127（多 60% 的在跑请求），
   吞吐**零增长**（3036 → 3026）。bf16 从 run 44 到 run 56 同样零增长（2226 → 2189）。
   并发继续往上加只会让 TTFT 爆炸（bf16 在并发 256 下 TTFT 47.5 秒），吞吐纹丝不动。
2. **因此压缩的边际价值是：1×→2× 容量 = +36% 吞吐；2×→4× 容量 = +0%。**
   这是在 kernel 完美（fp8 比 bf16 还快）的前提下测出来的上界。
3. **TriAxial 更糟：它买来的容量自己用不掉。** 它有 238 个请求的容量，
   但它自己的 kernel 在 run_req ≈ 100 就先饱和在 879 tok/s 了。
   它的压缩率相对于它自己的 kernel 是严重过配的。

> **压缩只在「bf16 能撑住的 batch ≪ 算力饱和的 batch」这个区间里才有价值。**
>
> 8B 模型 + 单张 H100 不在这个区间（bf16 已经能跑 44~56 个请求）。
> 论文的 32B + 单卡在这个区间（README 记录 bf16 只能跑 3~4 个请求），
> 这也是它能报出 1.41 倍吞吐的原因。
>
> 推论：新系统的设计目标应该是**"用最少的 bit 和最快的 kernel 刚好够到算力饱和点"**，
> 而不是"最大化压缩率"。立项前必须先测出目标部署（模型大小 / 上下文长度 / 并发）的饱和点。

---

## 4. 瓶颈三：静态分池让有效容量变成 min 而不是 sum

`--triaxial-int2-fraction f` 把显存静态切成 INT2 和 INT4 两块物理池。
设 workload 真实的 INT2 占比为 `a`，则有效容量是

```
capacity = min( n2 / a , n4 / (1 - a) )
```

而不是两个池的和。只要 `f` 和 `a` 不匹配，容量就由匹配最差的那个池决定。

### 4.1 tagger 产出的 INT2 占比 `a` 波动极大

实测本仓库的 tagger（默认策略）在 Qwen3 chat template 上的输出：

| workload | `a`（INT2 占比） |
| --- | ---: |
| random token / 单轮对话 / 2 轮对话 | **0.00** |
| 4 轮对话 | 0.25 |
| 7 轮对话 | 0.30 |
| OSWorld 重放（TRIAXIAL_README 记录） | 0.875 |

原因是默认策略把"当前轮"的全部内容判为 INT4，只有历史轮才降到 INT2。
轮数少的时候几乎没有历史，`a` 就是 0。

### 4.2 有效容量矩阵

数值按本机 Qwen3-8B 的字节预算（53.8 GiB，bf16 容量 391,726 token）算出，
表内是相对 bf16 的倍数：

| 配置 `f` ↓ ＼ 实际 `a` → | 0.00 | 0.30 | 0.50 | 0.85 | 0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.05 | **3.10×** | 0.54× | 0.33× | 0.19× | 0.17× |
| 0.25 | 2.67× | **2.96×** | 1.78× | 1.05× | 0.94× |
| 0.50 | 2.00× | 2.86× | **4.00×** | 2.35× | 2.11× |
| **0.85**（README 推荐值）| **0.73×** | **1.04×** | 1.45× | **4.85×** | 4.34× |
| 0.95 | 0.26× | 0.37× | 0.52× | 1.72× | **5.16×** |
| *理想（逐 token 分配，不分池）* | *3.20×* | *3.64×* | *4.00×* | *4.85×* | *5.16×* |

对角线是宣传数字，**一离开对角线就崩**：

- `f=0.85` 碰上一个 7 轮对话的真实 workload（`a=0.30`）只剩 **1.04×** ——
  而理想的逐 token 分配本该给 3.64×。
- `f=0.85` 碰上单轮 workload（`a=0`）只剩 **0.73×，比 bf16 还差**。
- 最后一行说明：**这些损失全部来自静态分池**。逐 token 分配在整个 `a` 范围内都能拿到 3.2~5.2×。

### 4.3 而且不是优雅降级，是直接崩

在并发 128、输入 8000 的压测下跑 `f=0.85`（workload 的 `a=0`）：

```
[08:00:27] TriAxialKV pool: 1614368 INT2 slots (50449 pages of 32) + 284898 INT4 slots
[08:00:27] KV Cache is allocated. #tokens: 1899266     <- 标称 4.85 倍容量
...
[08:00:27] Out of memory (TriAxialKV). Try to lower your batch size.
[08:00:27] Scheduler hit an exception:
RuntimeError: Out of memory (TriAxialKV). Try to lower your batch size.
```

同样的并发下，容量只有 391,726 的 bf16 跑完全程无事（run_req 稳定在 39.9）。
标称 4.85 倍容量的系统，因为 INT4 池只有 284,898 个 slot，实际只跑到 `avg running-req 14`，
然后崩掉。

### 4.4 一个结构性约束：decode 生成的 token 只能进 INT4

per-channel 量化需要凑满一整页（32 个 token）才能算出 scale/min，
这和增量解码天然不兼容。所以**解码产生的每一个 token 都只能进 INT4 池**。

结果是 INT4 池永远是稀缺资源，而它恰恰是被 `f=0.85` 切得最小的那块。
准入控制因此必须只看 INT4 池（`admission_size()`），这又让调度器变得过度保守。

---

## 5. 瓶颈四：prefill 路径把省下来的又还回去了

`TriaxialFlashInferBackend.forward_extend` 的做法是：**每一层**都把整个被 attend 的 prefix
用 `dequant_gather` 反量化成一块 bf16 scratch，把新 token 的 bf16 k/v 拼上去，
再让 FlashInfer 的 paged kernel 跑在这块 scratch 上。

### 5.1 `dequant_gather` 本身也卡在 4% 带宽

| prefix token 数 | `dequant_gather` | 等价的 bf16 gather | 慢了 | 有效带宽 |
| ---: | ---: | ---: | ---: | ---: |
| 2,048  | 79.5 us | 12.5 us | 6.4× | 127 GB/s |
| 8,192  | 290.5 us | 29.5 us | 9.8× | 139 GB/s |
| 16,384 | 569.6 us | 53.0 us | 10.7× | 142 GB/s |
| 32,768 | 1125.3 us | 98.5 us | **11.4×** | 144 GB/s |

和解码 kernel 一样的 ~140 GB/s 天花板（4% 峰值）。写入侧的量化 kernel
（`quant_int4_tokens` / `quant_int2_v_tokens` / `quant_int2_k_pages`）在 32k token 上是 313.5 us。

### 5.2 chunked prefill 会反复反量化同一段 prefix

第 i 个 chunk 要 attend 前面 `i*C` 个 token，这些 token 每次都要重新反量化一遍：

| 上下文长度 | chunk 大小 | chunk 数 | 每层反量化的 token 总量 | 相对上下文长度的放大 |
| ---: | ---: | ---: | ---: | ---: |
| 8k   | 8192 | 1  | 0 | 0× |
| 32k  | 8192 | 4  | 49,152 | 1.5× |
| 32k  | 4096 | 8  | 114,688 | 3.5× |
| 128k | 8192 | 16 | 983,040 | **7.5×** |
| 128k | 4096 | 32 | 2,031,616 | **15.5×** |

长上下文的 agentic 场景（正是这个方案的目标场景）会被这条放大直接打穿 TTFT。

### 5.3 另外两点

- **prefill 期间完全没有省显存**：scratch 是整个 batch 被 attend 窗口的全精度 bf16 副本。
- 本次端到端测试里 TTFT 没有恶化（bf16 2402 ms vs tri 2408 ms），
  但那是因为关掉了 radix cache 且输入 8000 < chunk 8192，
  每个请求只有一个 chunk、没有 prefix 要反量化。**一旦开启前缀复用或更长上下文，这条就会显现。**

---

## 6. 端到端汇总

### 6.1 解码密集（1k 输入 / 2k 输出，并发 128，128 个请求）

三组的并发（~124）和实际 batch（run_req ~95）完全一致，因此这是纯粹的 per-token 对比：

| 配置 | KV 容量 | 输出 tok/s | TPOT | 中位 ITL |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 391,726 | 5810 | 20.4 ms | 20.2 ms |
| **fp8** | 783,452（2.0×）| **6774（1.17×）** | 17.3 ms | 16.5 ms |
| TriAxialKV | 1,279,117（3.3×）| 2053 **（0.35×）** | 59.7 ms | 58.9 ms |

### 6.2 prefill 密集（8k 输入 / 256 输出，并发 32，64 个请求）

| 配置 | KV 容量 | 输出 tok/s | 总 tok/s | TPOT | TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| bf16 | 391,726 | 652 | 21050 | 38.6 ms | 2403 ms |
| **fp8** | 783,452（2.0×）| **719（1.10×）** | 23198 | 34.8 ms | 2237 ms |
| TriAxialKV | 1,899,266（4.85×）| 341 **（0.52×）** | 11005 | 82.8 ms | 2408 ms |

### 6.3 容量受限时（8k 输入 / 256 输出，并发 128，192 个请求）

| 配置 | KV 容量 | avg run_req | 输出 tok/s | TPOT |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 391,726 | 39.9 | 678 | 58.7 ms |
| **fp8** | 783,452 | 69.4 | **786（1.16×）** | 93.7 ms |
| TriAxialKV `f=0.85` | 1,899,266（标称）| 14.0 | **崩溃（OOM）** | — |
| TriAxialKV `f=0.05` | 1,279,117 | **84.3** | 359 **（0.53×）** | 272 ms |

最后一行是整个报告最清楚的一句话：
**容量收益完全兑现了（同时在跑的请求数从 39.9 涨到 84.3，2.1 倍），
但吞吐反而掉了 1.9 倍。** 显存省下来变成了 batch，可是每个 token 的 kernel 成本涨得更多，
乘积是负的。

### 6.4 fp8 是真正的基准线

`--kv-cache-dtype fp8_e4m3` 在**每一个维度上都严格优于 bf16**：
2 倍容量 + 1.10~1.17 倍吞吐 + 更低的 TTFT。
它快的根本原因是 **fp8 是硬件类型 —— 零条反量化指令，张量核直接吃**。

---

## 7. 对新系统的设计启示

### 7.1 基准线必须是 fp8，不是 bf16

论文和本实现都拿 bf16 比，这掩盖了真实差距。新方案要落地，要赢的是 fp8。
**任何低于 8 bit 的格式都要为软件反量化付费**，除非它能映射到硬件支持的类型 ——
Blackwell 的 nvfp4 / mxfp4 是这条路最自然的落点。

### 7.2 格式设计的第一约束是"能不能 128-bit 向量 load"，不是压缩率

当前 layout 逼出 `LDG.E.U8` / `LDG.E.U16`（平均 1.6 B/指令），这是死刑。
scale / min 的排布也要保证每个 tile 一两条向量 load 拿完，
而不是像现在 INT2 K 的 meta 那样需要 8 条标量 load 加 select。

### 7.3 反量化走 bit-trick，寄存器压到 ≤128

目标 occupancy ≥25%（≤128 寄存器），理想 50%（≤64）。
反量化用 `lop3 + prmt` 的 fp16 位技巧（Marlin / Machete 那套），
而不是逐元素的 shift / mask / convert / fma。
Triton 在这个问题上大概率到不了 —— 需要 CUTLASS / CuTe，配 TMA + wgmma 异步流水。
参照物：weight-only 的 dequant GEMM 已经能做到接近带宽上限，说明这条路可行。

### 7.4 放弃静态分池，改成统一字节池 + 逐 token 变长分配

有效容量必须是 sum 而不是 min（第 4.2 节最后一行：理想分配在整个 `a` 范围都有 3.2~5.2×）。
同时准入控制要基于**字节预算**而不是 slot 数，这样精度策略变化时不会崩，
只会平滑地少收几个请求。

### 7.5 量化粒度必须和增量解码兼容

per-channel 需要凑页 → 解码 token 进不去 → 稀缺池被打爆（第 4.4 节）。
要么用 per-token 分组，要么设计一个能在线增量更新 scale 的 per-channel 方案。

### 7.6 立项前先证明目标 workload 是容量受限的

第 3 节的数据说吞吐在 run_req ≈ 80 就饱和了。
先测出目标部署（模型大小 / 上下文长度 / 并发）的饱和点，再决定要几倍压缩。
**超过饱和点的压缩是纯浪费**，还要付 kernel 变慢的代价。

判据：`bf16 能撑的 batch` 显著小于 `算力饱和的 batch` 时，压缩才有价值。
这通常意味着：大模型 + 单卡（权重吃掉大部分显存）、长上下文、高并发、
或者 MHA / 高 KV head 数的模型。

### 7.7 prefill 不能走"反量化到 bf16 scratch"这条路

要么让 prefill kernel 原生吃压缩格式，
要么让前缀缓存命中的部分直接以压缩态参与计算。
否则长上下文 + chunked prefill 的重复反量化放大（最高 15.5×）会吃掉 TTFT。

### 7.8 其他落地缺口清单

- 前缀缓存里一个 token 只存一份，**位宽在第一次写入时固定**，之后复用不会改。
  这导致策略必须把"当前轮的图片"也标成 INT2（否则它变成历史后仍占 INT4 池）——
  精度策略被存储机制反向绑架了。
- 强制 `--page-size 1`，关掉分块 CUDA 图和混合分块。
- 不支持投机解码。
- 不能和 `--kv-cache-dtype` 共存。

---

## 8. 如何复现

脚本都在 `benchmark/mixed_precision_profile/`。先 `source /tmp/envs/sglang0510/triaxial_env.sh`。

```bash
cd benchmark/mixed_precision_profile
PY=/tmp/envs/sglang0510/bin/python

# 第 2.1 / 2.2 节：解码 roofline 全扫描
$PY bench_kv_decode_roofline.py --bs-list 1,2,4,8,16,32,64 \
    --seq-list 2048,8192,32768 --layers 16 --pools 4 --out roofline.json

# 第 2.3 节：寄存器数与 SASS 指令组成
#   （需要 nvdisasm / cuobjdump，在 /data/gmh/.conda/envs/quancache/bin 下）
rm -rf tcache && TRITON_CACHE_DIR=$PWD/tcache $PY kernel_anatomy.py
export PATH=/data/gmh/.conda/envs/quancache/bin:$PATH
for c in $(find tcache -name "_fwd_grouped_kernel_stage1*.cubin"); do
  cuobjdump -res-usage $c; nvdisasm -c $c > sass_$(basename $c .cubin).txt
done

# 第 5 节：prefill 路径成本
$PY bench_prefill_path.py

# 第 3 节：并发扫描（一次起服务，内部扫并发）
./sweep_conc.sh bf16 4096 1024 8,16,32,64,128,256
./sweep_conc.sh fp8  4096 1024 8,16,32,64,128,256
INT2_FRAC=0.05 ./sweep_conc.sh tri 4096 1024 8,16,32,64,128,256

# 第 6 节：单点端到端对比
./run_e2e.sh bf16 128 128 1024 2048     # 模式 请求数 并发 输入长度 输出长度
./run_e2e.sh fp8  128 128 1024 2048
INT2_FRAC=0.05 ./run_e2e.sh tri 128 128 1024 2048
# 复现第 4.3 节的崩溃：
INT2_FRAC=0.85 ./run_e2e.sh tri 192 128 8000 256
```

`sweep_conc.sh` 和 `run_e2e.sh` 的结果写到脚本同目录的 `results/` 下。
两个脚本默认用 `/data/gmh/model/Qwen3-8B`，可以用环境变量 `MODEL` 覆盖。

---

## 引用

- Wei Gao, Xinyu Zhou, Peng Sun, Tianwei Zhang, Yonggang Wen.
  *Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving.*
  MLSys 2025. arXiv:2503.24000.
- TriAxialKV 的实现与复现说明见本仓库的 `TRIAXIAL_README.md` 和 `TRIAXIAL_DESIGN.md`。

---

# 补充实验（2026-09-17 下午）

下面三章回应 `PROFILE_FOLLOWUP.md` 提出的三个缺口。前八章的数字没有改动。
新脚本和原始数据在 `benchmark/mixed_precision_profile/` 下，原始输出在它的 `results/followup_*/`。

补充实验的结论：

1. **TriAxialKV 解码比 bf16 每步多出的时间，98% 花在自研 attention kernel 上**，
   量化写入、显存管理、CPU 调度加起来不到 2%（第 9 章，Qwen3-8B）。
2. **在显存容量受限的场景（Qwen3-32B 单卡），TriAxialKV 赢 bf16，但输 fp8。**
   吞吐上限 282 tok/s，是 bf16（209）的 1.35 倍，是 fp8（362）的 0.78 倍。
   它能同时跑的请求数最多（21 个，对 bf16 的 6.7 个、fp8 的 12.8 个），
   但 attention kernel 太慢，把多出来的并发吃掉了。按 README 推荐的 `--triaxial-int2-fraction 0.85`
   配置在并发 8 时就崩溃（第 10 章）。
3. **SGLang 和最新 vLLM 的基线：解码速度（TPOT）在 bf16 下可以互换引用，吞吐不能。**
   bf16 的 TPOT 相差 5% 以内；fp8 的 TPOT vLLM 快约 20%；
   输出吞吐 vLLM 高 9%~15%（W2）和 33%~70%（W1），差距主要来自 SGLang 的请求准入和排队，不是 kernel（第 11 章）。

---

## 9. 一步推理的时间拆分（缺口 1）

### 9.1 结论

在负载 W1（输入 1024、输出 2048、并发 128）的解码中段，每一步的墙钟时间（从这一步第一个 GPU kernel 开始，到下一步第一个 GPU kernel 开始）：

| 模式 | 一步（ms） | 准备类（ms） | attention 计算 K1（ms） | 模型其余 M（ms） | 准备类占比 | K1 占比 | M 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 21.6 | 0.9 | 13.95 | 6.76 | 4.0% | 64.6% | 31.3% |
| fp8 | 19.7 | 2.8 | 10.26 | 6.69 | 14.1% | 52.0% | 33.9% |
| TriAxialKV | 63.5 | 1.7 | **55.05** | 6.80 | 2.6% | **86.7%** | 10.7% |

「准备类」= GPU 上由 CPU 准备工作发起的 kernel（P1g）+ GPU 显存管理（P2）+ KV 写入与量化（P3）
+ 反量化到临时缓冲区（P4）+ GPU 空闲（I）。CPU 端调度与输入准备（P1）单独用 CPU 毫秒报告，见 9.4 节。

这张表回答 `PROFILE_FOLLOWUP.md` 的核心问题：第 6.1 节里 TriAxialKV 的 TPOT 是 59.7 ms、bf16 是 20.4 ms，
多出来的时间落在哪里。本次抓取中两者一步相差 **41.9 ms**，拆开是：

| 类别 | TriAxialKV 比 bf16 多（ms） | 占差值 |
| --- | ---: | ---: |
| K1 attention 计算 | +41.10 | **98.1%** |
| I GPU 空闲 | +0.51 | 1.2% |
| P3 KV 写入与量化 | +0.27 | 0.7% |
| M 模型其余 | +0.04 | 0.1% |
| P1g、P2、P4 | 约 0 | 0% |
| P1 CPU 调度与输入准备（CPU 毫秒，不计入上面的和） | +0.86 | — |

**所以瓶颈就是 attention kernel 本身，不是准备工作。** 这和第 2 节 kernel 微基准的结论一致，
现在在真实引擎里得到了确认。

抓取得到的一步时间和不开计时的端到端测量吻合：bf16 21.6 ms 对第 6.1 节 TPOT 20.4 ms，
TriAxialKV 63.5 ms 对 59.7 ms。差出来的几毫秒是因为 TPOT 是整个运行的平均，
包含了 KV 更短的早期步骤和被 prefill 打断的步骤。

### 9.2 测量方法

**工具。** 这台机器上 GPU 性能计数器读不到，但 `torch.profiler` 走的是 CUPTI 的 activity 接口，
不需要计数器权限，可以用。先用一个小脚本确认了两件事：CUPTI 能记录 kernel；
**从 CUDA graph 回放出来的 kernel 也能被记录**（这点很关键，因为三种模式的解码都跑在 CUDA graph 里）。
抓取通过 SGLang 自带的 `/start_profile` 接口触发，记录 CPU 和 GPU 两种事件，
不记录 Python 调用栈、不记录张量形状，以降低开销。

**计时区间。** 给 SGLang 加了一个环境变量开关 `SGLANG_TRIAXIAL_PROFILE`
（新文件 `python/sglang/srt/utils/triaxial_profile.py`）。打开时，在调度循环的各个阶段、tagger、
分配器、元数据构造、两个显存池的写入函数、prefill 的反量化段、attention 后端的计算函数上
加 `torch.profiler.record_function` 区间，区间名字以类别开头（`P1::`、`P2::`、`P3::`、`P4::`、`K1::`、`STEP::`）。
关掉时，装饰器直接返回原函数，`with` 区间返回同一个空的上下文对象，服务路径不受影响。
**所有吞吐数字都是在关掉开关的情况下测的。** 打开时每个区间的开销实测 7.7 微秒，
一步解码大约 10 个区间，约 0.08 ms，是一步时间的 0.4%。
改动了 8 个 SGLang 文件共 48 行，现有的 37 个 TriAxialKV 单元测试全部通过。

**归类规则**（实现在 `analyze_trace.py`）：

1. 每个 GPU 事件（kernel、显存拷贝、显存填充）通过 CUPTI 的 correlation id 找到发起它的 CPU 调用，
   再找包住这个调用的最内层计时区间，按区间名字归类：`P1::` 归 P1g，`P2::` 归 P2，
   `P3::` 归 P3，`P4::` 归 P4，`K1::` 归 K1。
2. 如果最内层区间是 `STEP::`，说明这个 kernel 是从 CUDA graph 回放出来的，回放时没有 Python 代码运行，
   区间不存在。这时按 kernel 名字归类：名字含 FlashAttention、FlashInfer prefill/decode、
   TriAxialKV 的 `_fwd_grouped_kernel_stage1_mixed` / `_fwd_kernel_stage2` 的归 K1；
   `store_kvcache` 和 TriAxialKV 的三个量化 kernel 归 P3；`_dequant_gather_kernel` 归 P4。
3. 少数通用 kernel 名字在多个类别之间共用，按每层出现次数拆分（和 bf16 的 trace 对比数出来的）：
   fp8 模式下 `float8_copy_kernel` 每层跑 3 次，分别是把 k、v 转成 fp8（写入，归 P3）和把 q 转成 fp8
   （FA3 的输入，归 K1），所以 2/3 归 P3、1/3 归 K1；TriAxialKV 解码写入函数 `set_kv_buffer_int4`
   每层额外发起一次 `loc - offset` 加法、一次 `clamp_min` 和一次 `.contiguous()` 拷贝，归 P3。
4. 其余归 M（线性层、MLP、norm、RoPE、采样）。

按这套规则，三种模式的 M 分别是 6.76、6.69、6.80 ms，基本相同。M 本来就不应该随 KV 格式变化，
这可以作为归类没有串类别的一个检查。

**一步的边界。** 在 GPU 时间线上切：第 i 步从 CPU 区间 `STEP::decode` 第 i 次发起的第一个 GPU 事件开始，
到下一个 `STEP::`（不论 prefill 还是 decode）发起的第一个 GPU 事件为止。
GPU 空闲 I = 这段时间减去所有 GPU 事件时间段的并集。跳过前 2 步，每次抓取分析 56 步（prefill 30 步或 19 步）。
如果某个解码时间段里混进了 prefill 步发起的 kernel（overlap 调度下会发生，第 10 章的容量受限负载里出现过 5 次），
这个时间段整段丢弃，不参与平均。Qwen3-8B 的抓取里没有这种情况。
表里「一步」是各类别之和加 I，比实测的时间段长 0.1~0.2 ms，因为少数 GPU 事件在时间上有重叠
（例如另一个 CUDA stream 上的拷贝）。

**CUDA graph。** 三种模式的解码都用了 CUDA graph（服务日志每条 `Decode batch` 都是 `cuda graph: True`），
prefill 都是 eager 模式。所以 `PROFILE_FOLLOWUP.md` 里「如果 tri 模式没开 CUDA graph 就补一组
bf16 关 CUDA graph」的条件不成立，没有补这组数据。

### 9.3 抓取位置的影响：attention 占比随 KV 长度增长

一开始是在解码刚开始时抓的（128 个请求刚做完 prefill，每个请求的 KV 约 1100 个 token）。
后来发现这会低估 attention：W1 里每个请求的 KV 从 1024 长到 3072，平均约 2048。
所以又加了一组在解码中段抓的（等到 batch 里的 KV 总量达到 128 × 2048 = 262144 个 token 再开始，
抓取时实际是 263887~266110）。两组对比：

| 抓取位置 | 模式 | 一步（ms） | K1（ms） | M（ms） | I（ms） | P3（ms） | K1 占比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 解码开始（KV 约 1.1k） | bf16 | 16.2 | 7.76 | 6.72 | 1.60 | 0.06 | 48.0% |
| | bf16（重复一次） | 18.9 | 7.95 | 6.75 | 4.09 | 0.06 | 42.1% |
| | fp8 | 14.0 | 6.11 | 6.80 | 0.75 | 0.25 | 43.8% |
| | TriAxialKV | 37.8 | 29.39 | 6.80 | 1.21 | 0.33 | 77.8% |
| 解码中段（KV 约 2k） | bf16 | 21.6 | 13.95 | 6.76 | 0.78 | 0.06 | 64.6% |
| | fp8 | 19.7 | 10.26 | 6.69 | 2.51 | 0.24 | 52.0% |
| | TriAxialKV | 63.5 | 55.05 | 6.80 | 1.29 | 0.33 | 86.7% |

几点观察：

- **M 和 P3 不随 KV 长度变化，只有 K1 随 KV 长度线性增长。** KV 平均长度翻倍，
  bf16 的 K1 从 7.76 涨到 13.95 ms，TriAxialKV 的 K1 从 29.39 涨到 55.05 ms。
  所以上下文越长，attention kernel 在一步里的占比越高，TriAxialKV 的 kernel 劣势越突出。
- 同一个 bf16 配置重复抓两次，GPU 上的数字几乎一样（K1 7.76 对 7.95，M 6.72 对 6.75），
  但 GPU 空闲 I 从 1.60 变成 4.09 ms，CPU 端 P1 从 2.70 变成 5.94 ms。
  **GPU 端的拆分是可重复的，CPU 端的数字受这台机器的 CPU 争用影响**（见第 12 章第 1 条）。
- fp8 的 K1 比 bf16 少（10.26 对 13.95 ms），和第 2 节微基准里 fp8 kernel 比 bf16 快 1.38 倍一致。

### 9.4 CPU 端的准备工作（P1）

P1 是 CPU 上的调度和输入准备，单位是 CPU 毫秒。SGLang 用的是 overlap 调度：
GPU 在跑第 i 步时，CPU 同时在准备第 i+1 步。所以 **P1 不直接加到一步的墙钟时间上**，
它只有在比 GPU 慢的时候才会表现为 GPU 空闲 I。

还有一个陷阱：计时区间 `P1::process_batch_result`（处理上一步的结果）的大部分时间其实是
CPU 在等 GPU 算完（`cudaEventSynchronize`）。例如 bf16 解码开始时这个区间一步 14.3 ms，
其中 12.2 ms 是等待。下表的 P1 已经扣掉了区间内所有的同步等待。

W1 解码中段，每步的 CPU 工作（ms）：

| 阶段 | bf16 | fp8 | TriAxialKV |
| --- | ---: | ---: | ---: |
| 处理上一步结果（扣除等待） | 1.78 | 3.09 | 2.58 |
| 发起本步 kernel（不含下面的准备） | 1.07 | 1.13 | 1.17 |
| 组下一个 batch（含分配） | 0.29 | 0.28 | 0.29 |
| CUDA graph 回放前准备输入和元数据 | 0.21 | 0.19 | 0.28 |
| 收请求、处理输入（含 tagger） | 0.04 | 0.04 | 0.04 |
| **P1 合计** | **2.36** | **3.63** | **3.22** |
| 同一时间 GPU 空闲 I | 0.78 | 2.51 | 1.29 |

解码阶段 TriAxialKV 的 tagger 和两个池的分配器在 CPU 上的开销小到可以忽略（分配器和 tagger 在
「组下一个 batch」「收请求」这两行里，和 bf16 一样）。分配器在解码时没有发起任何 GPU kernel，P2 为 0。

### 9.5 prefill 阶段（负载 W2：输入 8000、输出 256、并发 32）

| 模式 | 分析的 prefill 步数 | 一步（ms） | P3 | P4 | K1 | M | I | K1 占比 | P1 CPU（ms） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 30 | 228.5 | 0.93 | 0 | 40.57 | 175.63 | 11.27 | 17.8% | 113.9 |
| fp8 | 30 | 216.8 | 2.37 | 0 | 36.29 | 171.58 | 6.49 | 16.7% | 100.1 |
| TriAxialKV | 19 | 255.4 | 4.04 | 8.64 | 72.95 | 163.45 | 6.06 | 28.6% | 224.2 |

每个 prefill 步处理约 8000 个 token（抓取期间平均 bf16 7996、TriAxialKV 7921）。

- **prefill 的一步里模型其余部分（线性层为主）占 64%~79%，attention 不是主要开销。**
- TriAxialKV 的 prefill attention（K1）比 bf16 慢 1.8 倍：72.95 对 40.57 ms。原因是它的 prefill
  不走 FA3，而是在临时 bf16 缓冲区上跑 FlashInfer 的 paged prefill kernel。只看这个 kernel 本身，
  同样约 8000 个 token，FlashInfer 一步 69.0 ms，FA3 一步 40.6 ms。
- P4（反量化到临时缓冲区）一步 8.64 ms，其中 `_dequant_gather_kernel` 6.8 ms，其余是把新 token 拷进临时缓冲区。
  这组负载关了前缀缓存，输入 8000 也小于分块大小 8192，本来预期不会有前缀要反量化。
  但一个请求如果被分块调度拆到两个 prefill batch 里，第二块就要把第一块反量化一遍，所以 P4 不是 0。
- **TriAxialKV 的 CPU 准备工作在 prefill 时很重：`init_forward_metadata` 每步 116.7 ms CPU**
  （bf16 的 FA3 后端在 1 ms 以下）。这是 `TriaxialFlashInferBackend` 用 numpy 逐 token 构造临时缓冲区的
  行号映射，再加 FlashInfer 的 plan 调用。因为 overlap 调度，这段 CPU 时间目前大多和上一步的 GPU 计算重叠，
  GPU 空闲只有 6 ms；但它给 prefill 设了一个下限：GPU 再快，一步也快不过这 117 ms 的 CPU 工作。
- 三种模式的 `process_batch_result` 在 prefill 期间都有每步约 100 ms 的 CPU 时间，
  区间里没有任何 torch 操作或 CUDA 调用，随运行进行从 0 增长到 430 ms。这个开销和 KV 格式无关，
  来源没有查清，见第 12 章第 3 条。它对关键路径的影响上限是 GPU 空闲 I（6~11 ms）。

### 9.6 汇总表（论文 motivation 用）

W1 解码中段，Qwen3-8B，单张 H100：

| 系统 | 准备类（P1g+P2+P3+P4+I） | attention 计算（K1） | 模型其余（M） | 一步合计 | CPU 端 P1（另计） |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGLang bf16 | 0.87 ms（4.0%） | 13.95 ms（64.6%） | 6.76 ms（31.3%） | 21.58 ms | 2.36 ms |
| SGLang fp8 | 2.78 ms（14.1%） | 10.26 ms（52.0%） | 6.69 ms（33.9%） | 19.73 ms | 3.63 ms |
| TriAxialKV | 1.65 ms（2.6%） | 55.05 ms（86.7%） | 6.80 ms（10.7%） | 63.49 ms | 3.22 ms |

fp8 的准备类比 bf16 高，主要是 GPU 空闲 I（2.51 对 0.78 ms），它来自 CPU 端的波动（9.3 节），不是 fp8 本身的开销；
fp8 真正多出来的写入开销 P3 只有 0.18 ms。

---

## 10. 容量受限场景：Qwen3-32B 单卡（缺口 2）

### 10.1 结论

第 3 节指出 Qwen3-8B 在单张 H100 上吞吐不受显存容量限制，所以压缩帮不上忙。
这一章换成 Qwen3-32B：bf16 权重约 61 GB，留给 KV 的显存只够约 7 个请求（每个 5120 个 token），
这是论文声称压缩收益最大的场景。

**结论：在这个场景里，TriAxialKV 赢 bf16，但输 fp8。**

| 配置 | KV 容量（token） | 容量倍数 | 吞吐上限（输出 tok/s） | 对 bf16 | 对 fp8 | 同时在跑的请求数 | 此时的 TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 39,326 | 1.00 | 208.7 | 1.00 | 0.58 | 6.7 | 29.9 ms |
| fp8 | 78,652 | 2.00 | 362.2 | 1.74 | 1.00 | 12.8 | 31.5 ms |
| TriAxialKV，`INT2_FRAC=0.05` | 128,412 | 3.27 | 282.1 | **1.35** | **0.78** | 21.2 | 70.4 ms |
| TriAxialKV，`INT2_FRAC=0.85` | 190,674（标称） | 4.85（标称） | 并发 8 时崩溃 | — | — | — | — |

吞吐上限取并发扫描（输入 4096、输出 1024）里输出吞吐最高的一点：bf16 在并发 64，fp8 和 TriAxialKV 在并发 32。
三者到这个点之后吞吐都不再上升（见 10.2 节）。

这张表是论文 motivation 最直接的一句话：**在压缩最该有用的场景里，TriAxialKV 用 3.27 倍的容量
只换来比 bf16 高 35% 的吞吐，而只有 2 倍容量的 fp8 高 74%。** 原因是它同时在跑的请求最多（21.2 个），
但每一步慢一倍以上（TPOT 70 ms，对 fp8 的 31 ms），多出来的并发被 attention kernel 吃掉了。

### 10.2 并发扫描（输入 4096，输出 1024，请求数 = 2 × 并发）

输出吞吐（tok/s）：

| 并发 | bf16 | fp8 | TriAxialKV 0.05 | TriAxialKV 0.85 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 69.5 | 72.7 | 62.1 | 61.4 |
| 4 | 131.2 | 138.1 | 113.6 | 110.6 |
| 8 | 167.6 | 247.5 | 176.9 | **崩溃** |
| 16 | 199.8 | 309.7 | 252.3 | — |
| 32 | 196.2 | 362.2 | 282.1 | — |
| 64 | 208.7 | 357.8 | 281.3 | — |

平均同时在跑的请求数（从服务日志的解码批次取平均）：

| 并发 | bf16 | fp8 | TriAxialKV 0.05 |
| ---: | ---: | ---: | ---: |
| 2 | 2.0 | 2.0 | 2.0 |
| 4 | 3.9 | 3.9 | 3.9 |
| 8 | 5.2 | 7.9 | 7.9 |
| 16 | 6.4 | 10.5 | 15.7 |
| 32 | 6.4 | 12.8 | 21.1 |
| 64 | 6.7 | 12.8 | 21.2 |

TPOT（ms）：

| 并发 | bf16 | fp8 | TriAxialKV 0.05 |
| ---: | ---: | ---: | ---: |
| 2 | 21.1 | 20.0 | 31.3 |
| 4 | 25.2 | 23.9 | 33.4 |
| 8 | 28.5 | 27.8 | 42.2 |
| 16 | 29.0 | 31.2 | 58.3 |
| 32 | 30.1 | 31.5 | 70.4 |
| 64 | 29.9 | 33.8 | 72.8 |

读法：

- **并发 2 和 4 时显存够用，TriAxialKV 是三者中最慢的**（62 对 bf16 的 69 tok/s）：
  这时比的只是每一步的速度，它的 TPOT 比 bf16 高约 50%。
- **从并发 8 开始 bf16 被容量卡住**（同时在跑的请求停在 5~7 个），TriAxialKV 开始反超 bf16。
- **fp8 在每一个并发点都比 TriAxialKV 快**。fp8 容量是 bf16 的 2 倍，每一步又和 bf16 一样快，
  两个好处同时拿到；TriAxialKV 容量更大，但每一步慢，两者抵消后仍然输给 fp8。
- 三者在并发 32~64 都停止增长：bf16 和 fp8 是因为容量满了（请求在排队，首 token 延迟在并发 64 时
  分别是 206 秒和 102 秒）；TriAxialKV 在并发 64 时同时在跑 21.2 个，容量（约 25 个）还没满，
  吞吐已经不涨了，因为每一步随 batch 变大而变慢（TPOT 从 31 涨到 73 ms）。

### 10.3 负载 W2（输入 8000，输出 256，并发 32，64 个请求）

| 配置 | 输出 tok/s | 总 tok/s | 对 bf16 | 对 fp8 | TPOT（ms） | TTFT（ms） | 同时在跑 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 91.4 | 2948 | 1.00 | 0.69 | 40.5 | 57,021 | 4.0 |
| fp8 | 132.8 | 4283 | 1.45 | 1.00 | 58.7 | 32,758 | 7.8 |
| TriAxialKV 0.05 | 103.3 | 3333 | **1.13** | **0.78** | 126.9 | 33,479 | 12.4 |
| TriAxialKV 0.85 | 崩溃 | — | — | — | — | — | — |

和并发扫描的结论一样：TriAxialKV 赢 bf16、输 fp8。prefill 很重的负载里，它的 TPOT 是 fp8 的 2.2 倍。

### 10.4 `INT2_FRAC=0.85` 为什么崩溃

和第 4.3 节是同一个原因。合成负载（随机 token）没有聊天模板的结构，tagger 把所有 token 判成 INT4
（INT2 占比 a = 0），而 0.85 的配置只留了 28,626 个 INT4 槽位，约 5.6 个 5120-token 的请求。
并发 8 时 INT4 区用完，服务直接报 `Out of memory (TriAxialKV)` 退出：

```
[2026-09-17 14:49:40] Out of memory (TriAxialKV). Try to lower your batch size.
RuntimeError: Out of memory (TriAxialKV). Try to lower your batch size.
```

W2 负载（并发 32）同样崩溃。并发 2 和 4 能跑完，吞吐和 0.05 的配置几乎一样（61.4 对 62.1、110.6 对 113.6），
因为这时两种配置都只用了 INT4 区里的一小部分。

### 10.5 一步推理的时间拆分（Qwen3-32B，容量受限解码）

方法同第 9 章。负载是输入 4096、输出 1024、并发 32，等解码开始后 60 秒再抓 120 个 forward。
抓取时三种配置都已经把 KV 用满：

| 配置 | 抓取时 batch | batch 里的 KV（token） | 一步（ms） | K1 attention | M 模型其余 | P3 写入 | I 空闲 | 准备类合计 | K1 占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 7~8 | 约 36,000 | 27.4 | 4.06 | 22.72 | 0.08 | 0.49 | 0.60（2.2%） | 14.8% |
| fp8 | 15 | 约 70,000 | 30.4 | 5.62 | 23.79 | 0.40 | 0.55 | 0.99（3.2%） | 18.5% |
| TriAxialKV 0.05 | 24 | 约 118,000 | 71.0 | **45.26** | 24.42 | 0.56 | 0.75 | 1.34（1.9%） | **63.7%** |

（bf16 的抓取里有 5 个解码时间段混进了 prefill 的 kernel，按第 9.2 节的规则丢弃了，分析的是其余 106 步。）

读法：

- **32B 模型在小 batch 下，一步的大头是线性层（M，约 23 ms）**，因为每一步都要把 61 GB 权重读一遍。
  bf16 和 fp8 的 attention 只占 15%~19%。
- **TriAxialKV 的 attention 占 64%。** 按每个 KV token 算，fp8 的 attention 是 5.62 ms / 7.0 万 token，
  TriAxialKV 是 45.26 ms / 11.8 万 token，**每个 token 贵 4.8 倍**，和第 2 节微基准的 5.1 倍一致。
- 准备类（写入、显存管理、GPU 空闲）三者都在 1.4 ms 以内，不是瓶颈。

**一个估算：如果 TriAxialKV 的 kernel 和 fp8 一样快会怎样。** 保持它的 batch（24）和其他各项不变，
只把 attention 换成 fp8 的每 token 成本，一步是 24.42 + 0.56 + 0.75 + 118,000 × (5.62 / 70,000) ≈ 35 ms，
解码吞吐约 24 / 35 ms ≈ 680 tok/s，而 fp8 在同一抓取里是 15 / 30.4 ms ≈ 490 tok/s，**高约 39%**。
这个数不是测出来的，只是在「其他都不变」的假设下的上限估计，它说明：在容量受限的场景里，
低于 8 bit 的混合精度 KV **有潜力赢 fp8，前提是 attention kernel 做到 fp8 的每 token 速度**。

---

## 11. SGLang 与最新 vLLM 的基线对照（缺口 3）

### 11.1 结论

模型 Qwen3-8B，负载 W1（输入 1024、输出 2048、并发 128、128 个请求）和 W2（输入 8000、输出 256、并发 32、64 个请求），
两边都用固定长度（不随机抽长度）、忽略结束符、关前缀缓存、分块 8192、显存比例 0.88。
DiffKV 那边还没有产出 vLLM 的数字（它的 `benchmarks/mixed_precision_profile/` 目录不存在），所以自己测了。

| 引擎 | KV | 负载 | 输出 tok/s（各次） | TPOT ms（各次） | TTFT ms（各次） |
| --- | --- | --- | --- | --- | --- |
| SGLang | bf16 | W1 | 3370 / 4201 / 3990 / 3934 | 21.6 / 20.5 / 20.7 / 21.0 | 15918 / 3879 / 6706 / 7551 |
| vLLM | bf16 | W1 | 5577 / 5732 / 5720 | 20.4 / 20.9 / 20.5 | 3880 / 2681 / 3102 |
| SGLang | bf16 | W2 | 626 / 607 / 612 / 631 | 33.4 / 33.7 / 33.4 / 33.6 | 4355 / 4223 / 4253 / 4170 |
| vLLM | bf16 | W2 | 688 / 696 / 694 | 32.7 / 33.2 / 34.5 | 3450 / 3225 / 2888 |
| SGLang | fp8 | W1 | 5683 / 6328 / 5594 / 6144 | 18.5 / 17.9 / 19.5 / 18.3 | 7861 / 4429 / 6590 / 5023 |
| vLLM | fp8 | W1 | 7048 / 6940 | 15.2 / 14.9 | 5127 / 5335 |
| SGLang | fp8 | W2 | 502 / 695 / 686 / 654 | 25.7 / 30.1 / 30.3 / 30.2 | 9588 / 3989 / 4106 / 3976 |
| vLLM | fp8 | W2 | 778 / 774 | 27.4 / 28.5 | 3436 / 3202 |

（SGLang 每格 4 次：1 次单独运行 + 同一个服务上连续 3 次；vLLM bf16 每格 3 次，fp8 每格 2 次。）

按 `PROFILE_FOLLOWUP.md` 的标准（差距在 10% 以内可以互换引用）：

| 指标 | bf16 | fp8 |
| --- | --- | --- |
| TPOT（解码每步速度） | **可以互换**：SGLang 20.5~21.6、vLLM 20.4~20.9 ms（W1），W2 两边都是 33 ms 左右 | **不能**：vLLM 快约 20%（W1：14.9~15.2 对 17.9~19.5 ms） |
| 输出吞吐 | **不能**：vLLM 高 33%~70%（W1）、9%~15%（W2） | **不能**：vLLM 高 10%~26%（W1）、11%~55%（W2，SGLang 有一次偏低） |

**建议：合并表里跨引擎只比较解码每步时间和第 9 章那样的时间拆分；吞吐倍数各自对自己引擎的 bf16 报。**
例如 TriAxialKV 报相对 SGLang bf16 的倍数，DiffKV 报相对 vLLM bf16 的倍数，不要拿 TriAxialKV 的吞吐直接除以 vLLM 的吞吐。

### 11.2 吞吐差距来自哪里

W1 下两边的 TPOT 几乎一样，吞吐却差 30% 以上，说明差距不在解码 kernel。为了分清是 SGLang 服务端的问题
还是 SGLang 压测客户端的问题，又用 vLLM 的压测客户端（`vllm bench serve`）去压 SGLang 的服务：

| 服务端 | 客户端 | W1 输出 tok/s | TTFT 平均 / 中位 / 标准差（ms） | 整轮耗时（秒） |
| --- | --- | ---: | --- | ---: |
| SGLang | SGLang 客户端 | 3370~4201 | 3879~15918（平均，4 次） | — |
| SGLang | vLLM 客户端 | 4079 / 4238 | 2610 / 2068 / 4106；2340 / 1843 / 3985 | 64.3 / 61.9 |
| vLLM | vLLM 客户端 | 5720 / 5732 | 2681 / 2680 / 719；3102 / 3530 / 1058 | 45.7 / 45.8 |

- 换成 vLLM 的客户端后，SGLang 的平均首 token 延迟降下来了（约 2.5 秒，原来 3.9~15.9 秒），但输出吞吐只从 3370~4201 变成 4079~4238。
- 真正拉开差距的是**少数请求的首 token 极晚**：SGLang 的首 token 延迟标准差是 4.0~4.1 秒，vLLM 只有 0.7~1.1 秒。
  这几个请求把整轮耗时从 46 秒拖到 62~64 秒，按「总生成 token 数 ÷ 整轮耗时」算的吞吐就低了约 30%。
- 这和第 12 章第 2 条记录的 SGLang prefill 准入波动是同一个现象。它属于 SGLang 的调度行为，
  和 KV 格式、attention kernel 无关，但会影响所有基于 SGLang 的系统（包括 TriAxialKV）在突发负载下的吞吐。

---

## 12. 补充实验中没做成的和不确定的

1. **CPU 端的数字受机器上其他任务影响，可重复性差。** 这个容器的 CPU 配额是 8 个核
   （`/sys/fs/cgroup/cpu.max` = 800000/100000），和同一台机器上的其他 agent（例如同时在做的 DiffKV profile、
   多个 Claude 会话、pip 安装和编译）共用。同一个 bf16 配置抓两次，GPU 上的数字几乎一样，
   但 GPU 空闲从 1.60 变成 4.09 ms、CPU 准备工作从 2.70 变成 5.94 ms（9.3 节）。
   所以 9 章里 P1 和 I 的数字只能看量级，不要比较两种模式之间零点几毫秒的差别。
   每次运行前后都记录了 cgroup 的节流计数（每个结果目录下的 `gpu_audit.txt`）。
2. **SGLang 的 prefill 准入批次在不同运行之间波动很大。** 同一个 W1 配置（128 个请求同时到达）连跑 6 次，
   一次 prefill 平均吃进的请求数差很多：有时 8~10 个一批（共约 21 个 prefill batch），
   有时 1 个一批（共约 130 个 prefill batch），首 token 延迟因此在 1.8 秒到 27 秒之间变化。
   这个现象最初被怀疑是计时开关造成的，用 6 次交替开关的运行排除了：
   关开关的 3 次分别是 129、105、52 个 prefill batch，开开关的 3 次分别是 112、80、21 个，没有系统性差别
   （`results/followup_checks/arrival_ab/summary.tsv`）。根本原因没有查清。
   它不影响稳态解码的拆分，但会让 W1 这种「请求同时涌入」负载的输出吞吐和首 token 延迟在单次运行之间差很多。
   **第 6.1 节的 W1 数字也受这个波动影响**（那次运行恰好是批量准入的一次）。
3. **prefill 期间 `process_batch_result` 每步约 100 ms 的 CPU 时间来源不明。** 三种模式都有，
   区间内没有任何 torch 操作或 CUDA 调用，随运行进行从 0 增长到 430 ms。
   想用 `py-spy` 看 Python 调用栈，但容器里没有 ptrace 权限（Permission denied）；
   用 `torch.profiler` 带调用栈抓一次可以查，但没有做。
4. **没有用 `ncu` 或 `nsys`。** GPU 性能计数器读不到（`ERR_NVGPUCTRPERM`），`nsys` 没有安装。
   时间拆分全部来自 `torch.profiler`（CUPTI activity 接口）。
5. **归类规则里有少量按每层次数推算的拆分**（fp8 的 `float8_copy_kernel`、TriAxialKV 的写入小 kernel，见 9.2 节第 3 条）。
   涉及的时间每步不到 0.3 ms，对结论没有影响，但不是逐个 kernel 从调用点确认的。
6. **中途被打断两次。** 会话中断会重启容器、杀掉所有进程（包括用 `setsid` 脱离的后台任务）。
   为此把所有实验放进了可续跑的队列 `run_followup_queue.sh`：做完的跳过，做到一半的结果目录挪到
   `results/_aborted/`（没有覆盖或删除任何数据）。`_aborted/` 下的目录都是不完整或作废的运行，报告里没有用。
   其中两个 W2 抓取作废的原因是抓取时机错了（抓到的是压测客户端预热请求的解码，不是 prefill），修正后重抓。
7. **vLLM 的版本和来源。** 用的是 `/data/gmh/workspace/vllm`（版本号 `0.26.1rc1.dev66+g0e776005a`，
   当前提交 `9b96da421 Tolerant prefix caching: quantized blocks are hit again`）。这是一个带 QuanCache 改动的分支，
   不是原版 vLLM；它的改动集中在前缀缓存，本次关掉了前缀缓存，所以对基线的影响应该很小，但没有和原版 vLLM 核对。
   注意力后端是 FlashAttention 3，CUDA graph 是 FULL_AND_PIECEWISE 模式。按要求只读使用，没有改动那个仓库，
   结果写在本仓库的 `results/followup_vllm/`。
   vLLM 启动时需要 CUDA 编译工具链，而 `/tmp/quancache_local.env` 没有设置 `CUDA_HOME`，
   第一次启动因 `Could not find nvcc` 失败；在压测脚本里补上 `CUDA_HOME=/tmp/envs/quancache` 后正常。
8. **Qwen3-32B 的模型文件。** `/data/gmh/model/Qwen3-32B` 是一个没下载完的目录（17 个分片只有 4 个，
   还留着 aria2 的断点文件），加载不了。实验用的是用户重新下载到 `/tmp/models/Qwen3-32B` 的完整副本
   （17 个分片、707 个张量，逐个核对过 safetensors 头和文件大小）。
9. **32B 的实验只跑了一次。** 每个并发点只有一次运行，没有重复。考虑到第 2 条的准入波动，
   单点吞吐可能有 ±10% 左右的误差；但 TriAxialKV 在并发 16~64 上连续 3 个点都落在 bf16 和 fp8 之间，结论不依赖单个点。
10. **32B 用了 `--cuda-graph-max-bs 64` 和 `--mem-fraction-static 0.90`**（三种配置相同），
   目的是给 61 GB 权重之外留出激活值的空间。这和 8B 的设置（0.88、默认 CUDA graph 上限 256）不同，
   两个模型之间不要直接比较绝对数字。
11. **10.5 节「kernel 和 fp8 一样快」的估算不是测量值**，只是把 attention 的每 token 成本换掉、其他项不变的上限估计。
   真实情况下 batch 更大时线性层（M）也会略微变慢。

---

## 13. 补充实验的复现

脚本都在 `benchmark/mixed_precision_profile/`。所有实验放在一个可续跑的队列里：

```bash
cd benchmark/mixed_precision_profile
./stop_followup.sh          # 停掉可能在跑的旧队列和服务（只按精确的进程名匹配）
./run_followup_queue.sh     # 按顺序跑完所有实验；中断后重跑同一条命令即可续跑
```

队列做完后会自动运行 `summarize_followup.py`，把本报告第 9~11 章的表格写到 `results/followup_tables.md`。

| 文件 | 作用 |
| --- | --- |
| `run_followup_queue.sh` | 实验队列。做完的跳过，做到一半的挪到 `results/_aborted/`；32B 的任务会等模型文件下载完整再开始 |
| `profile_step.sh` | 开计时开关起服务、压测、在指定时机调用 `/start_profile` 抓 trace（W1 解码、W2 prefill、S4096 容量受限解码） |
| `analyze_trace.py` | 把一个 trace 拆成 P1~I 各类别（归类规则见第 9.2 节） |
| `bench_point.sh` | 关计时开关，一次起服务、跑一个或多个压测点（吞吐数字都来自它） |
| `vllm_multi.sh` | 同一个服务上连跑 W1、W2 多次；可以选服务端（SGLang / vLLM）和压测客户端（SGLang / vLLM） |
| `vllm_point.sh` | vLLM 单点压测（第 11 章的第一次运行） |
| `arrival_ab.sh` | 交替开关计时开关，检查它是否影响请求准入（第 12 章第 2 条） |
| `gpu_guard.sh` | 每次开跑前确认 GPU 没被别人占用，并记录显存和 CPU 节流计数 |
| `serve.sh` | 起 SGLang 服务，参数可用环境变量覆盖 |
| `stop_followup.sh` | 安全地停掉队列和它起的所有服务 |
| `summarize_followup.py` | 生成 `results/followup_tables.md` |

计时开关的代码在 `python/sglang/srt/utils/triaxial_profile.py`，改动的 8 个 SGLang 文件在 9.2 节列出。
开关默认关闭；要抓 trace 时设 `SGLANG_TRIAXIAL_PROFILE=1`，然后调 SGLang 的 `/start_profile`。

原始数据：

| 目录 | 内容 |
| --- | --- |
| `results/followup_breakdown/` | 每次抓取的 trace、服务日志、压测日志、GPU 与 CPU 审计记录 |
| `results/followup_points/` | 第 10 章 32B 的吞吐点，以及第 11 章 SGLang 的单次运行 |
| `results/followup_vllm/` | 第 11 章的 vLLM 运行、SGLang 重复运行和交叉客户端运行 |
| `results/followup_checks/arrival_ab/` | 计时开关对准入影响的 6 次对照 |
| `results/followup_logs/` | 每个队列任务的日志和 `queue.log` |
| `results/_aborted/` | 作废或中断的运行，报告没有使用 |
