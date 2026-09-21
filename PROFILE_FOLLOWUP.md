# 补充实验要求：TriAxialKV profile 的三个缺口

日期：2026-09-17。这份文件是交给执行 agent 的任务说明，是对本目录
`PROFILE_FINDINGS.md` 的补充。已有的数据（解码 kernel 的带宽利用率、并发扫描、
静态分池的容量矩阵、prefill 反量化成本、三个端到端负载点）质量足够，**不需要重做**。
缺的是下面三项。做完后把结果追加到 `PROFILE_FINDINGS.md`（新增章节，不要改已有章节的数字），
脚本和原始数据放进 `benchmark/mixed_precision_profile/`。

同样的测量正在对 DiffKV 做，任务说明在 `/data/gmh/workspace/DiffKV/PROFILE_PROMPT.md`。
两边的分类和负载点必须一致，最后要合并成一张表。

## 缺口 1（最重要）：没有「一步推理的时间拆分」

现有报告分别测了 kernel 单独调用的时间和端到端的 TPOT（每个输出 token 的平均间隔），
但没有把两者连起来。读者无法直接看到：TriAxialKV 的 TPOT 是 59.7 ms，bf16 是 20.4 ms，
多出来的 39 ms 里，自研 attention kernel 占多少，量化写入占多少，
tagger（给 token 分配位宽的那段逻辑）和分配器占多少，CPU 端准备占多少。
论文的 motivation 需要的正是这张表。

要求：对 `bf16`、`fp8`、`tri`（`INT2_FRAC=0.05`）三种模式，
在负载 W1（输入 1024 / 输出 2048，并发 128）的稳定解码阶段，
和负载 W2（输入 8000 / 输出 256，并发 32）的 prefill 阶段，
各抓一段（跳过预热，至少连续 50 步解码、5 次 prefill），把一步的墙钟时间拆成下面几类。
先试 `torch.profiler`（SGLang 自带 `/start_profile` 接口）；CUPTI 不可用时
退回用环境变量控制的 CUDA event 计时，并说明同步带来的失真。

| 类别 | TriAxialKV 里对应什么 |
| --- | --- |
| P1 CPU 调度与输入准备 | 调度器、tagger、`TriaxialFlashInferBackend` 的 metadata 构造、INT2/INT4 两个池的分配 |
| P2 GPU 显存管理 | 页的分配回收如果有 GPU kernel 就算这里，没有就写 0 |
| P3 KV 写入与量化 | `quant_int4_tokens`、`quant_int2_v_tokens`、`quant_int2_k_pages` |
| P4 反量化到临时 bf16 缓冲区 | prefill 时的 `dequant_gather`（解码阶段应为 0） |
| K1 attention 计算 kernel | 解码：混合精度 Triton kernel 的 stage1 + stage2；prefill：FlashInfer |
| M 模型其余部分 | 线性层、MLP、norm、RoPE、采样 |
| I GPU 空闲 | 一步之内 GPU 上没有 kernel 在跑的时间 |

报告两种视图：GPU 时间占比（P2+P3+P4+K1+M = 100%），
以及墙钟时间占比（再加上 I）。P1 单独报告 CPU 毫秒数。
最后给一张汇总表：每种模式在 W1 解码的一步里，「准备类」（P1+P2+P3+P4+I）、
「attention 计算」（K1）、「模型其余」（M）各多少毫秒、各占百分之多少。
另外记录三种模式是否都开了 CUDA graph；如果 `tri` 模式没开，
补一组 bf16 关掉 CUDA graph 的数据，把这一项的影响分出来。

## 缺口 2：只测了压缩没有用处的场景

现有端到端数据全部来自 Qwen3-8B，报告第 3 节自己也指出：8B 模型在单张 H100 上
bf16 已经能同时跑 44~56 个请求，吞吐不受显存容量限制，压缩本来就帮不上忙。
只用这组数据写 motivation，审稿人会说这是故意挑了对压缩不利的场景。

要求：用 `/data/gmh/model/Qwen3-32B` 在单卡上重跑三种模式。bf16 权重约 64 GB，
留给 KV 的显存很少，bf16 只能同时跑个位数的请求，这是论文声称收益最大的场景。

- 负载：输入 4096 / 输出 1024 的并发扫描（并发 2,4,8,16,32,64），
  以及输入 8000 / 输出 256、并发 32。
- 每个点报告：输出 tok/s、TPOT、TTFT（首 token 延迟）、平均同时在跑的请求数、KV 容量。
- `tri` 模式的 `INT2_FRAC` 取两个值：0.05（合成负载下 tagger 几乎不产生 INT2）和 README 推荐的 0.85。
  0.85 如果崩溃，记录崩溃时的并发和报错即可。
- 要回答的问题：在容量受限的场景下，TriAxialKV 是否赢 bf16？是否赢 fp8？
  如果赢 bf16 但输 fp8，这就是最有力的一句 motivation。
- 对这个场景同样做一次缺口 1 的时间拆分（只需要解码阶段）。

## 缺口 3：对照组里没有最新 vLLM

现有对照组是 SGLang 的 bf16 和 fp8。DiffKV 的对照组将是最新 vLLM 的 bf16 和 fp8。
为了让两份报告能合并，需要知道两个引擎的基线差多少。

要求：用 `/data/gmh/workspace/vllm`（环境 `/tmp/envs/quancache/bin/python`，
若存在 `/tmp/quancache_local.env` 先 source；**只读使用，不修改那个仓库的任何文件，
不往它的 `results/` 写东西**）在 Qwen3-8B 上跑 W1、W2 两个负载点的 bf16 和 fp8
（`--kv-cache-dtype fp8`，`--no-enable-prefix-caching`），
压测用 `vllm bench serve --dataset-name random`，输入输出长度、并发、请求数和 SGLang 那边完全相同。
报告一张四行的表（SGLang bf16 / SGLang fp8 / vLLM bf16 / vLLM fp8），
差距在 10% 以内就说明两个引擎的基线可以互换引用。
如果 DiffKV 那边的 agent 已经测了 vLLM 的这两个点，直接引用它的数字，不要重复跑。

## 约束

- 开跑前用 `nvidia-smi` 确认 GPU 没有别的进程在用。DiffKV 的 profile 可能同时在进行，
  两边不能同时占卡，否则双方的数字都作废。
- 查找或结束服务进程用 `ps -eo args --no-headers | grep -E "sglang|vllm[.]entrypoints"`，
  不要用 `pkill -f` / `pgrep -f`。
- 后台长任务用 `setsid ... & disown` 加轮询等待。脚本运行期间不要编辑正在执行的 shell 脚本。
- 不要覆盖 `results/` 下已有的数据，新数据用新的子目录。
- 不要做任何 git 操作。
- 计时代码必须用环境变量开关，关掉时不影响性能；吞吐数字在关掉计时的情况下测。
- 写作要求同 `PROFILE_FINDINGS.md`：简单直白的中文，术语第一次出现要解释，
  每个数字说明它回答什么问题、和谁比，结论在前，失败如实写。
