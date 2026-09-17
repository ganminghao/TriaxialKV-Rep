# TriAxialKV on SGLang —— 使用说明书

这个仓库是 SGLang v0.5.10 的一个分支，在里面实现了论文 TriAxialKV（arXiv 2605.17170）的
KV cache 混合精度方案：把每个 token 的 KV cache 按"它是什么内容"分别存成 INT2 或 INT4，
而不是统一的 bf16。目的是在同样显存下装下更多请求，从而提高吞吐。

几个第一次出现的词，先解释：

- **KV cache**：推理时缓存下来的 key 和 value 张量，长上下文时它占显存的大头。
- **INT2 / INT4**：用 2 个比特或 4 个比特存一个数值，配一个 fp16 的缩放系数和零点，
  读取时再还原成 bf16。bf16 是 16 个比特。
- **打标（tagging）**：请求进来时扫一遍 token，给每个 token 贴一个标签，据此决定用几个比特。
  标签只看聊天模板里的特殊符号（角色标记、图片占位符、工具调用括号），不需要跑模型。
- **分片池（pool）**：显存里存 KV cache 的那块区域。本实现里有两块，一块放 INT2，一块放 INT4。

底层的数据布局、位打包方式、kernel 接口写在 `TRIAXIAL_DESIGN.md`（英文），改 kernel 前必读。

---

## 1. 仓库里改了什么

基线是 upstream 的 `v0.5.10` tag（`git describe --tags` 可以确认）。
注意 pip 显示的版本号是 `0.5.19`，那是 setuptools-scm 从 git 标签推导出来的结果，不代表基线版本。

**核心实现与测试（9 个文件，约 3100 行）**

| 文件 | 作用 |
| --- | --- |
| `TRIAXIAL_DESIGN.md` | 数据布局与 kernel 接口规格（英文） |
| `python/sglang/srt/managers/triaxial_tagger.py` | 打标器和标签到位宽的策略表，纯 numpy，不依赖引擎 |
| `python/sglang/srt/mem_cache/triaxial_pool.py` | INT2/INT4 双池和它的分配器 |
| `python/sglang/srt/layers/attention/triton_ops/triaxial_kv.py` | 量化、反量化、融合解码的 Triton kernel，只依赖 torch 和 triton |
| `python/sglang/srt/layers/attention/triaxial_backend.py` | 两个注意力后端：prefill 走 FlashInfer，decode 走上面的 kernel |
| `python/sglang/test/test_triaxial_kv.py` | kernel 单元测试，37 个用例 |
| `python/sglang/test/bench_triaxial_decode.py` | 解码 kernel 的微基准，和 bf16 kernel 对比 |
| `python/sglang/test/triaxial_e2e_check.py` | 端到端输出检查：发固定提示词、存结果、比较两次运行 |
| `python/sglang/test/triaxial_tag_stats.py` | 离线统计一份请求集合的标签分布和 INT2 占比 |

**改动的 upstream 文件（9 个，共 257 行）**

| 文件 | 改了什么 |
| --- | --- |
| `srt/server_args.py` | 加三个命令行参数，以及开启后的参数校验与后端切换 |
| `srt/model_executor/model_runner_kv_cache_mixin.py` | 每 token 显存开销的计算、双池和分配器的创建 |
| `srt/mem_cache/common.py` | prefill 时按位宽路由分配，以及对应的驱逐判断 |
| `srt/managers/scheduler.py` | 建请求时调用打标器，把逐 token 位宽挂到请求上 |
| `srt/managers/schedule_batch.py` | 请求上的位宽字段；解码前的显存检查改看 INT4 区 |
| `srt/managers/schedule_policy.py` | 准入预算改看 INT4 区（见第 6 节的坑） |
| `srt/layers/attention/attention_registry.py` | 注册两个新后端名字 |
| `srt/layers/attention/triton_backend.py` | 取 value 维度时兼容没有稠密缓冲的池 |
| `benchmark/datasets/openai_dataset.py` | 基准客户端统计图片 token 数；兼容 transformers 5 的返回类型 |

**基准工具（`benchmark/triaxialkv_osworld/`）**

| 文件 | 作用 |
| --- | --- |
| `build_replay.py` | 从公开的 OSWorld 轨迹压缩包重建出智能体当时发给模型的聊天请求 |
| `tasks_instructions.json` | 369 个 OSWorld 任务的任务描述文本 |
| `run_bench.sh` | 起服务 + 跑压测，一条命令一组数据 |
| `REPLAY_NOTES.md` | 重建规则的详细说明（英文），包括日志与步骤如何对齐 |

`scripts/` 下另有三个脚本：`triaxial_setup_env.sh` 建环境，
`triaxial_prefetch_wheels.py` 加速下载依赖包，`triaxial_fetch_data.sh` 下数据并重建请求。
根目录还有本文件 `TRIAXIAL_README.md`，以及 upstream `README.md` 顶部加的一段指路说明。

---

## 2. 推到云端

当前所有改动都还没提交，远端 `origin` 指向本机的一个目录，不是云端。步骤如下，**这些命令需要你自己执行**：

```bash
cd /data/gmh/workspace/sglang-triaxialkv
git status --short                      # 应是 10 个改动 + 18 个新增，共 28 个文件
git add -A
git commit -m "TriAxialKV: mixed INT2/INT4 KV cache on SGLang v0.5.10"

git remote add cloud git@github.com:<你的账号>/<仓库名>.git
git push -u cloud triaxialkv            # 只推这一个分支
```

两点提醒：

- 这个仓库是从本机的 SGLang 完整克隆来的，`.git` 有 353 MB，第一次推送会比较久。
  只推 `triaxialkv` 一个分支可以避免把 upstream 的其他分支也推上去，但历史对象还是要传。
- 如果只想传很小的东西，另一条路是在新机器上从 GitHub 克隆官方 SGLang 的 `v0.5.10` tag，
  再把补丁打上去。补丁这样生成：`git diff v0.5.10 > triaxial.patch`，
  但新增文件需要先 `git add -N .` 才会进补丁。

---

## 3. 在另一台机器上装起来

前提：一张 NVIDIA GPU（本项目在 H100 80GB 上开发），CUDA 驱动，conda，可访问网络。

### 3.1 代码和环境

```bash
git clone -b triaxialkv <你的云端地址> sglang-triaxialkv
cd sglang-triaxialkv
./scripts/triaxial_setup_env.sh --prefetch --with-cuda-toolkit
```

脚本做的事：建一个 python 3.11 的 conda 环境（默认在 `/tmp/envs/sglang0510`），
装 `numactl`（提供 `libnuma.so.1`，缺它 sgl_kernel 加载不了），
装 CUDA 工具链（可选，见下），下载并安装依赖，最后以可编辑方式安装本仓库，
并写出一个 `$ENV_PREFIX/triaxial_env.sh` 供后续脚本引用。

几个选项：

- `--prefetch`：先用 aria2c 多线程把大的依赖包下到本地再装。
  在本机的代理下，pip 直接下载大包只有 0.2 MB/s，aria2c 有 4 到 8 MB/s，差 20 倍以上。
  如果你的网络直连 PyPI 很快，可以不加这个选项。
- `--with-cuda-toolkit`：把 CUDA 工具链装进环境里，约 2 GB。
  必须让 `CUDA_HOME` 指向一个含有 `nvcc` 的 CUDA 安装，否则 SGLang 导入 deep_gemm 时会直接断言失败。
  如果机器上已有 `/usr/local/cuda`，脚本会自动找到，就不用加这个选项。
- 环境变量 `ENV_PREFIX` 改环境路径，`WHEEL_DIR` 改依赖包缓存目录。

版本是被 upstream 锁死的，不要随意升级：
torch 2.9.1（cu128）、triton 3.5.1、transformers 5.3.0、flashinfer 0.6.7.post2、sglang-kernel 0.4.1。
这套和别的项目环境互相冲突，所以必须单独建一个环境。

### 3.2 模型

论文用的是 Qwen3-VL-32B-Thinking，本项目用的是同样大小的 Instruct 版本。
固定输出长度做压测时两者吞吐没有差别，只有需要历史里出现真实思考段落时才需要 Thinking 版。

```bash
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --local-dir /tmp/models/Qwen3-VL-32B-Instruct
```

约 62 GB。国内可以设 `HF_ENDPOINT=https://hf-mirror.com`。
另外建议也下一个 `Qwen/Qwen3-8B`（约 16 GB），纯文本模型，调试改动时起服务快得多。

### 3.3 数据

```bash
source /tmp/envs/sglang0510/triaxial_env.sh
./scripts/triaxial_fetch_data.sh
```

脚本下载 OSWorld 官方公开的一份评测轨迹（2.7 GB，369 个任务，2494 个记录步骤），
校验完整性，然后按 OSWorld 官方智能体的提示词格式逐步重建请求，
输出 `/tmp/datasets/osworld_trajs/replay_15step_h4.jsonl`，2473 个请求，约 6 GB。

重建出的请求：平均 8829 个 token，带满 4 轮历史的请求平均 12056 个 token。
论文说它们的平均 prefill 长度是 11000 个 token，量级一致。

### 3.4 验证装好了

```bash
source /tmp/envs/sglang0510/triaxial_env.sh
cd sglang-triaxialkv/python
$PY -m pytest -q sglang/test/test_triaxial_kv.py       # 期望 37 passed
```

这 37 个用例覆盖：三个量化 kernel 与纯 torch 参考实现逐字节相等、
量化再还原的误差不超过半个量化步长、反量化聚合在混合 slot 列表上逐位相等、
融合解码 kernel 与"先还原再做标准注意力"的结果在 bf16 容差内一致、以及 CUDA graph 捕获后重放一致。

---

## 4. 怎么跑

### 4.1 起一个服务

```bash
source /tmp/envs/sglang0510/triaxial_env.sh
$PY -m sglang.launch_server \
  --model-path /tmp/models/Qwen3-VL-32B-Instruct \
  --port 30000 --host 127.0.0.1 \
  --triaxial-kv --triaxial-int2-fraction 0.85 \
  --context-length 32768 --mem-fraction-static 0.90 \
  --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
  --page-size 1 --trust-remote-code
```

启动日志里应该看到两行，确认生效了：

```
TriAxialKV tagger ready. TriaxialPolicy: 28/56 tag combinations at INT2
TriAxialKV pool: 140608 INT2 slots (4394 pages of 32) + 24815 INT4 slots, offset=140640, int2_fraction=0.850
```

### 4.2 三个新参数

- `--triaxial-kv`：开关。开启后自动做这些事：把 prefill 后端设成 `triaxial_flashinfer`、
  decode 后端设成 `triaxial_triton`、强制 `--page-size 1`、关掉分块 CUDA 图和混合分块。
  不支持投机解码，也不能同时指定 `--kv-cache-dtype`（存储格式由本方案自己管）。
- `--triaxial-int2-fraction`：INT2 区占总 token 容量的比例，默认 0.85。
  这个值要和策略实际产生的 INT2 占比匹配，配低了 INT2 区不够用会退化成 INT4，配高了 INT4 区会先耗尽。
  用第 4.4 节的统计脚本先量一下。
- `--triaxial-policy`：标签到位宽的策略。可选 `default`、`current_int4`、`all4`、`all2`，或一个 JSON 文件路径。
  JSON 格式：`{"default": 4, "rules": [{"temporal": [...], "modal": [...], "semantic": [...], "bits": 2}]}`，
  先匹配上的规则说了算，没写的轴表示全匹配。
  三个轴的取值：时间轴 `older/m2/m1/current`，模态轴 `text/image`，
  语义轴 `inst/user/assistant/reasoning/tool_call/obs/delim`。

`default` 策略的内容是：系统提示和工具描述、模板符号、工具调用用 INT4；当前轮的文字用 INT4；
其余一律 INT2，**包括当前轮的图片**。最后这条是有意为之，原因见第 6 节第一条。

### 4.3 端到端输出检查

先跑基线再跑本方案，然后比较两次的输出：

```bash
# 终端 A：起 bf16 基线服务（端口 30002）
# 终端 B：
$PY -m sglang.test.triaxial_e2e_check run --url http://127.0.0.1:30002 --out bf16.json --max-tokens 200 \
    --replay /tmp/datasets/osworld_trajs/replay_smoke_20.jsonl --num 6
# 换成 TriAxialKV 服务再跑一次，存成 tri.json，然后：
$PY -m sglang.test.triaxial_e2e_check diff bf16.json tri.json
```

不加 `--replay` 就用脚本内置的四条纯文字提示词，适合用 Qwen3-8B 快速验证。

**怎么判断算对了**：量化本来就会改变输出，所以不要求逐字相同。要看的是：
在 OSWorld 请求上，模型输出的工具调用格式正确，动作类型和基线一致，点击坐标只差几个像素。
9 月 11 日的检查结果是 6 个请求全部满足，其中一个逐字相同。

### 4.4 量一份请求集合的 INT2 占比

```bash
$PY -m sglang.test.triaxial_tag_stats \
    --model /tmp/models/Qwen3-VL-32B-Instruct \
    --replay /tmp/datasets/osworld_trajs/replay_15step_h4.jsonl \
    --decode-len 300 --limit 300
```

它会打印每个标签组合占多少 token、用几个比特，以及建议的 `--triaxial-int2-fraction`。
在 OSWorld 重放上，`default` 策略给出 INT2 占 87.5%，建议值 0.846，所以我们用 0.85。

### 4.5 吞吐基准

```bash
cd benchmark/triaxialkv_osworld
./run_bench.sh bf16 60 64      # 模式、请求数、最大并发
./run_bench.sh tri  60 64
```

四种模式：`bf16` 是论文的基线（prefill 用 FlashInfer，decode 用 SGLang 自带的 Triton kernel），
`bf16-fi` 两边都用 FlashInfer，`fp8` 用 SGLang 自带的 fp8 KV cache，`tri` 是本方案。

结果写到 `results/<模式>_<时间戳>/`，里面有 `server.log`、`bench.log`、`info.txt`（摘要）。
可以用环境变量覆盖 `MODEL`、`REPLAY`、`PORT`、`MEM_FRAC`、`CHUNK`、`INT2_FRAC`、`POLICY`。

**服务跑着的时候不要编辑这个脚本**：bash 是按字节偏移惰性读脚本的，中途改会让正在执行的脚本错乱。

---

## 5. 目前的结果（2026-09-11，单张 H100 80GB，Qwen3-VL-32B-Instruct）

60 个请求，最大并发 64，`--mem-fraction-static 0.90`：

| 指标 | bf16 基线 | TriAxialKV | 比值 |
| --- | --- | --- | --- |
| KV cache 容量（token） | 34118 | 165423 | 4.85 倍 |
| 总吞吐（token/秒） | 1885 | 2662 | 1.41 倍 |
| 服务端同时在跑的请求数 | 3 到 4 | 14 | 约 4 倍 |
| 压测耗时（秒） | 236.8 | 167.7 | 0.71 倍 |
| 每个输出 token 的平均间隔（毫秒） | 46.4 | 135.2 | 2.9 倍 |

论文报告的是 1.52 倍吞吐、并发 11.78 对 3.46。我们的并发倍数和论文一致，吞吐倍数偏低 0.11。
差距来自解码 kernel：在 batch 12、序列长 11000 的微基准下，
本方案的融合解码 kernel 每层一次调用要 525 微秒，SGLang 的 bf16 kernel 是 203 微秒，
慢 2.6 倍，尽管读的字节少 4.6 倍。瓶颈是指令数不是显存带宽——把所有 KV 加载换成常数后仍要 470 微秒。

已知的优化方向是给 INT2 的 key 做按页对齐的快速路径，目前 INT2 的 key 仍按元素逐个取。

微基准复现：

```bash
$PY sglang/test/bench_triaxial_decode.py
```

---

## 6. 已知的坑

**图片必须标成 INT2，哪怕它在当前轮。**
前缀缓存里一个 token 只存一份，按第一次写入时的精度保存，之后复用时不会改。
一张截图第一次出现时属于"当前轮"，如果按时间轴给它 INT4，
下一步它变成历史后仍然是 INT4，整段历史都留在 INT4 区，INT4 区很快耗尽然后报错。
这是第一次冒烟崩溃的原因。想试原来的做法用 `--triaxial-policy current_int4`。

**准入预算必须看 INT4 区，不能看总量。**
INT2 区不够用时可以退到 INT4，反过来不行，所以真正的瓶颈是 INT4 区。
调度器决定还能收多少请求时，如果只看总剩余 token 数，就会收下 INT4 区装不下的请求然后崩。
做法是在分配器上加一个 `admission_size()`，按 `1 - int2_fraction` 折算 INT4 区还能吸收多少混合 token，
调度策略优先调用它。而 `available_size()` 必须保持精确，否则 SGLang 空闲时的显存泄漏检查会报错。

**`CUDA_HOME` 是必须的。**
不设置的话 SGLang 导入 deep_gemm 时直接断言失败，报错信息看不出是缺 CUDA_HOME。
`triaxial_env.sh` 会设好。

**缺 `libnuma.so.1` 会让 sgl_kernel 加载失败。**
报错是 "Could not load any common_ops library"，真正原因藏在最后一行。
装 conda 包 `numactl` 解决。

**显存参数组合有限制。**
`--mem-fraction-static 0.90` 配 `--chunked-prefill-size 16384` 在 prefill 阶段会显存不足，
因为一次前向的激活值太大。用 8192 就没问题。这两个值都写进 `run_bench.sh` 的默认值了。

**代理下 pip 下载大包极慢。**
0.2 MB/s，而同一条链路 curl 和 aria2c 有 4 到 8 MB/s。
原因是 pip 的 HTTP 缓存加上单条长连接。用 `--prefetch` 绕开。

---

## 7. 下一步

1. 给解码 kernel 做按页对齐的快速路径，把每层 525 微秒压下来。这是吞吐倍数偏低的直接原因。
2. 用 300 到 600 个请求跑 `bf16`、`fp8`、`tri` 三组正式对比，替掉现在 60 个请求的冒烟数据。
3. 和 QuanCache 做对比实验。QuanCache 在另一个仓库（`/data/gmh/workspace/vllm`），
   对比要分三层做：系统吞吐各自对自己的 bf16 基线报倍数；
   策略比较在同一个引擎里做（打标器是纯 numpy，可以搬过去）；
   格式比较在同样位宽预算下比 INT2/INT4 和 fp8、nvfp4。

---

## 8. 引用

- 论文：TriAxialKV: Toward Extreme Low-Precision KV-Cache Quantization for Agentic Inference Tasks，
  arXiv 2605.17170。代码未开源，本仓库是按论文描述的独立实现。
- 基线引擎：SGLang v0.5.10，Apache 2.0。
- 数据：OSWorld-Verified 轨迹数据集 `xlangai/ubuntu_osworld_verified_trajs`，MIT。
  智能体提示词格式取自 OSWorld 仓库的 `mm_agents/qwen3vl_agent.py`，Apache 2.0。
