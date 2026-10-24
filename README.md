# KDN demo：按 Agent 等待信号调度 KV

全部用 C 写。模拟器只依赖 libc；真跑的引擎基于 llama.cpp 的 C API，在 Metal 上运行。

## 核心想法

每次模型调用结束时，系统已经知道接下来在等什么，不需要猜：

| 调用结束时的信号 | 下一次请求在 5 分钟以上之后才来的比例 | 13 万次调用里的占比 |
|---|---|---|
| 本地文件工具（Read/Edit/Write…） | 0.0% | 9.8% |
| MCP 工具 | 0.0% | 3.9% |
| Bash | 2.4% | 76.2% |
| end_turn（等人回复） | **20.2%** | 8.7% |
| AskUserQuestion | **27.7%** | 0.1% |

所有超过 5 分钟的空档里，**47.6% 只靠 end_turn 这一个信号就能提前认出来**。

据此有三个优化：

1. **信号调度**：给每个会话估计"下次什么时候会用"，HBM 满了先挪最晚才用的。
   - Bash、人工回复等信号用同一会话上一次同类空档来预测。
   - 已经超时还没来的会话，按重尾分布重新估计（已经等了 w 秒，大概还要再等 w 秒）。
   - 预测空档超过 30 分钟，直接降到 SSD。
2. **共享前缀去重**：系统提示加工具定义约 6 万 token，所有会话共用一份；子代理模板约 1.7 万 token。会话只占私有部分，降级和恢复也只搬私有部分。
3. **按 LRU 最差、Belady 最优做上下界**，看信号调度离理论上限有多远。

## 运行

```sh
make                                   # 首次会克隆并编译 llama.cpp（Metal）
make models/qwen3-1.7b-q4_k_m.gguf     # 1.1GB 模型
./bin/sim 20 50 100 200                # 生产规模回放，约 1 秒
./bin/sim --kv v41flash 200            # 换成 V4.1 Flash 的 KV 大小
./bin/live                             # MacBook 真跑：微基准 + 轨迹回放
./bin/live --skip-bench --users 8 --calls 300
./bin/prefill_bench                    # prefill 速度排查
```

`data/trace.csv` 是从 13 万次 Claude Code 调用（2026-08-07～10-19）导出的轨迹。每行是一次调用：会话号、共享前缀长度、开始/结束时间、上下文 token 数、输出 token 数、结束信号。

## 结果

### 生产规模回放（`bin/sim`，MLA 类 KV 70KB/token，HBM 450GB，DRAM 1TB，SSD 16TB）

| 用户数 | 策略 | HBM 命中 | 平均附加 TTFT | P99 附加 TTFT |
|---|---|---|---|---|
| 100 | LRU | 48.8% | 0.32s | 3.7s |
| 100 | LRU + 共享前缀 | 56.7% | 0.22s | 3.0s |
| 100 | **信号调度 + 共享前缀** | **75.4%** | **0.16s** | 3.1s |
| 100 | Belady 上限 | 85.2% | 0.09s | 1.6s |
| 200 | LRU | 25.1% | 0.68s | 4.7s |
| 200 | LRU + 共享前缀 | 30.6% | 0.48s | 4.3s |
| 200 | **信号调度 + 共享前缀** | **58.0%** | **0.34s** | 4.1s |
| 200 | Belady 上限 | 72.4% | 0.20s | 3.4s |

换成 V4.1 Flash 的 KV 大小（890B/token）时，200 个用户 HBM 也装得下：LRU 命中 100%。信号调度把 0.6% 预测为长空档的会话主动降到了 SSD，但附加 TTFT 仍然约等于 0。这种情况下分层没有收益。

### MacBook M3 Max 实测（`bin/live`，Qwen3-1.7B Q4_K_M）

**同一段上下文，三种取回方式的耗时：**

| 上下文 | 从头重算 | DRAM→GPU | SSD→GPU（绕过页缓存） | KV 大小 |
|---|---|---|---|---|
| 2k | 0.79s | 0.006s | 0.044s | 229MB |
| 8k | 4.06s | 0.037s | 0.21s | 918MB |
| 16k | 11.3s | 0.098s | 0.55s | 1.8GB |
| 32k | 34.7s | 0.18s | 0.92s | 3.7GB |

**真实轨迹回放**（8 个用户，最繁忙 2 小时里的 300 次调用，token 数除以 40）：

| 策略 | GPU 命中 | 平均 TTFT | P95 | 后台 I/O |
|---|---|---|---|---|
| LRU | 89.4% | 179ms | 362ms | 18.2s |
| 信号调度 + 共享前缀 | 92.8% | 156ms | 314ms | 3.3s |

16 个用户（24 个会话），同一配置跑 4 次。命中率每次相同；延迟取 4 次的中位数，括号里是范围：

| 策略 | GPU 命中 | 平均 TTFT | 平均恢复耗时 | 平均 prefill |
|---|---|---|---|---|
| LRU | 77.0% | 335ms（286–408） | 80–192ms | 196–265ms |
| LRU + 共享前缀 | **88.5%** | 288ms（216–383） | 19–130ms | 189–253ms |
| 信号调度 | 84.1% | 325ms（255–490） | 60–247ms | 195–243ms |
| 信号调度 + 共享前缀 | 87.8% | 285ms（236–349） | 30–96ms | 205–260ms |

- **共享前缀稳定有效。** 命中率从 77% 提到 88%，平均 TTFT 中位数降约 15%。
- **信号调度在这个规模没有额外收益。** 单独用时命中率从 77% 提到 84%，但有了共享前缀以后，LRU 已经到 88.5%。16 个用户时私有 KV 只有 GPU 层的 4 倍，LRU 已经够用；信号调度的优势要在更大的压力下才显现，见上面 200 用户的模拟。
- **延迟波动主要来自恢复耗时。** 同一配置在不同轮次之间差 3–6 倍。原因是这台机器 swap 用了 31GB，DRAM 层的主机缓冲区会被换出。真实部署要把 DRAM 层锁在内存里（mlock 或大页）。
- **之前的归因是错的。** 之前把 16 用户延迟没降归因于 `kv_unified`，实测不成立，见下面的 prefill 排查。

### prefill 速度排查（`bin/prefill_bench`）

M3 Max（30 核 GPU）上 Qwen3-1.7B Q4_K_M 的 prefill 速度：

| 测量 | 空 cache | 8k 深度 |
|---|---|---|
| llama.cpp 官方 `llama-bench` | 1770 tok/s | 985 tok/s |
| 本引擎 | 约 1240 tok/s | 约 710 tok/s |

- 约 1k tok/s 是这台机器的真实水平。上下文越深越慢，主要是注意力的开销；32k 微基准平均深度 16k，所以只有约 900 tok/s。
- 换 flash attention、ubatch（512、1024、2048）或序列数，差别都在 10% 以内。
- 本引擎比 `llama-bench` 慢 25–30%，原因还没查清。
- `kv_unified` 下，别的会话占着 64k 个单元时，一个 64–512 token 的请求只慢 9–14%，不足以解释回放结果。

## 局限

- **多用户是叠加出来的。** 多租户回放是把同一个人不同日子的轨迹叠在一起，不是真实的多用户数据。
- **附加 TTFT 只是一阶估算。** 模拟器用固定带宽和固定 prefill 速度，没有排队，也没有和计算争带宽。
- **共享前缀用的是实测均值。** 长度取 A2 报告的实测均值，没有逐个会话核对前缀内容是否完全一致。
- **MacBook 上没有独立显存。** DRAM 层就是主机缓冲区，两次拷贝都在同一块统一内存里完成，所以 DRAM→GPU 比真实 PCIe 快。SSD 层用的是 F_NOCACHE 的真实读写。
- **llama.cpp 用了 `kv_unified`。** 共享前缀因此能真正复用 KV 单元。注意力会扫到别的会话的单元，实测开销在 14% 以内。
- **延迟受内存压力影响大。** DRAM 层没有锁页，机器 swap 紧张时恢复耗时会差几倍，比较策略要看多轮结果。

## 有状态会话服务（`session/`，Rust）

把上面的想法做成能真用的东西：一个推理服务 `kdn-server`，加一个本地编码 Agent `kdn-agent`。

今天的 LLM API 是无状态的：Agent 每次调用都把全部历史重发一遍，服务端靠猜来命中缓存。这里反过来，**会话是服务端的对象，自己持有 KV**：

- **客户端只追加增量。** 一条用户消息，或一批工具结果。
- **生成的 token 直接进 KV。** 不会把历史重新分词。
- **客户端告诉服务端在等什么。** 等哪个工具、预计多久，还是在等人、或者已经结束。服务端据此决定 KV 放哪一层：
  - **GPU**：llama.cpp 里的一个活序列。
  - **DRAM**：`llama_state_seq_get_data` 拷出的主机缓冲区。
  - **SSD**：用 `F_NOCACHE` 写的文件，按 1GB 分块读写。
- **代码仓库和工具都留在本地。** 只有工具结果（截断后）作为增量发出去。

### 设计要点

- **放置策略。** 按 Agent 的等待提示决定放哪一层，逻辑照搬 `src/common.h` 的预测器和驱逐键。
  - 短工具等待：留在 GPU。
  - 等人：空闲 `--human-idle` 秒（默认 10）后降到 DRAM，DRAM 里再空闲 `--dram-idle` 秒（默认 600）后写 SSD。
  - 预测空档超过 `--cold-gap`（默认 30 分钟，比如 `done`）：直接写 SSD。
  - GPU 或 DRAM 超容量时，"预测下次使用最晚"的先挪走。DRAM 写 SSD 在后台线程做，写完之前会话仍可直接从内存恢复。
- **token 历史永远保留，并且落盘。** KV 丢了、换了模型、崩溃了，都能从 token 历史重算回来。
- **KV 文件带着它覆盖的 token 序列。** 恢复时和当前历史取公共前缀：
  - 回滚过：截掉多余部分（`seq_rm`）。
  - 崩溃前文件只覆盖一部分：恢复前缀，只重算剩下的。
- **持久化。** 每轮写一次元数据 JSON（token 历史、对话记录、轮次边界、预测器状态）。正常退出（Ctrl-C / SIGTERM）时，GPU 和 DRAM 里的 KV 全部写到 SSD，重启后直接从 SSD 恢复。
- **fork 和 rollback。**
  - fork 用 `seq_cp`：`kv_unified` 下共享 KV 单元，不复制。
  - rollback 截断 token 历史，GPU 上的 KV 用 `seq_rm` 截掉。
- **共享前缀缓存。** 系统提示加工具定义相同的会话，新建或重算时直接 `seq_cp` 已有的前缀。
- **聊天模板。** 从 GGUF 元数据读出 `tokenizer.chat_template`，按 Qwen3 的真实格式拼：
  - ChatML，Hermes 风格的 `<tools>`，输出用 `<tool_call>{...}</tool_call>`。
  - 工具结果放在 user 轮里的 `<tool_response>`，连续多个合成一轮。
  - 生成到 `<|im_end|>` 停。这个 token 留到下一次增量开头再进 KV，所以每段增量都从特殊 token 开始，分词边界和整段分词一致。
  - 默认非思考模式（生成提示带空的 `<think>\n\n</think>\n\n`），可以用 `--thinking` 或在建会话时打开。
  - `kdn-server check-template` 用 minijinja 渲染 GGUF 里的 Jinja 模板，逐步核对。
- **引擎线程。** 单个线程独占 llama context，从命令队列取请求，支持多个会话。HTTP 用 axum，append 用 SSE 流式返回。
- **FFI 选型。** 用 `server/csrc/shim.c` 包一层只有基本类型参数的 C 函数，再在 Rust 里手写 `extern "C"`，链接仓库里已经编好的 `libllama.dylib`，并写入 rpath。
  - 不用 bindgen：要 libclang，而且 `llama_context_params` 这些结构体布局会跟着 llama.cpp 版本变，Rust 侧手抄一份容易不知不觉地对错位。shim 按头文件编译，ABI 自然对得上。
  - 不用 `llama-cpp-2`：它会自带一份 llama.cpp 重新编译，版本也和这里不一样。

### 运行

```sh
make                                    # C demo 和 llama.cpp（已有就跳过）
curl -L -o models/qwen3-0.6b-q8_0.gguf \
  https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
cd session && cargo build --release && cd ..

./session/target/release/kdn-server                                          # 默认 0.6B，127.0.0.1:8399
./session/target/release/kdn-server --model models/qwen3-1.7b-q4_k_m.gguf    # 换 1.7B（工具调用好一些）

# 在任意目录（即"仓库"）里：
cd ~/some/repo
/path/to/kdn-demo/session/target/release/kdn-agent              # 交互；bash / 写文件要 y/n 确认
/path/to/kdn-demo/session/target/release/kdn-agent --yes -p "make test fails, find and fix the bug"
/path/to/kdn-demo/session/target/release/kdn-agent --continue   # 接着本目录上次的会话（重启后、几小时后都行）
/path/to/kdn-demo/session/target/release/kdn-agent --thinking   # 思考模式

# 自检和测量
./session/target/release/kdn-server check-template               # 增量拼接 vs GGUF 模板
./session/target/release/kdn-server verify --tokens 3000         # SSD/DRAM 恢复 vs 重算：token 和 logits 对比
./session/target/release/kdn-agent bench --tokens 3000 --reps 7  # 各层恢复延迟、每轮字节（要先起服务）
```

Agent 里的命令：

| 命令 | 作用 |
|---|---|
| `/sessions` | 列出服务端所有会话：token 数、轮数、KV 位置、上次活动 |
| `/resume <id>` | 切到某个会话 |
| `/new` | 新开会话 |
| `/fork` | 从当前会话 fork 一个，并切过去 |
| `/rollback N` | 只保留前 N 轮（每次 append 算一轮） |
| `/stats` | 当前会话、本进程的发送字节和无状态 API 对比、服务端各层占用 |
| `/demote dram\|ssd\|drop` | 强制降级，测试用 |
| `/quit` | 发 `done` 提示（服务端把 KV 写 SSD）后退出 |

工具：`read_file`、`list_dir`、`grep`、`write_file`、`edit_file`、`bash`。

- 文件工具只能访问工作目录内，路径先规范化再检查，挡住 `../` 和符号链接逃逸。
- `bash`、`write_file`、`edit_file` 默认要确认，`--yes` 自动同意。
- 工具输出超过 8KB 时，保留开头和结尾。

Agent 自动发等待提示：

- 要人确认时：先发 `human`。
- 开始跑工具时：发 `tool` 加预计时长。
- 一轮结束：发 `human`。
- 退出：发 `done`。

### API

| 接口 | 说明 |
|---|---|
| `POST /v1/sessions` | `{system, tools, thinking?, label?, sampling?}` → 会话；系统提示立即 prefill，或命中共享前缀 |
| `POST /v1/sessions/{id}/append` | `{messages:[{role:"user"\|"tool",content}], max_tokens?, temperature?, top_p?, top_k?, seed?, hint?, stream?, debug?}`；默认 SSE：`event: token` 逐段文本，`event: done` 是正文、思考内容、解析好的 `tool_calls`、统计和本次放置结果 |
| `POST /v1/sessions/{id}/hint` | `{waiting:"tool"\|"human"\|"ask"\|"done", tool?, expected_s?}` → 预测空档和放置结果 |
| `POST /v1/sessions/{id}/fork` | `{turn?}` → 新会话，`seq_cp` 共享父会话的 KV |
| `POST /v1/sessions/{id}/rollback` | `{turn:N}` 截到第 N 轮 |
| `GET /v1/sessions/{id}[?messages=true]` | token 数、层、轮数、等待状态（可带对话记录） |
| `GET /v1/sessions`、`DELETE /v1/sessions/{id}` | 列表 / 删除（连同 KV 文件） |
| `POST /v1/admin/sessions/{id}/demote` | `{tier:"gpu"\|"dram"\|"ssd"\|"drop"}` 强制搬运，测试用 |
| `GET /v1/admin/stats` | 各层会话数、占用、共享前缀、后台 I/O |

`append` 的 `done.stats` 字段：

| 字段 | 含义 |
|---|---|
| `kv_tier` | 请求到来时 KV 在哪：`gpu` / `dram` / `ssd` / `recompute` |
| `restore_ms` | 把 KV 搬回 GPU 的耗时 |
| `recompute_tokens`、`recompute_ms` | 需要重算的 token 数和耗时 |
| `prefix_hit_tokens` | 命中共享前缀的 token 数 |
| `new_tokens`、`prefill_ms` | 本次增量的 token 数和 prefill 耗时 |
| `ttft_ms` | 首 token 延迟 |
| `gen_tokens`、`decode_tok_s` | 生成的 token 数和解码速度 |
| `total_tokens` | 会话当前总 token 数 |
| `request_bytes` | 本次请求体字节数 |
| `waited` | 上一次等待的预测空档 vs 实际空档 |

### 实测（M3 Max 36GB）

测量期间机器很忙：swap 用了 21.5GB，还有一个模拟器在占 GPU。下面的延迟要看量级，同一配置在不同时段能差 2–3 倍，尤其是重算。

**模板核对**（`check-template`，0.6B 和 1.7B 的 GGUF 模板结果相同）

- **场景 A**：非思考模式，5 次 append（2 次用户提问，3 批工具结果，其中两批各有 2 个结果）。每一步都和模板 `add_generation_prompt=true` 的整段渲染一致，分段分词和整段分词逐 token 一致（432 token）。
- **场景 B**：思考模式，一次提问内的工具循环，逐字节一致（279 token）。
- **唯一的差别是有意的。** Qwen3 模板重渲染历史时，会删掉旧 assistant 轮的空 think 块（场景 A 删了 4 个）。KV 保留的是模型生成时真实看到的 token，不回头改写历史。

**正确性：同一个 append，KV 来源不同**（`verify`，3k token 历史，贪心生成 48 token，比较每一步的完整 logits）

| 模型 | GPU（基准） | DRAM 恢复 | SSD 恢复 | 从 token 历史重算 |
|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | — | token 一致，logits 差 **0** | token 一致，logits 差 **0** | token 一致，最大绝对差 **0.021** |
| Qwen3-1.7B Q4_K_M | — | token 一致，logits 差 **0** | token 一致，logits 差 **0** | token 一致，最大绝对差 **0.100** |

恢复是按位拷贝，所以完全一致。重算时 prefill 的分块方式不同（原来是逐轮 prefill 加逐 token 解码，重算是 1024 一块），浮点累加顺序变了，所以有微小差别，但贪心输出相同。

**恢复延迟**（`bench`，走 HTTP，同一会话每次先强制放到目标层，再 append 同一条短消息；中位数）

| 会话 | KV 在 | 搬运 | 重算 | TTFT |
|---|---|---|---|---|
| 0.6B，3.6k token（较空闲时） | GPU | — | — | **19ms** |
|  | DRAM | 23ms | — | **49ms** |
|  | SSD（358MB） | 93ms | — | **122ms** |
|  | 丢失 → 重算 | — | 2.8k token | **550ms** |
| 0.6B，3.1k token（繁忙时，7 次） | GPU / DRAM / SSD / 重算 | — | — | 38 / 67–86 / 119–125 / 1439–1537ms |
| 0.6B，8.0k token | GPU | — | — | **66ms** |
|  | DRAM | 58ms | — | **116ms** |
|  | SSD（919MB） | 198ms | — | **252ms** |
|  | 丢失 → 重算 | — | 7.3k token，6.2s | **6.2s** |
| 1.7B，3.1k token | GPU / DRAM / SSD / 重算 | — / 7ms / 75ms / 1.1s | — | 33 / 49 / 125 / **1182ms** |

- 重算时，系统提示加工具定义（约 730 token）命中共享前缀缓存，不算在重算里。
- Qwen3 0.6B 和 1.7B 的 KV 都是 112KB/token（f16，28 层 × 8 个 KV 头 × 128）。

**每轮发送字节：有状态增量 vs 无状态重发**（无状态 = OpenAI 风格请求体：全部 messages 加 tools）

| 场景 | 有状态 | 无状态 | 倍数 |
|---|---|---|---|
| bench 会话构造到 8k token，13 轮合计 | 24.2KB | 184.7KB | 7.6× |
| bench 第 13 轮（8k 上下文时读一段文件） | 2.1KB | 26.1KB | 12.3× |
| 端到端 Agent 修 bug（1.7B，16 次请求合计） | 4.1KB | 102.3KB | 25× |
| 4.4k token 会话里问一句话 | 62–145B | 17–18.5KB | 120–290× |

**端到端**（临时 git 仓库里放一个有越界 bug 的 `stats.c`：`for (i = 1; i <= argc; i++)` 读到 `argv[argc]`，`make test` 段错误）

- **管线全部跑通。** 读文件、`list_dir`、`grep`、跑 `make test`、`edit_file`、再跑测试都正常；等待提示、增量、流式输出、统计也都正常。整个过程 KV 一直在 GPU 上，单步 TTFT 15–125ms。
- **模型能力不够。**
  - 0.6B 改坏文件后反复跑同一条命令。客户端拦截了第 3 次重复调用，但模型还是在打转。
  - 1.7B 非思考模式能正确调用全部工具，但把 bug 归到 `argc - 1` 上。
  - 1.7B 思考模式会推理几百 token，但同样没看出 `<=`，最后改成了 `sum / argc`。
  - 结论：**管线可用，这两个小模型修不了这个 bug。** 要实际干活，用 `--model` 换更大的模型。

**重启、崩溃、换模型**

- **正常重启。** Ctrl-C 后 3 个会话的 KV 用 0.43s 写到 SSD。重启后 `kdn-agent --continue` 接上 4180 token 的旧会话：KV 从 SSD 恢复 99ms，TTFT 165ms。
- **崩溃（`kill -9`）。** 崩溃前 KV 在 GPU 上；SSD 文件是上次落盘的，覆盖 4536 token 里的前 4431 个。重启后恢复这个前缀（122ms），只重算 105 token，TTFT 242ms。
- **换模型。** 用 0.6B 建的会话在 1.7B 服务上打开时，KV 文件因模型标识不同被拒绝，从 token 历史重算。两个模型的分词器相同，所以会话可以跨模型继续。
- **提示驱动的自动降级。** `--human-idle 3 --dram-idle 8` 时，一轮结束后 3s 从 GPU 降到 DRAM，8s 后写到 SSD。

**fork / rollback**

- **fork。** 4684 token 的会话用 `seq_cp` fork，0.27ms。
- **rollback。** 子会话 `/rollback 2` 删掉 2922 token，接着提问，回答和截断后的历史一致；父会话不受影响。

**容量驱逐**（`--gpu-tokens 3000 --dram-tokens 1500`，4 个会话各 834 token）

- **GPU 满了挪等人的。** A 等 bash（2s）、B 等人（60s）、C 等 read_file（0.5s），这时 D 进来。被挪到 DRAM 的是 B。
- **超时的等待也会被挪。** 再来一个会话时，A 的 bash 已经超时，按重尾估计被挪到 DRAM。
- **DRAM 满了写 SSD。** DRAM 因此超出容量，B 被后台写到 SSD。

### 局限

- **小模型修不了 bug。** 0.6B 做工具调用会打转；1.7B 能用对工具，但这个 bug 没找对。服务和客户端本身对模型没有假设，换大模型只需要 `--model`。
- **一次只处理一个请求。** 单个引擎线程，不同会话之间没有合批，并发请求排队。
- **`kv_unified` 会拖慢解码。** 注意力会扫到别的序列的 KV 单元：fork 出来的子会话和父会话同时在 GPU 上时，解码从约 140 tok/s 掉到约 60 tok/s。
- **历史 think 块和模板不一致。** Qwen3 模板重渲染时会删掉历史里的 think 块，有状态会话不删（见上）。所以同一段对话交给无状态 API 时，模型看到的历史和这里不完全一样。
- **工具输出里的特殊 token 会生效。** 分词时开着 `parse_special`，和 HF 模板加分词器的行为一致。如果文件内容里有 `<|im_end|>` 这类字符串，会被当成特殊 token（提示注入面）。
- **容量按 token 计，偏保守。** fork 和共享前缀共用的 KV 单元会被重复计数。DRAM 层没有 mlock，机器 swap 紧张时恢复耗时会波动。
- **上下文满了只能回滚或新开会话。** 单会话上限是 `--n-ctx`（默认 32768），没有做上下文压缩或滑窗。
- **没有鉴权。** 默认只监听 127.0.0.1。
