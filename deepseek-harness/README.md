# 单路 DSH × Molt agentic RL

每个 episode 是一个原生 DSH session：模型调用工具、提交候选解、读取 verifier 反馈、继续优化。相同训练题独立采样多条 episode；没有外层三路搜索和择优继承。Molt 使用 `--train.agent_path` 加载本目录插件。

所有新增代码、复制的 runtime、数据和日志均放在本目录。现有 `science-agent-sandbox`、Molt 框架和历史训练脚本均未修改；复用现有 Python/训练镜像，不安装依赖。

## 文件

- `molt_adapter/agent.py`：Molt `ChatAgentRunner` 插件；异步启动独立 DSH worker，返回 reward 和 episode 指标。
- `molt_adapter/worker.py`：原生 DSH session，关闭 subagent 与自动 compaction。
- `molt_adapter/bridge.py`：连接 Molt session endpoint，锁定采样参数，兼容空工具名 SSE；将 Molt JSON 原始生成文本中的 Qwen JSON/XML 工具调用转成 DSH 工具调用和 SSE。原始 token/logprob 由 Molt 保存。
- `molt_adapter/isolation.py`：非 root Bubblewrap，按 episode 挂载 workspace/home，隐藏 verifier、训练数据、其他 episode 及宿主 `/proc`；共享网络供本地 policy/verifier 和网页工具使用。
- `molt_adapter/verifier.py`：调用交接包真实 verifier，保存提交与评分，计算 episode reward。
- `molt_adapter/dataset.py`：冻结训练题转换与源文件 SHA256 校验。
- `molt_adapter/train.py`：独立 Ray 入口，传递 DSH 环境变量，再运行原版 Molt CLI。
- `molt_adapter/scripts/train_4plus4.sh`：独立 8 GPU 配方。
- `vendor/PROVENANCE.json`：复制的 SDK/runtime/profile/Bubblewrap 来源和哈希。

## 训练数据与 reward

`data/train.jsonl` 有 **2,399 道训练题**：AC 524、LJCR 1,841、Endless 34。来源为已有冻结训练题快照；源文件及 verifier 代码哈希均已核验，记录在 `data/train.provenance.json`。不引入评测轨迹，prompt 不含 teacher 解答。

这是可用的有限 verifier 子集，并非交接包全部 3,536 题。第一版未接入需要 Lean 的 SAIR。所有输入必须标记 `generation_split=train`；新增数据仍需独立核对题目来源，标签本身不能证明无评测泄漏。

只对可信 verifier 判定有效且目标值有限的候选评分；无有效解为 0。优化方向来自题目，取整个 episode 最优目标值，不按提交次数加分，完全相同的提交直接复用评分。verifier/模型服务故障上抛，不作为零分训练样本。

- 从头探索，有公开 baseline：`reward = 0.5 + 0.5*tanh(有向改进/scale)`，`scale=max(1, abs(baseline)*0.1)`。同题各采样固定相同 baseline/scale。
- 从头探索，无 baseline：最小化使用 `1/(1+max(0, objective)/scale)`；最大化使用 `max(0, objective)/(scale+max(0, objective))`，默认 scale=10。
- 只验证有效性的题目：有效为 1。
- 接手状态：label 可额外提供 `initial_candidate` 字符串；先重新验证，作为初始文件交给模型，reward 为 `tanh(max(0, 有向改进)/scale)`。维持原解为 0，严格改进才有收益。当前数据转换只生成从头探索题，尚未自动导入瓶颈状态。

## 启动

### 新增：v451-p9 优化类 1K，84k / 48 并发 / 8 GPU

`data/verifiable_1k_v451p9/train.molt.jsonl` 包含全部 1,000 题（target400 + source300 + general300），独立于原 2,399 题文件。按用户要求纳入 target400；其 49 个种子来自 OpenProblemBench-L2 和 EinsteinArena，因此本训练池不声明与这些评测来源独立。来源和 SHA256 记录在同目录 `train.provenance.json`、`source_index.jsonl`。

公开题面、参数、答案格式和精确 verifier 已快照至本目录，训练仅需 GPFS2 挂载。`train.jsonl` 保留原结构；实际加载 `train.molt.jsonl`，将异构 `dsh_item` 编码成 JSON 字符串，在 DSH agent 入口恢复，避免 Arrow 对各题参数结构和大整数强制推断失败。未复制私有 `trusted_specs.jsonl` 和 teacher 答案；reward anchor 使用公开题目元数据中的 `qa.baseline_score`，只进入可信评分 label，不进入 prompt。评分使用 `verify_candidate.py` 的 `valid/score/direction/feedback`，兼容旧数据 verifier。

默认基模为原始 `Qwen3.5-35B-A3B`，路径为 `/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/model/Qwen3.5-35B-A3B`，可通过 `MODEL_PATH` 覆盖。新入口 `molt_adapter/scripts/train_v1114_1k_84k_8gpu.sh` 固定 86,016 token、全部 1,000 题、8×8 采样、生成组并发 6（最多 48 条单路 DSH）、4 actor + 4 vLLM；其余沿用下面的异步训练配方。episode 仍默认 1,800 秒，每次生成最多 16,384 token，未改成 teacher 的四小时并行搜索。

```bash
# 仅查看提交命令
bash "$DSH_ROOT/molt_adapter/scripts/submit_v1114_1k_84k_8gpu.sh" --dry-run
# 提交单机 8 卡训练
bash "$DSH_ROOT/molt_adapter/scripts/submit_v1114_1k_84k_8gpu.sh"
```

提交器默认使用 `yqy:molt-cu13-runtime` 镜像、48 CPU、768,000 MiB、`ailab_ai4sgi_pool`，仅挂载 GPFS2。新脚本本次尚未提交训练；已通过 shell 语法和命令展开检查。独立全量 verifier QA 对每题检查有效解、无效解、另一有效得分，共 3,000 次调用全部通过，报告位于 `runs/v1114_1k_preparation/verifier_qa.json`；这不等同于 GPU RL 已验收。

### 原有训练入口

在现有 Molt 训练镜像、单节点 8 GPU 环境运行。必须显式指定实际要训练的 HF checkpoint；没有默认沿用历史 checkpoint。默认 64k，上调到 128k 可设 `MAX_LEN=131072`，需另行验证显存和吞吐。

```bash
export DSH_ROOT=/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/yangqingyu/workspace/harness/deepseek-harness
export MODEL_PATH=/path/to/qwen3.5-35b-a3b-hf-checkpoint
export DATASET="$DSH_ROOT/data/train.jsonl"
bash "$DSH_ROOT/molt_adapter/scripts/train_4plus4.sh"
```

保留参考配方：4 actor + 4 vLLM，8 题 × 每题 8 次采样、生成组并发 6（最多 48 个 episode）；异步队列 2，partial rollout 开启；`reinforce_baseline`、geo IS、ratio mask `[0.5,5]`、score centering=0、prefix caching 开启，保留最终模型保存。

默认每 episode 1,800 秒、128 次模型请求，每次生成最多 16,384 token；Molt 对上下文剩余容量继续裁剪。`MAX_LEN` 是上下文/轨迹段容量，不是 4 小时预算，也不是任意历史重写后全 episode 的累计 token 上限。wall/request cap 仍限制整个 episode。达到预算时返回已验证最优解并标记 truncated；只在最终口头回答中出现、尚未验证的候选不会获得分数。

可设置 `DSH_WALL_SECONDS`、`DSH_MAX_REQUESTS`、`DSH_MAX_COMPLETION_TOKENS`、`DSH_REQUEST_TIMEOUT`、`DSH_PYTHON`；默认工具 Python 为镜像的 `/usr/bin/python3`，worker 默认使用训练 Python。工具可用库以目标镜像为准。使用现有 GPFS2 交接包 verifier，无需访问 `/mnt/shared-storage-user`：

```text
/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/sujiaming/DSH_Teacher3536_Handoff_20261113/teacher_rollout/verifier_sources
```

`DSH_VERIFIER_ROOT` 可覆盖该路径。运行日志、候选、真实评分、DSH events、模型请求参数、Molt rollout dump 和 checkpoint 存入 `runs/<RUN_NAME>/`。此脚本是在作业容器内运行的入口，尚未新增 rjob 提交器。

## 已验证与待验证

2026-11-18 本机 CPU 验收：

- SDK/runtime 可导入，真实非 root Bubblewrap 文件隔离通过。
- 新增 **8 项测试全部通过**，包括原生 DSH 调 bash→真实 LJCR verifier→反馈后再次推理，以及真实 Molt chat capture→DSH→reward 回传。生成 action token 与模拟 policy 输出逐一一致，logprob 对齐；测试不使用外部模型或 GPU。
- Python 编译与 shell 语法检查通过。
- Molt 现有相关测试：53 passed、2 skipped、2 failed。两项失败均为 `test_masked_history_rerender_preserves_tool_protocol` 在 masking/compaction 关闭时，测试期待丢弃工具字段，但现有代码始终保留工具字段。本次没有修改 Molt 文件，也没有改动这些断言。
- **尚未启动实际 GPU RL**；目标镜像、真实 Qwen checkpoint 的 tool/thinking 模板、训练梯度更新、异步权重切换及 partial resume、多并发资源占用、64/128k 显存仍需小规模训练验收。CPU 假 token policy 验收不能代替这些检查。网页工具能力沿用原生 DSH，本次未重新验收外网搜索。

复验（需要宿主 namespace 权限，普通受限开发沙箱不能运行 Bubblewrap）：

```bash
DSH_RUN_NATIVE_TESTS=1 /root/miniconda3/envs/verl/bin/python -m pytest -q "$DSH_ROOT/tests"
```

CPU 集成验收摘要保存在 `runs/molt_native_acceptance/acceptance.json`。

## 单卡 rjob 镜像验收

2026-11-18：`dsh-molt-smoke-1gpu-1118-r2-80336888` 已 Succeeded。使用用户指定镜像 `registry.h.pjlab.org.cn/ailab-ai4sgi-ailab_ai4sgi_pool/yqy:molt-cu13-runtime`，1 H200、8 CPU、64,000 MiB，仅挂载 GPFS2。

镜像 Python `/usr/bin/python3` 为 3.12.3，PyTorch 2.13.0+cu130、CUDA 13.0。CUDA 矩阵运算、Bubblewrap 非 root 文件隔离、原生 DSH 工具执行、真实 LJCR verifier、Molt action token/logprob 对齐及 reward 回传通过。没有安装依赖。镜像没有 pytest，因此 `scripts/rjob_smoke_1gpu.sh` 直接调用 `molt_adapter/acceptance.py`；此前因缺 pytest 失败的作业不代表 runtime 失败。

日志及环境记录：`runs/dsh-molt-smoke-1gpu-1118-r2/`。模拟 policy 发出两次生成，得到一个轨迹段和 reward 0.880797。模型生成部分使用固定模拟 token；此项验收没有加载实际模型权重，没有 RL 梯度更新。退出日志中的 uvicorn lifespan `CancelledError` 来自测试关闭服务器，作业退出码为 0。

## 评分接口容错修复

`/score` 要求 `{"content": "候选文本"}`。缺失字段、传入对象或非法 JSON 返回 HTTP 400，保留本次 episode，允许模型纠正后重试。评分调用方退出或超时导致的 BrokenPipe 不再标记整条 rollout 为基础设施失败；已验证候选保留。模型请求通道断连、真实 verifier 故障继续上抛，避免产生错误训练样本。异常审计包含请求路径和栈。2026-11-19 回归 40 passed、1 skipped；既有 worker 已导入旧代码，修复对新启动 worker 生效。
