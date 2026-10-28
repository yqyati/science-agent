# HLE360：独立 API 评估

`eval_hle_api.py` 直接通过 OpenAI Chat Completions API 调用解题模型，复用当前
Harness 的问题包装、ReAct 循环、搜索、读网页、Python、强制收尾和 Judge 链。
不需要 Molt、Ray、本地 checkpoint 或训练 GPU。仍需当前 Harness 的 Python
依赖（包括 qwen-agent、openai、httpx），以及可用的工具沙盒和工具 API 凭据。

## 配置与运行

在本仓库目录执行：

```bash
cp hle_api.env.example hle_api.local.env
```

编辑 `hle_api.local.env`，填写**解题模型**的地址、模型名和密钥：

```bash
HLE_API_BASE_URL=http://s-20261020163637-mpm6m.ailab-agentshared.pjh-service.org.cn/v1
HLE_API_MODEL=DeepSeek-V4.1-Flash-1017
HLE_API_KEY=EMPTY
HLE_API_DIRECT=1
HLE_API_EXTRA_BODY='{"chat_template_kwargs":{"thinking":true,"reasoning_effort":20}}'
```

这里的整数 effort=20 是内网 DeepSeek 接口的配置，不代表其他服务也接受这个
字段。`HLE_API_DIRECT=1` 仅为解题 API 主机绕过环境代理；使用需要代理的外部
服务时设为 0。换普通 OpenAI 兼容服务时，可以将 `HLE_API_EXTRA_BODY` 设为 `'{}'`，或者
填写该服务要求的扩展参数。模型名使用目标服务实际提供的 ID。

工具凭据继续使用现有 `sandbox.local.env`。解题模型的 `HLE_API_KEY` 与工具使用的
`API_KEY`、Judge 凭据分开；不要覆盖工具的 `API_KEY` 来设置解题模型。
`hle_api.local.env` 已加入 Git 忽略规则。

先用两题检查模型接口、工具调用和 Judge：

```bash
HLE_API_ENV_FILE="$PWD/hle_api.local.env" bash run_hle_api.sh \
  --run-dir "$PWD/api_eval_runs/flash-smoke" --limit 2 --workers 2
```

全量 HLE360：

```bash
HLE_API_ENV_FILE="$PWD/hle_api.local.env" bash run_hle_api.sh \
  --run-dir "$PWD/api_eval_runs/flash-hle360"
```

启动脚本先加载沙盒凭据，再加载解题 API 配置，并显式恢复当前 DeepSeek Judge
默认值，避免旧的 `sandbox.local.env` 中 `o3-mini` 覆盖默认模型。
直接运行 Python 时需自行准备工具环境，并设置 Judge 环境变量。
脚本不自动安装依赖，也不提交 RJob；可以在能访问解题 API 与沙盒网关的现有
环境运行。全量启动前应确认 smoke 题目完成且 Judge 有效。

## 默认评估配置

| 项目 | 默认值 |
| --- | --- |
| 数据 | 与最近 Molt 作业相同的 `HLE_all_mini.jsonl`，360 题 |
| 并发 | 16 个独立题目，每题独立沙盒生命周期 |
| 采样 | temperature=1.0，top_p=0.95 |
| 上下文硬上限 | 122,880 token（120k） |
| 软阈值 | 110,592 token（108k） |
| compact / masking | 均关闭 |
| 强制收尾 | 开启，复用当前函数、提示和收尾采样参数 |
| 答案格式 | `boxed`，与现有问题包装、收尾示例一致 |
| 工具 | search、visit、python、google_scholar |
| 最大模型调用轮数 | 100，另有现有收尾请求 |
| 单题时间检查 / 单次请求超时 | 9,000 秒 / 600 秒 |
| 沙盒配额 | `10.12.111.143/ailab-ai4sgi` |
| Judge | `bailian/deepseek-v4.1-flash`，thinking enabled，6,000 输出 token |
| Judge 兜底 | 共用现有 `judge_client.py`；默认沙盒服务后接内网直连 Flash |

单题时间限制沿用 Harness 在轮次边界的检查，不会强杀正在执行的 HTTP 或工具请求。
Judge 重试等待和耗时另计。强制收尾沿用现有的 temperature=0 行为。

解题模型 API 直接由当前进程调用；工具及默认主 Judge 继续通过沙盒执行。
一题正常解题与判分处于同一个沙盒作用域，Judge 失败时仍可能按现有策略使用
替换沙盒或直连服务。参考答案只用于判分，不发送给解题模型。

Judge 可通过 `SCIENCE_AGENT_JUDGE_CONFIG` 指定有序服务列表，详见
[sandbox_backend.md](sandbox_backend.md#ordered-judge-services)。使用启动脚本时，
单项覆盖变量为 `HLE_JUDGE_BASE_URL`、`HLE_JUDGE_MODEL`、`HLE_JUDGE_THINKING`、
`HLE_JUDGE_MAX_TOKENS` 和 `HLE_JUDGE_TIMEOUT`；密钥仍使用现有 Judge 配置。

## 上下文计数与图片

API 模型不一定提供本地 tokenizer。可用 `--tokenizer-path /local/solver-tokenizer`
加载解题模型自己的本地 tokenizer；未提供或加载失败时使用 Harness 的请求长度
估算，并在结果中记录 `context_tokens_estimated=true`。请求输出预算受剩余上下文
空间限制，不额外施加累计输出 120k 的限制。

因此这里沿用的是 **120k/108k 的预算配置和触发逻辑**，实际 token 边界无法保证与
本地 Qwen 模型完全一致，最终硬限制仍由 API 服务执行。若服务规定每次最多输出
32k，可显式加 `--max-output-tokens 32768`，该差异会记录在 manifest 中。

HLE360 中有 48 题声明了图片。入口保留图片输入，**不会静默删掉图片题**；服务必须
支持图片才能完成这些题。无法处理图片会保存明确的解题错误，并在整体统计中计为
空答案。图片上下文成本默认按每张 4,096 token 估计，可通过
`--image-tokens-per-image` 调整；不依赖 Qwen 专用视觉 processor。

## 保存、续跑与重判

每个 run 目录保存：

- `manifest.json`：采样、预算、模型地址、Judge 配置、数据和代码指纹，不保存密钥。
- `dataset.json`：本次选中的题目和参考答案。
- `rollouts/*.json`：完整消息、推理、工具调用、答案、结束原因和 API usage。
- `grades/*.json`：逐题判分、Judge 服务切换记录、截断标记及诊断信息。
- `summary.json`：进度、已完成正确率、全量口径正确率、空答案及 Judge 错误数量。
- `sandbox_trace.jsonl`：按 question_id 关联的工具和模型请求追踪。

重复执行相同命令会跳过已保存轨迹和判分。只重试 Judge，复用所有已有答案：

```bash
HLE_API_ENV_FILE="$PWD/hle_api.local.env" bash run_hle_api.sh \
  --run-dir "$PWD/api_eval_runs/flash-hle360" --stage judge --retry-failed
```

有效的判错不会触发服务切换，也不会被 `--retry-failed` 重判。`--stage all
--retry-failed` 还会重新生成失败或空答案轨迹，应当有意使用。也支持
`--stage rollout` 只生成、稍后再执行 judge 阶段。

无需 API 或模型依赖即可查看已保存结果：

```bash
python3 eval_hle_api.py --run-dir "$PWD/api_eval_runs/flash-hle360" --stage summary
```

`pass1_completed` 的分母是已有判分记录数，`pass1_overall` 的分母是本次全部题目数，
包括尚未完成的题目。无效 Judge 保留 score=-1，汇总正确率中按零计，同时单独报告
`judge_errors`。完整评测须确认 `pending=0`，并检查 Judge 异常数量。

恢复运行时会验证配置、数据和源码指纹。修改模型、Judge、预算、题目选择或代码后
请创建新的 run 目录；不要将 smoke 的 `--limit 2` 结果直接混入全量目录。同一目录
只允许一个进程运行，防止重复请求和文件覆盖。完整轨迹可能含题目、工具返回和
内联图片，磁盘占用仍需按实际生成量预留。

## 本地验证

```bash
python3 -m unittest discover -s tests -p test_hle_api.py
python3 -m unittest discover -s tests -p test_judge_client.py
bash -n run_hle_api.sh
```

以上测试使用模拟 API 和工具，验证预算、整数 effort、图片保留、参考答案隔离、
断点续跑、仅重判、有效判错缓存、空答案及汇总口径；不代表目标 API 已联调通过。
