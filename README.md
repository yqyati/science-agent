# Science Agent Sandbox

科学问题求解 Harness，提供多轮推理、搜索、网页阅读、Python 工具和答案判分。支持两种使用方式：

- **直接调用模型 API 评估 HLE360**：使用本仓库的独立入口，无需 Molt、Ray、本地模型权重或训练 GPU。
- **接入 Molt 训练或评估**：由 Molt 提供模型推理接口并记录训练轨迹，Harness 负责解题和工具执行。

以下说明独立 API 评估的使用方法。模型 API、工具沙盒与 Judge 的地址和凭据分别配置。

## 1. 准备环境

使用已安装本 Harness 依赖的 Python 环境，包括 `qwen-agent`、`openai`、`httpx`、`transformers` 及各工具依赖。启动脚本不会自动安装依赖。

运行环境需要能访问解题 API、沙盒网关及配置的 Judge 服务。工具凭据继续使用 `sandbox.local.env`；已有配置可以直接复用。首次使用时：

```bash
cp sandbox.local.env.example sandbox.local.env
```

按实际环境填写沙盒 Token、工具 API 凭据等，具体见[沙盒配置说明](docs/sandbox_backend.md)。沙盒配置文件和解题模型的本地配置文件均已加入 Git 忽略规则。

后续命令均在本仓库根目录执行。

## 2. 配置解题 API

```bash
cp hle_api.env.example hle_api.local.env
```

编辑 `hle_api.local.env`。例如，使用内网 DeepSeek Flash：

```bash
HLE_API_BASE_URL=http://s-20261020163637-mpm6m.ailab-agentshared.pjh-service.org.cn/v1
HLE_API_MODEL=DeepSeek-V4.1-Flash-1017
HLE_API_KEY=EMPTY
HLE_API_DIRECT=1
HLE_API_EXTRA_BODY='{"chat_template_kwargs":{"thinking":true,"reasoning_effort":20}}'
```

| 变量 | 用途 |
| --- | --- |
| `HLE_API_BASE_URL` | 解题 API 地址，通常以 `/v1` 结尾 |
| `HLE_API_MODEL` | 服务实际提供的模型 ID |
| `HLE_API_KEY` | 解题模型密钥；上述免鉴权内网服务使用 `EMPTY` |
| `HLE_API_DIRECT` | 设为 `1` 时，为解题 API 主机绕过环境代理；需要代理的服务设为 `0` |
| `HLE_API_EXTRA_BODY` | 服务专用的请求扩展参数，普通接口可设为 `'{}'` |

上述接口的 `reasoning_effort` 使用 0～100 的整数，示例设为 20。其他服务不一定接受相同参数，需按实际接口修改。

**不要用解题模型的密钥覆盖 `sandbox.local.env` 中工具使用的 `API_KEY`。** 解题模型使用独立的 `HLE_API_KEY`，Judge 也沿用自身配置。

## 3. 运行评估

先用前两题验证模型接口、工具调用和判分：

```bash
HLE_API_ENV_FILE="$PWD/hle_api.local.env" bash run_hle_api.sh \
  --run-dir "$PWD/api_eval_runs/flash-smoke" \
  --limit 2 --workers 2
```

确认两题均完成，并检查 `summary.json` 中的 `judge_errors`、`empty_answers` 后，再运行完整 HLE360：

```bash
HLE_API_ENV_FILE="$PWD/hle_api.local.env" bash run_hle_api.sh \
  --run-dir "$PWD/api_eval_runs/flash-hle360"
```

两次使用不同输出目录。`run_hle_api.sh` 在当前环境运行，不会自动提交集群作业。

### 默认配置

| 项目 | 默认值 |
| --- | --- |
| 数据集 | 与现有 Molt 评估相同的 `HLE_all_mini.jsonl`，360 题 |
| 并发 | 16 |
| 采样 | temperature=1.0，top_p=0.95 |
| 上下文硬上限 | 122,880 token（120k） |
| 上下文软阈值 | 110,592 token（108k） |
| compact、masking | 关闭 |
| 强制收尾 | 开启，复用当前收尾提示与逻辑 |
| 最终答案格式 | `boxed` |
| 工具 | 搜索、网页阅读、Python、Google Scholar |
| 沙盒配额 | `10.12.111.143/ailab-ai4sgi` |
| Judge | `bailian/deepseek-v4.1-flash`，开启 thinking，输出上限 6,000 token |

每题正常解题与判分复用同一个沙盒作用域。Judge 服务失败后，按现有重试策略使用替换沙盒或直连兜底。参考答案只进入判分请求，不发送给解题模型。

如需覆盖参数，可直接加命令行选项，例如：

```bash
HLE_API_ENV_FILE="$PWD/hle_api.local.env" bash run_hle_api.sh \
  --run-dir "$PWD/api_eval_runs/another-model" \
  --workers 8 --temperature 1.0 --top-p 0.95
```

全部选项：

```bash
python3 eval_hle_api.py --help
```

## 4. 查看结果

```bash
python3 eval_hle_api.py \
  --run-dir "$PWD/api_eval_runs/flash-hle360" --stage summary
```

输出文件：

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 模型、预算、采样、Judge 配置及数据和源码指纹，不保存密钥 |
| `dataset.json` | 本次选中的题目和参考答案 |
| `rollouts/*.json` | 完整对话、推理、工具调用、最终答案、结束原因和 API usage |
| `grades/*.json` | 逐题判分、Judge 切换记录及诊断指标 |
| `summary.json` | 完成进度、正确率、空答案、Judge 错误及截断数量 |
| `sandbox_trace.jsonl` | 带题目 ID 的模型调用和工具追踪 |

正确率口径：

- `pass1_completed`：正确题数 / 已有判分记录数。
- `pass1_overall`：正确题数 / 本次选中题目数；未完成题目暂按零计。
- `pending`：尚无判分记录的题数；全量完成时应为 0。
- `judge_errors`：未获得有效判分的题数；保留 score=-1，正确率统计按零计。

## 5. 断点续跑与单独重判

中断后执行相同命令，会复用已有轨迹和判分，从未完成部分继续。相同输出目录会校验配置、数据和源码指纹；更换模型、预算、Judge 或修改代码后，请使用新的输出目录。

只对 Judge 失败的已有答案重新判分，不重新生成解题轨迹：

```bash
HLE_API_ENV_FILE="$PWD/hle_api.local.env" bash run_hle_api.sh \
  --run-dir "$PWD/api_eval_runs/flash-hle360" \
  --stage judge --retry-failed
```

有效的判错不会触发兜底，也不会因 `--retry-failed` 被重判。若使用默认 `all` 阶段并添加 `--retry-failed`，还会重新生成失败或空答案轨迹。

也可以先 `--stage rollout` 只生成，再用 `--stage judge` 判分。恢复运行时保留原来的题目选择和采样参数。

## 6. 更换 Judge

可在 `hle_api.local.env` 中指定服务列表：

```bash
SCIENCE_AGENT_JUDGE_CONFIG=/absolute/path/to/judge_services.json
```

参考 [judge_services.example.json](judge_services.example.json)。列表区分沙盒调用与直接调用，服务失败后顺序尝试下一项。密钥通过环境变量引用，配置中填写变量名。

启动脚本还支持 `HLE_JUDGE_MODEL`、`HLE_JUDGE_BASE_URL`、`HLE_JUDGE_THINKING`、`HLE_JUDGE_MAX_TOKENS`、`HLE_JUDGE_TIMEOUT` 覆盖默认 Judge 参数；显式服务列表中的配置按列表规则生效。

## 7. API 模型的差异

- **上下文计数**：可以通过 `--tokenizer-path` 指定解题模型自己的本地 tokenizer；否则使用请求长度估算，并记录 `context_tokens_estimated`。120k/108k 沿用此前预算配置，实际 token 边界仍取决于 API 服务。
- **图片题**：HLE360 有 48 题声明了图片。入口保留图片输入，目标 API 必须支持图片；不支持时会记录解题错误和空答案。图片预算默认估计为每张 4,096 token，可用 `--image-tokens-per-image` 调整。
- **输出限制**：如果目标 API 单次最多输出 32k，可加 `--max-output-tokens 32768`，该配置差异会记录在结果中。
- **超时**：单题 9,000 秒的时间检查在轮次边界执行，不会强杀当前请求；Judge 重试耗时另计。

## 8. 验证与相关文档

```bash
python3 -m unittest discover -s tests -p test_hle_api.py
python3 -m unittest discover -s tests -p test_judge_client.py
bash -n run_hle_api.sh
```

当前验证包括模拟接口、预算与参数转发、轨迹保存、续跑、只重判和沙盒作用域等；目标模型 API 的实际兼容性需通过小规模评估确认。

- [API 评估详细说明](docs/hle_api.md)
- [沙盒配置及 Judge 服务列表](docs/sandbox_backend.md)
- [Harness 原始快照说明](SNAPSHOT.md)
