# 异步关键进展核查（独立版本）

原版 `react_agent_qwen_api.py`、`react_agent_openai_api_tools.py`、system prompt、
训练和评测入口均不改动。只有显式选择新版 agent，且设置
`SCIENCE_AGENT_PROGRESS_VERIFIER=1`，才追加关键进展的 system 指令并启动核查。
仅设置环境变量，不会让原版入口启用 verifier。

## 选择新版

调用方使用与原来相同的构造参数和 `_run` 接口，只换 import：

```python
# Qwen 原生入口的新版
from react_agent_qwen_api_progress import MultiTurnReactAgent

# 或 OpenAI 工具入口的新版（二选一）
from react_agent_openai_api_tools_progress import MultiTurnReactAgent
```

上述模块位于 `inference/`，与原版一样需要该目录在 Python import 路径中。
新版采用继承方式复用原版推理循环，没有复制或改写原循环。
旧版 Molt adapter、`eval_hle_api.py` 和 `eval_critpt.py` 仍使用旧版；本次不切换它们。

参考根目录 `progress_verifier.env.example` 配置独立 verifier 服务，例如：

```bash
export SCIENCE_AGENT_PROGRESS_VERIFIER=1
export PROGRESS_VERIFIER_BASE_URL='https://your-verifier-service/v1'
export PROGRESS_VERIFIER_MODEL='your-verifier-model'
export PROGRESS_VERIFIER_API_KEY='your-verifier-key'
export PROGRESS_VERIFIER_LOG_DIR='/absolute/path/to/run/progress_verifier'
```

使用正常 harness 依赖环境，包括 `qwen-agent` 和 `openai`。不会复用或覆盖
解题模型、Judge、工具的客户端或凭据。接口须支持 Chat Completions；
图片题需 verifier 服务支持图像输入。不支持或无法判定时应返回 `uncertain`。

## 模型输出与执行位置

system 指令让模型自主选择有实质结论的检查点，每轮至多一个：

```text
<think>本轮推理……</think>
<progress>已确立的中间结论，以及此前已经获得的证据。</progress>
<tool_call>下一次工具调用……</tool_call>
```

这是位置示意，具体工具调用沿用模型原来的文本或原生 function calling 格式。
`<progress>` 放在可见 content 中，位于推理之后、同轮工具调用之前。
不要求每次工具调用都报告进展；计划和未返回的工具结果不是进展。

检测到声明后，立即在后台提交核查，并继续执行本轮工具。
下一轮模型请求前，只取已经完成的反馈，不等待未完成的核查。
通过的核查只记审计日志，不返回 agent；失败返回；不确定仅在 issue、evidence 和具体
suggested_check 均非空时返回。verifier 指令要求 suggested_check 写明获取缺失证据的
具体动作，否则留空。此筛选不能保证自然语言建议一定有用，仍需审计。
接口失败仅记录。无反馈不代表通过，可能仍在核查、已跳过或服务不可用。

反馈作为带 checkpoint 编号、原声明和核查依据的 `checkpoint_feedback` 消息注入，
位置在工具响应之后，不插入未完成的工具调用与响应之间。
当前轮已经生成的调用不会因迟到反馈自动撤销，模型需在后续轮次修正。

## Verifier 输入与输出

每次输入包含：

- 原题 user message，包括原有图片内容；不包含标准答案、Judge 分数或训练 reward。
- 上个检查点的声明和当前状态，以及此前检查点的状态摘要。未完成核查明确标为 pending。
- 从上次检查点之后到当前声明之前的真实推理、消息、工具调用和已返回结果。
- 当前声明和 checkpoint 编号。当前声明之后的文字、尚未执行的下一次调用均不进入本次证据。

上次声明之后的文字和工具调用保留在下次证据区间中，不丢掉对应工具调用与响应的关联。
证据使用完整轨迹，不使用给主模型做 compact/masking 后的视图。
先前检查点状态不等于已证明事实，verifier 被明确要求独立审查。
Verifier 本身暂不执行工具，也不主动读取沙盒文件；缺乏必要证据时应报告不确定。

输出使用严格 JSON：

```json
{
  "verdict": "pass",
  "progress": "substantive",
  "issue": "",
  "evidence": "依据所提供的计算或工具结果……",
  "suggested_check": ""
}
```

`verdict` 可为 `pass/fail/uncertain`；`progress` 可为
`substantive/redundant/uncertain`。服务超时、返回截断或无效 JSON 记录为
`unavailable`，不当作推理错误。无自动重试，不给数值过程奖励，不修改最终 Judge。

## 预算与保存

默认每条轨迹最多 8 个检查点，同时最多 2 个核查，单次输入最多 2,000,000 UTF-8
字节，API 超时 90 秒，输出上限 2,048 token。容量满、标签无效或输入过大时跳过
本次核查并记录原因；证据游标不前移。模型可继续使用工具，不因核查被跳过而等待。
并发上限按轨迹计算：16 条轨迹最多同时 32 个 verifier 请求。

每条启用的轨迹在 `PROGRESS_VERIFIER_LOG_DIR` 下写独立 UUID 命名的 JSONL：

- `progress_checkpoint_submitted`：检查区间、声明、完整 verifier 输入。
- `progress_verification_completed`：核查结果和耗时。
- `progress_feedback_delivered`：反馈已经注入主模型上下文。
- `progress_feedback_suppressed`：结果静默保留及原因，不重复处理。
- `progress_checkpoint_skipped`：跳过原因。
- `progress_verifier_closed`：轨迹结束时的检查点快照。

该审计文件不受通用 trace 的字段截断限制；包含原题、图片和工具证据，注意磁盘用量。
结果字典新增 `progress_verifier`，含 `log_path`、检查点快照及 `audit_write_errors`。
审计写入失败会告警并累计错误数，不能把缺失审计的样本当作完整过程监督。

结束或异常时关闭线程池，不等待 verifier。尚在执行的请求可以在轨迹返回后写入
完成事件，因此返回快照中 pending 不等于失败；后续分析应按 JSONL 的最终事件合并。
反馈标记为 delivered 只表示已经加入上下文，不保证随后的模型请求成功。
服务调用受超时约束；Python 进程退出时仍可能等待正在执行的线程完成。

## 验证

```bash
python3 -m pytest tests/test_progress_verifier.py -q
```

覆盖真实原版推理循环中的工具执行与延迟注入、原题图片、区间衔接、未完成检查点、
反馈不丢失、不重复发送、服务异常、容量限制、无阻塞结束及异常清理。
本测试不需要访问真实 verifier API；真实服务的模型格式和图像兼容性需单独试运行。

## 真实链路 smoke

新增独立入口，不改原评测脚本。使用模型 API 和 Python 沙盒，不申请 GPU：

```bash
PYTHON_BIN=/path/to/harness/python \
PROGRESS_VERIFIER_EXTRA_BODY='{"thinking":{"type":"disabled"}}' \
bash run_progress_verifier_smoke.sh \
  --run-dir log/progress-smoke-new-run \
  --solver-extra-body '{"thinking":{"type":"disabled"}}'
```

示例 thinking 参数适用于本次 DeepSeek 服务，其他服务按接口调整或省略。
该 smoke 启动器加载 `sandbox.local.env`，仅在此 smoke 中默认将 `API_BASE`、
`SUMMARY_MODEL_NAME`、`API_KEY` 用作解题和 verifier 服务，方便单独验证接口。
显式设置 `PROGRESS_VERIFIER_*` 可以覆盖核查服务；这不改变核心 verifier
没有凭据回退的规则，也不会调用最终 Judge。

模型完成四次独立 Python 调用，给出整数求和与平方和，并自主输出关键进展。
`summary.json` 检查答案、工具实际执行、核查提交、有效返回、反馈进入后续模型
请求（若有需要返回的反馈）、通过结果静默处理及审计落盘；任一条件不成立，入口以非零状态退出。
`rollout.json`、`requests.json`、`trace.jsonl` 和 `verifier/*.jsonl` 保存对应证据。
使用新的 run-dir，避免与前次运行混合。
