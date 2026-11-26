# Science benchmarks through a model API

`run_science_api.sh` uses the same solver API, Sandbox tools, Judge chain,
resume logic and output layout as `run_hle_api.sh`. It does not launch a model
server, allocate GPUs, import Molt or submit cluster jobs. The model service can
be an external provider or an existing OpenAI-compatible server.

| `--benchmark` | Default questions | Protocol |
|---|---:|---|
| `hle` | 360 | Existing HLE candidate selection and strict boolean grading |
| `amobench` | 50 | All final boxes, list of acceptable references, mathematical equivalence |
| `fso` | 100 | Complete final response including labels and units; grade every requested part |
| `imo_answerbench` | 400 | Original dataset user prompt; same answer/grading protocol as AMO |
| `cmt_benchmark` | 50 | Question plus parameters/functions; complete final response; strict boolean grading |

These four added protocols match Molt's `prepare_amobench.py`,
`prepare_frontierscience_olympiad.py` and `ScienceLegacyReactAgent` evaluation
branches. Forced finalization is disabled for the four added benchmarks, as in
those GPU recipes; the existing HLE default remains enabled. Defaults otherwise
remain temperature 1.0, top-p 0.95, 122880 trajectory/context tokens, 16 workers,
and search/visit/python/google_scholar tools. Context estimates without a local
tokenizer remain approximate. Provider context/output limits still apply.

## Configure and run

Reuse the [HLE API solver configuration](hle_api.md): `sandbox.local.env` holds
tool credentials and the Judge service configuration; a separate
`hle_api.local.env` holds `HLE_API_BASE_URL`, `HLE_API_MODEL`, `HLE_API_KEY` and
optional provider request options. The `HLE_API_*` names also configure the new
benchmark entry, so existing solver configuration files work unchanged.
`SCIENCE_AGENT_JUDGE_CONFIG` continues to select the configured Judge chain.

For example, after filling in the ignored solver configuration file:

```bash
HLE_API_ENV_FILE="$PWD/hle_api.local.env" bash run_science_api.sh \
  --benchmark amobench --run-dir "$PWD/api_eval_runs/amo-api-smoke" --limit 2
```

For a full evaluation, omit `--limit`. Select the other datasets by changing
`--benchmark` to `fso`, `imo_answerbench` or `cmt_benchmark`, using a distinct run
directory for each benchmark, model or configuration. CLI `--base-url`,
`--model`, `--api-key-env` and `--request-extra-body` are also supported.
No evaluation starts merely by preparing or summarizing data.

Default input paths use the same research datasets as the GPU recipes. On
another machine, copy the original JSONL files and pass `--input /path/to/file`.
`--expected-rows` overrides the full dataset size when using a custom subset.
The full input is checked for duplicate IDs and row count before `--limit`.
CMT image questions are rejected, matching the current GPU adapter; the
supported CMT set is text-only.

Validate and save the converted questions without credentials, tools or network:

```bash
python3 eval_hle_api.py --benchmark imo_answerbench \
  --run-dir /tmp/imo-api-prepare --stage prepare
```

## Resume and grading

The stages `all`, `rollout`, `judge` and `summary` behave as in the HLE entry.
Use the same arguments and output directory to resume; completed results are
cached. `--stage judge` grades saved rollouts without regenerating answers.
`--retry-failed` retries failed/empty rollouts or Judge errors. References are
excluded from the solving task and supplied only to the Judge.

Each run saves `dataset.json`, `manifest.json`, per-question `rollouts/` and
`grades/`, tool artifacts and `summary.json`. The manifest records the benchmark,
input/code hashes and model configuration. Changed configurations or source
require a new run directory. Old manifests created before this adaptation also
require a new directory. `summary.json` reports completion, pass@1, empty answers,
Judge errors, truncation, token usage and tool calls. Judge errors remain
explicit; they count as failures in overall pass@1.

CritPt keeps its [dedicated API workflow](critpt.md), which converts answers to
code and submits them to the official grading service. Other datasets outside
the table are not enabled by this entry.

## Validation status

All 600 rows across the four added datasets were compared against the current
GPU preparers and Judge prompts: solving prompts, references and Judge messages
match exactly. Offline tests cover answer extraction, CMT parameters/functions,
image rejection, duplicate validation, reference separation, cached grading and
HLE compatibility. No live provider/Sandbox evaluation was run for this change.
