# CritPt evaluation with the Sandbox ReAct harness

`eval_critpt.py` reuses the current `MultiTurnReactAgent` and its Sandbox tools.
It does not invoke the HLE/o3-mini judge. The public input must contain exactly
the 70 unique CritPt challenge IDs, their backgrounds, questions and answer
templates. The current adapter supports the text-only public set.

The stages are independent:

1. `prepare`: validate the input and record the dataset, inference settings and
   hashes of the relevant working code. No API calls.
2. `rollout`: solve using Search, Scholar, saved webpages and persistent Python.
   Preserve the whole final answer region, including multiple results. Save
   messages, usage, termination and truncation for every attempted problem.
3. `convert`: use the same served model to place the immutable final answer into
   its `code_template`. This request receives the template and final answer,
   not the original question or reference answer. Record every conversion
   attempt, including usage from unsuccessful responses.
4. `submit`: validate the full batch and send it to the official CritPt service.
   Save the service response, including accuracy, timeout and judge-error fields.

`--step all` performs rollout, conversion and submission, rather than skipping
generation. It requires a grading key before starting. The default step is
`prepare`; no step other than `submit` or `all` calls the grading service.

## Configuration

Defaults match the current HLE360 sampling/tool configuration: temperature 0.6,
top-p 1, 16 workers, 122880 generated tokens per trajectory, a 122880-token
context cap, 100 LLM calls, and 9000 seconds per problem. Compaction, observation
masking, forced finalization and retries of length-truncated responses are off.
Search returns 10 results without snippet truncation; Scholar is not additionally
limited. Python returns up to 50000 characters. Each tool Sandbox uses the
configured CPU image and has a 14400-second lifetime with recovery enabled.

The adapter caps each solver request by the remaining trajectory and context
budgets, reserving one token for server-side template boundaries. Context-length
errors activate the existing backend-informed output-budget adjustment, including
further reduction when the reported input count is only a lower bound. The minimum
output budget is one token; an input that cannot fit still fails without compaction.
API completion usage accounts for generated tokens; if usage is absent,
the agent token estimator is used and the rollout explicitly records this.
Without a local tokenizer, context counting is approximate; the serving backend
must enforce its actual context limit. Use `--model-path` for the checkpoint's
tokenizer when available. Request/response transport retries can consume usage
not reported by the provider; reported usage is not a billing audit.

Formatting has a **separate** 8192-token request budget and at most three
attempts, with the same model, temperature and top-p. Formatting requests disable
Qwen thinking through `chat_template_kwargs.enable_thinking=false`, so the saved
answer is returned as code rather than a transcript containing competing drafts.
A formatting failure never
automatically reruns the solver. Code is checked for syntax, the answer function
signature, placeholders and imports absent from the template. Generated code is
never executed on the host. These checks do not prove numerical correctness or
that code is safe to execute elsewhere; the official grader runs its own checks.

## Usage

Serve the desired HF checkpoint through an OpenAI-compatible endpoint first.
The runner does not allocate GPUs, submit rjobs or start a model server.
Install the harness's usual qwen-agent dependencies in that runtime. Launch
through `run_sandbox_harness.sh` to load the ignored local tool configuration.
Model requests use `--base-url` plus `OPENAI_API_KEY` (or `API_KEY`); explicitly
choose the served checkpoint with `--model`.

Example, after setting `CRITPT_INPUT`, `CRITPT_RUN_DIR`, `CRITPT_MODEL` and
`CRITPT_MODEL_URL` in your shell:

```bash
bash run_sandbox_harness.sh python3 sandbox_smoke.py --search --page-file

bash run_sandbox_harness.sh python3 eval_critpt.py \
  --step rollout --input "$CRITPT_INPUT" --run-dir "$CRITPT_RUN_DIR" \
  --model "$CRITPT_MODEL" --base-url "$CRITPT_MODEL_URL" --limit 2

bash run_sandbox_harness.sh python3 eval_critpt.py \
  --step convert --input "$CRITPT_INPUT" --run-dir "$CRITPT_RUN_DIR" \
  --model "$CRITPT_MODEL" --base-url "$CRITPT_MODEL_URL" --limit 2
```

For the full evaluation, run those stages with `--limit 70`. Prefix smoke uses
the same full input and manifest, so completed smoke artifacts can be reused.
Keep the other settings unchanged when resuming. A changed dataset, model,
budget, endpoint or relevant working code requires a new run directory.

Then, with an approved `CRITPT_EVAL_API_KEY` configured locally:

```bash
bash run_sandbox_harness.sh python3 eval_critpt.py \
  --step submit --input "$CRITPT_INPUT" --run-dir "$CRITPT_RUN_DIR" \
  --model "$CRITPT_MODEL" --base-url "$CRITPT_MODEL_URL"
```

The default grading endpoint is
`https://artificialanalysis.ai/api/v2/critpt/evaluate`. `--eval-url` or
`CRITPT_EVAL_SERVER_URL` may select a compatible proxy. The official service
requires approved grading access and accepts only the full 70-problem batch;
its default quota is 10 batches per 24 hours. Public results contain aggregate
metrics, not necessarily individual correctness labels.

## Failures and resume

Completed artifacts are reused. Failed artifacts are also retained and are not
retried on resume unless `--retry-failed` is supplied for that stage. An empty
final answer remains a failed rollout and is never handed to the formatter to
invent an answer. Rollout/conversion commands exit nonzero if selected problems
remain incomplete.

This first implementation refuses official submission when any final answer
or valid conversion is missing. It does not invent a placeholder for unsolved
problems or report accuracy on the successfully converted subset. A canonical
representation of unsolved submissions would need validation against the
official grading protocol before enabling that behavior.

Submission attempts are recorded before HTTP dispatch. A lost response is not
automatically replayed: inspect `submission_attempt.json`, then explicitly use
`--resubmit` if another grading request is intended. A saved `evaluation.json`
is reused unless that flag is supplied.

Artifacts: `manifest.json`, `dataset.json`, `rollouts/`, `conversions/`,
`summary.json`, `submission.json`, `submission_attempt.json`, `evaluation.json`.
Summary separates solver/conversion token usage, absent usage, empty answers,
truncation, conversion failures, tool calls and LLM turns. Credentials are not
included in the manifest or grading payload.

Protocol references:
[CritPt paper §3](https://arxiv.org/html/2509.26574#S3) and
[Artificial Analysis API](https://artificialanalysis.ai/documentation#critpt-api).
