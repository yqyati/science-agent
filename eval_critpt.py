#!/usr/bin/env python3
"""ReAct CritPt rollouts -> independent code conversion -> official batch grading."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.request

from science_agent.eval.critpt import (
    EVALUATION_URL, atomic_json, build_batch, code_prompt, extract_code,
    extract_final_answer, fingerprint, load_dataset, problem_id, solver_prompt, validate_code,
)

ROOT = Path(__file__).resolve().parent


def post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("HTTP response must be a JSON object")
    return result


def configure_environment(args) -> dict:
    settings = {
        "SCIENCE_TRAJECTORY_MAX_TOKENS": str(args.max_tokens),
        "QWEN_API_MAX_OUTPUT_TOKENS": str(args.max_tokens),
        "TASK_TIMEOUT_SEC": str(args.task_timeout),
        "MAX_LLM_CALL_PER_RUN": str(args.max_llm_calls),
        "AGENT_CONTEXT_COMPACTION": "0", "AGENT_OBSERVATION_MASKING": "0",
        "AGENT_EVAL_FORCE_FINALIZE": "0", "LLM_RETRY_INCOMPLETE_OUTPUT": "0",
        "LLM_AUTO_ADJUST_MAX_OUTPUT_TOKENS_ON_CONTEXT_ERROR": "1",
        "LLM_CONTEXT_ERROR_MIN_OUTPUT_TOKENS": "1",
        "SCIENCE_AGENT_TOOLS": "search,visit,python,google_scholar",
        "PYTHON_SESSION_BACKEND": "gateway_sandbox",
        "SCIENCE_AGENT_TOOL_HTTP_BACKEND": "gateway_sandbox",
        "SCIENCE_AGENT_SANDBOX_HTTP_STRICT": "true",
        "SCIENCE_AGENT_READ_PAGE_MODE": "file",
        "SCIENCE_AGENT_SEARCH_MAX_RESULTS": "10",
        "SCIENCE_AGENT_SEARCH_SNIPPET_CHARS": "0",
        "SCIENCE_AGENT_SCHOLAR_MAX_RESULTS": "0",
        "SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS": "0",
        "PYTHON_TOOL_RETURN_MAX_CHARS": "50000",
        "SCIENCE_AGENT_SANDBOX_TIMEOUT": "14400",
        "SCIENCE_AGENT_SANDBOX_RECOVERY": "1",
        "AGENT_EVAL_ENABLE_MODEL_IMAGE_INPUTS": "1",
    }
    for key, value in settings.items():
        os.environ[key] = value
    os.environ.setdefault("OPENAI_REQUEST_TIMEOUT", "600")
    os.environ.setdefault("SCIENCE_AGENT_SANDBOX_IMAGE", "registry.h.pjlab.org.cn/ailab-ai4sgi-ailab_ai4sgi_pool/yqy:science-agent-cpu-20261010")
    os.environ.setdefault("SERVED_MODEL_NAME_FALLBACKS", "")
    if os.environ["SERVED_MODEL_NAME_FALLBACKS"].strip():
        raise ValueError("Disable served-model fallbacks for a checkpoint comparison")
    # Record actual tool/runtime settings, excluding credentials.
    keys = set(settings) | {
        "SCIENCE_AGENT_SANDBOX_IMAGE", "SCIENCE_AGENT_SANDBOX_BASE_URL",
        "SCIENCE_AGENT_SANDBOX_RESOURCE", "SCIENCE_AGENT_SANDBOX_NETWORK_DEFAULT_ACTION",
        "OPENAI_REQUEST_TIMEOUT", "LLM_API_MAX_RETRIES", "AGENT_EVAL_ANSWER_EXTRACTION_MODE",
    }
    return {key: os.environ.get(key) for key in sorted(keys)}


def initialize_run(args, rows: list[dict], runtime: dict) -> None:
    config = {
        "dataset_sha256": fingerprint(rows), "model": args.model,
        "model_path": args.model_path or args.model, "base_url": args.base_url,
        "temperature": args.temperature, "top_p": args.top_p,
        "max_tokens": args.max_tokens, "max_context_tokens": args.max_context_tokens,
        "task_timeout": args.task_timeout, "max_llm_calls": args.max_llm_calls,
        "code_max_tokens": args.code_max_tokens, "code_attempts": args.code_attempts,
        "code_enable_thinking": False,
        "code_timeout": args.code_timeout, "eval_url": args.eval_url,
        "runtime": runtime,
        "code_sha256": fingerprint({
            name: (ROOT / name).read_text() for name in (
                "eval_critpt.py", "science_agent/eval/critpt.py",
                "inference/react_agent_openai_api_tools.py", "inference/react_agent_qwen_api.py",
                "inference/sandbox_gateway.py",
            )
        }),
    }
    manifest = args.run_dir / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous != config:
            changed = sorted(key for key in config if config[key] != previous.get(key))
            raise ValueError(f"Run configuration changed ({', '.join(changed)}); use a new run directory")
    else:
        atomic_json(manifest, config)
        atomic_json(args.run_dir / "dataset.json", rows)


def make_solver(args):
    """Load the existing agent after configuring its import-time environment."""
    sys.path.insert(0, str(ROOT / "inference"))
    from react_agent_openai_api_tools import MultiTurnReactAgent

    class BudgetExceeded(Exception):
        pass

    class CritPtAgent(MultiTurnReactAgent):
        def _extract_terminal_answer(self, text):
            return extract_final_answer(text)

        def call_server(self, messages, planning_port, **kwargs):
            self.last_messages = copy.deepcopy(messages)
            tools = kwargs.get("tool_schemas")
            remaining = min(
                args.max_tokens - self.output_tokens,
                # The serving template can add a boundary token beyond the local count.
                args.max_context_tokens - self.count_tokens(messages, tools) - 1,
            )
            if remaining <= 0:
                raise BudgetExceeded("trajectory/context budget exhausted")
            requested = kwargs.get("max_output_tokens") or self.max_output_tokens
            kwargs["max_output_tokens"] = min(requested, remaining)
            response, usage = super().call_server(messages, planning_port, **kwargs)
            completion = usage.get("completion_tokens")
            if completion is None:
                self.usage_estimated = True
                text = str(getattr(response, "content", "") or "") + str(getattr(response, "reasoning_content", "") or "")
                completion = self.count_tokens([{"role": "assistant", "content": text}])
            self.output_tokens += int(completion)
            return response, usage

    def solve(row):
        agent = CritPtAgent(function_list=["search", "visit", "python", "google_scholar"], llm={
            "model": args.model_path or args.model, "served_model_name": args.model,
            "generate_cfg": {"temperature": args.temperature, "top_p": args.top_p, "presence_penalty": 0},
        })
        agent._get_api_base = lambda: args.base_url
        agent._get_api_key = lambda: args.model_api_key
        agent.output_tokens = 0
        agent.usage_estimated = False
        agent.last_messages = []
        started = time.monotonic()
        try:
            result = agent._run({"item": {**row, "question": solver_prompt(row), "dataset_base_dir": str(args.input.parent)}})
        except BudgetExceeded:
            result = {"prediction": "", "termination": "trajectory/context budget exhausted",
                      "messages": agent.last_messages, "truncated": True, "trajectory_budget_exhausted": True}
        except Exception as exc:
            result = {"prediction": "", "termination": "solver exception", "messages": agent.last_messages,
                      "error": safe_error(exc, args)}
        result.update({"elapsed_seconds": time.monotonic() - started,
                       "accounted_completion_tokens": agent.output_tokens,
                       "completion_tokens_estimated": agent.usage_estimated})
        result["truncated"] = bool(result.get("truncated")) or result.get("termination") == "exceed available llm calls"
        return result

    return solve


def safe_error(exc: Exception, args) -> str:
    value = f"{type(exc).__name__}: {exc}"
    for secret in (args.model_api_key, args.eval_api_key):
        if secret:
            value = value.replace(secret, "<redacted>")
    return value[:2000]


def convert_answer(row: dict, answer: str, args, request=post_json) -> dict:
    attempts = []
    for index in range(args.code_attempts):
        record = {"attempt": index + 1}
        started = time.monotonic()
        try:
            response = request(args.base_url.rstrip("/") + "/chat/completions", {
                "model": args.model, "messages": [{"role": "user", "content": code_prompt(row, answer)}],
                "temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.code_max_tokens,
                # Formatting needs the final code, not another reasoning transcript.
                "chat_template_kwargs": {"enable_thinking": False},
            }, {"Authorization": "Bearer " + args.model_api_key}, args.code_timeout)
            choice = response["choices"][0]
            record.update({"finish_reason": choice.get("finish_reason"), "usage": response.get("usage"),
                           "message": choice["message"]})
            if choice.get("finish_reason") != "stop":
                raise ValueError("Code conversion did not finish with stop")
            code = extract_code(choice["message"].get("content") or "")
            validate_code(code, row["code_template"])
        except Exception as exc:
            record["error"] = safe_error(exc, args)
        else:
            record["elapsed_seconds"] = time.monotonic() - started
            attempts.append(record)
            return {"status": "completed", "generated_code": code, "attempts": attempts,
                    "answer_sha256": fingerprint(answer)}
        record["elapsed_seconds"] = time.monotonic() - started
        attempts.append(record)
    return {"status": "failed", "attempts": attempts, "answer_sha256": fingerprint(answer)}


def parallel_stage(rows, args, stage, worker):
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, row): row["id"] for row in rows}
        for future in as_completed(futures):
            pid = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                raise RuntimeError(f"{stage} artifact failure for {pid}") from exc
            print(f"[{stage}] {pid}: {status}", flush=True)


def run_rollouts(rows, args, solve):
    def worker(row):
        path = args.run_dir / "rollouts" / f"{problem_id(row['id'])}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["status"] == "completed" or not args.retry_failed:
                return "saved " + previous["status"]
        try:
            result = solve(row)
        except Exception as exc:
            result = {"prediction": "", "termination": "solver exception", "error": safe_error(exc, args)}
        prediction = str(result.get("prediction") or "").strip()
        status = "completed" if prediction and not result.get("error") else "failed"
        atomic_json(path, {**result, "id": row["id"], "prediction": prediction, "status": status})
        return status
    parallel_stage(rows, args, "rollout", worker)


def run_conversions(rows, args, request=post_json):
    def worker(row):
        pid = problem_id(row["id"])
        rollout_path = args.run_dir / "rollouts" / f"{pid}.json"
        if not rollout_path.exists():
            return "missing rollout"
        rollout = json.loads(rollout_path.read_text())
        answer = rollout.get("prediction", "")
        if rollout["status"] != "completed" or not answer.strip():
            return "no final answer; not converted"
        path = args.run_dir / "conversions" / f"{pid}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            same = previous.get("answer_sha256") == fingerprint(answer)
            if same and (previous["status"] == "completed" or not args.retry_failed):
                return "saved " + previous["status"]
        result = convert_answer(row, answer, args, request)
        atomic_json(path, {**result, "id": row["id"]})
        return result["status"]
    parallel_stage(rows, args, "conversion", worker)


def summarize(rows, args):
    summary = {"selected": len(rows), "rollouts_completed": 0, "no_final_answer": 0,
               "solver_failed": 0, "truncated": 0, "conversions_completed": 0, "conversions_failed": 0,
               "solver_completion_tokens": 0, "conversion_completion_tokens": 0,
               "solver_usage_missing": 0, "conversion_usage_missing": 0,
               "tool_calls": 0, "llm_turns": 0}
    for row in rows:
        pid = problem_id(row["id"])
        path = args.run_dir / "rollouts" / f"{pid}.json"
        if path.exists():
            result = json.loads(path.read_text())
            summary["rollouts_completed"] += result["status"] == "completed"
            summary["solver_failed"] += result["status"] == "failed"
            summary["no_final_answer"] += not bool(result.get("prediction", "").strip())
            summary["truncated"] += bool(result.get("truncated"))
            tokens = result.get("completion_tokens")
            summary["solver_usage_missing"] += tokens is None
            summary["solver_completion_tokens"] += tokens or 0
            for message in result.get("messages", []):
                if message.get("role") == "assistant":
                    summary["llm_turns"] += 1
                    summary["tool_calls"] += len(message.get("tool_calls") or [])
        path = args.run_dir / "conversions" / f"{pid}.json"
        if path.exists():
            result = json.loads(path.read_text())
            summary["conversions_completed"] += result["status"] == "completed"
            summary["conversions_failed"] += result["status"] == "failed"
            for attempt in result["attempts"]:
                tokens = (attempt.get("usage") or {}).get("completion_tokens")
                summary["conversion_usage_missing"] += tokens is None
                summary["conversion_completion_tokens"] += tokens or 0
    atomic_json(args.run_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def submit(rows, args, request=post_json):
    # Validate all artifacts before touching the official service or consuming quota.
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    config = {
        "use_golden_for_prev_steps": False, "parsing": False, "multiturn_with_answer": False,
        "use_python": True, "use_web_search": True, "harness": "science-agent-sandbox",
        "solver": manifest, "format_conversion": {"model": args.model, "max_tokens": args.code_max_tokens},
    }
    payload = build_batch(rows, args.run_dir, args.model, config)
    result_path = args.run_dir / "evaluation.json"
    if result_path.exists() and not args.resubmit:
        print("Official result already saved; not submitting again", flush=True)
        return
    attempt_path = args.run_dir / "submission_attempt.json"
    if attempt_path.exists() and not args.resubmit:
        raise ValueError("A grading submission was already attempted; inspect submission_attempt.json before using --resubmit")
    atomic_json(args.run_dir / "submission.json", payload)
    atomic_json(attempt_path, {"status": "started", "timestamp": time.time(), "payload_sha256": fingerprint(payload)})
    try:
        result = request(args.eval_url, payload, {"x-api-key": args.eval_api_key}, args.eval_timeout)
        accuracy = result.get("accuracy")
        if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Official response has no valid accuracy")
        atomic_json(result_path, result)
    except Exception as exc:
        atomic_json(attempt_path, {"status": "failed_or_unconfirmed", "error": safe_error(exc, args)})
        raise
    atomic_json(attempt_path, {"status": "completed", "timestamp": time.time()})
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=("prepare", "rollout", "convert", "submit", "all"), default="prepare")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Exact served checkpoint/model name")
    parser.add_argument("--model-path", default="", help="Local tokenizer/model directory, if available")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_API_BASE") or os.getenv("API_BASE", ""))
    parser.add_argument("--model-api-key", default=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", ""))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=70, help="Deterministic prefix for smoke; official submission requires 70")
    parser.add_argument("--max-tokens", type=int, default=122880)
    parser.add_argument("--max-context-tokens", type=int, default=122880)
    parser.add_argument("--max-llm-calls", type=int, default=100)
    parser.add_argument("--task-timeout", type=float, default=9000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--code-max-tokens", type=int, default=8192)
    parser.add_argument("--code-attempts", type=int, default=3)
    parser.add_argument("--code-timeout", type=float, default=600)
    parser.add_argument("--eval-url", default=os.getenv("CRITPT_EVAL_SERVER_URL", EVALUATION_URL))
    parser.add_argument("--eval-api-key", default=os.getenv("CRITPT_EVAL_API_KEY", ""))
    parser.add_argument("--eval-timeout", type=float, default=7200)
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly retry failed artifacts in the selected stage")
    parser.add_argument("--resubmit", action="store_true", help="Explicitly send another full grading request")
    args = parser.parse_args()
    for key in ("workers", "max_tokens", "max_context_tokens", "max_llm_calls", "task_timeout", "code_max_tokens", "code_attempts", "code_timeout", "eval_timeout"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not 1 <= args.limit <= 70:
        parser.error("--limit must be between 1 and 70")
    if not math.isfinite(args.temperature) or args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be finite and nonnegative; top-p must be in (0, 1]")
    if args.step in {"rollout", "convert", "all"} and not (args.base_url and args.model_api_key):
        parser.error("Model base URL and API key are required")
    if args.step in {"submit", "all"} and (args.limit != 70 or not args.eval_api_key):
        parser.error("Official submission requires --limit 70 and CRITPT_EVAL_API_KEY")
    rows = load_dataset(args.input)
    runtime = configure_environment(args)
    initialize_run(args, rows, runtime)
    selected = rows[:args.limit]
    try:
        if args.step in {"rollout", "all"}:
            if not os.getenv("OPEN_SANDBOX_GATEWAY_TOKEN"):
                parser.error("Launch through run_sandbox_harness.sh with a Sandbox token")
            run_rollouts(selected, args, make_solver(args))
        if args.step in {"convert", "all"}:
            run_conversions(selected, args)
        if args.step in {"submit", "all"}:
            submit(rows, args)
    finally:
        summary = summarize(selected, args)
    if args.step == "rollout" and summary["rollouts_completed"] != len(selected):
        raise SystemExit("Some rollouts failed; records were saved without rerunning the solver")
    if args.step == "convert" and summary["conversions_completed"] != len(selected):
        raise SystemExit("Some answers could not be converted; inspect rollouts/conversions and summary.json")


if __name__ == "__main__":
    main()
