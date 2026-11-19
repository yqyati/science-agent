"""Live API + Python Sandbox smoke for the separate progress-verifier harness."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "inference"))

QUESTION = """This is a multi-turn integration check. Determine A = sum of the integers
from 1 to 100 inclusive, and B = sum of their squares. Use four separate Python
tool calls, one per turn: (1) directly sum the integers; (2) directly sum the
squares; (3) independently check A with its closed-form formula; (4) independently
check B with its closed-form formula. Before the first call, derive and state the
closed-form formula for A as a <progress> declaration with justification, then
issue the first tool call in that SAME turn. After each result report any newly
established substantive progress before the next tool call. Follow checkpoint
feedback if it arrives, but do not wait for it. Finish after all four computations,
with <answer> containing A and B. Do not use search or external sources."""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--solver-base-url", default=os.getenv("PROGRESS_SMOKE_SOLVER_BASE_URL", os.getenv("API_BASE", "")))
    parser.add_argument("--solver-model", default=os.getenv("PROGRESS_SMOKE_SOLVER_MODEL", os.getenv("SUMMARY_MODEL_NAME", "")))
    parser.add_argument("--solver-key-env", default="API_KEY")
    parser.add_argument("--solver-extra-body", type=json.loads, default={})
    args = parser.parse_args(argv)
    for name, value in [("solver-base-url", args.solver_base_url), ("solver-model", args.solver_model),
                        (args.solver_key_env, os.getenv(args.solver_key_env)),
                        ("PROGRESS_VERIFIER_BASE_URL", os.getenv("PROGRESS_VERIFIER_BASE_URL")),
                        ("PROGRESS_VERIFIER_MODEL", os.getenv("PROGRESS_VERIFIER_MODEL"))]:
        if not value:
            parser.error(f"Missing {name}; load tool credentials through run_sandbox_harness.sh")
    if args.run_dir.exists():
        parser.error("Use a new run directory; smoke evidence must not mix with an earlier run")
    args.run_dir.mkdir(parents=True)
    os.environ.update({
        "SCIENCE_AGENT_PROGRESS_VERIFIER": "1",
        "PROGRESS_VERIFIER_LOG_DIR": str(args.run_dir / "verifier"),
        "SCIENCE_AGENT_TRACE_FILE": str(args.run_dir / "trace.jsonl"),
        "SCIENCE_AGENT_TRACE": "1",
        "SCIENCE_AGENT_TOOLS": "python",
        "SCIENCE_TRAJECTORY_MAX_TOKENS": "32768", "QWEN_API_MAX_OUTPUT_TOKENS": "4096",
        "MAX_LLM_CALL_PER_RUN": "10", "TASK_TIMEOUT_SEC": "480",
        "OPENAI_REQUEST_TIMEOUT": "180", "LLM_API_MAX_RETRIES": "1",
        "AGENT_CONTEXT_COMPACTION": "0", "AGENT_OBSERVATION_MASKING": "0",
        "AGENT_EVAL_FORCE_FINALIZE": "0", "SERVED_MODEL_NAME_FALLBACKS": "",
        "SCIENCE_AGENT_DEEPSEEK_V4_REASONING": "0",
        "PYTHON_SESSION_BACKEND": "gateway_sandbox",
        "SCIENCE_AGENT_TOOL_HTTP_BACKEND": "gateway_sandbox",
    })
    print("Loading the real harness runtime...", flush=True)
    from react_agent_openai_api_tools_progress import MultiTurnReactAgent
    from science_agent.tracing import trace_context

    requests = []
    class SmokeAgent(MultiTurnReactAgent):
        def _get_api_base(self):
            return args.solver_base_url
        def _get_api_key(self):
            return os.environ[args.solver_key_env]
        def _deepseek_v4_profile_model_name(self, model_name):
            return ""
        def call_server(self, messages, planning_port, **kwargs):
            requests.append(copy.deepcopy(messages))
            print(f"Policy request {len(requests)}: {len(messages)} messages", flush=True)
            extra = copy.deepcopy(args.solver_extra_body)
            extra.update(kwargs.get("request_extra_body") or {})
            kwargs["request_extra_body"] = extra
            result = super().call_server(messages, planning_port, **kwargs)
            print(f"Policy response {len(requests)} received", flush=True)
            return result

    agent = SmokeAgent(function_list=["python"], llm={
        "model": args.solver_model, "served_model_name": args.solver_model,
        "generate_cfg": {"temperature": 1.0, "top_p": 0.95, "max_input_tokens": 32768},
    })
    # This API smoke needs no local model/tokenizer downloads.
    agent._context_tokenizer_unavailable = True
    report = {"passed": False, "solver_model": args.solver_model,
              "verifier_model": os.environ["PROGRESS_VERIFIER_MODEL"],
              "question_sha256": hashlib.sha256(QUESTION.encode()).hexdigest()}
    started = time.monotonic()
    try:
        with trace_context(run_id=args.run_dir.name, task_id="progress-smoke"):
            rollout = agent._run({"item": {"question": QUESTION, "id": "progress-smoke"}})
        (args.run_dir / "rollout.json").write_text(json.dumps(rollout, ensure_ascii=False, indent=2))
        (args.run_dir / "requests.json").write_text(json.dumps(requests, ensure_ascii=False, indent=2))
        events = [json.loads(line) for path in (args.run_dir / "verifier").glob("*.jsonl")
                  for line in path.read_text().splitlines()]
        feedback_requests = sum(any(m.get("role") == "user" and isinstance(m.get("content"), str)
                                    and m["content"].startswith("<checkpoint_feedback>")
                                    for m in request) for request in requests)
        tool_outputs = "\n".join(str(m.get("content", "")) for m in rollout["messages"] if m.get("role") == "tool")
        completed = {e["checkpoint_id"]: e["time"] for e in events
                     if e["event"] == "progress_verification_completed"}
        trace_events = [json.loads(line) for line in (args.run_dir / "trace.jsonl").read_text().splitlines()]
        tool_events = [e for e in trace_events if e["event"] == "model_tool_call_start"]
        submitted = [e for e in events if e["event"] == "progress_checkpoint_submitted"]
        checks = {
            "answer_correct": all(v in rollout.get("prediction", "") for v in ("5050", "338350")),
            "python_tools_executed": sum(m.get("role") == "tool" for m in rollout["messages"]) >= 4,
            "python_outputs_correct": all(v in tool_outputs for v in ("5050", "338350")),
            "checkpoint_submitted": any(e["event"] == "progress_checkpoint_submitted" for e in events),
            "verifier_valid_response": any(e["event"] == "progress_verification_completed"
                                            and e["result"]["verdict"] in {"pass", "fail", "uncertain"} for e in events),
            "feedback_in_later_model_request": (
                feedback_requests > 0 if any(e["event"] == "progress_feedback_delivered" for e in events)
                else any(e["event"] == "progress_feedback_suppressed" for e in events)),
            "passed_checks_silent": not any(
                record["result"]["verdict"] == "pass" and record["delivered"]
                for record in rollout.get("progress_verifier", {}).get("checkpoints", [])
                if "result" in record),
            "tools_ran_while_verifier_pending": any(
                start["time"] <= tool["time"] < completed.get(start["record"]["checkpoint_id"], float("inf"))
                for start in submitted for tool in tool_events),
            "audit_saved": rollout.get("progress_verifier", {}).get("audit_write_errors") == 0,
        }
        report.update(checks=checks, passed=all(checks.values()), model_requests=len(requests),
                      feedback_requests=feedback_requests, termination=rollout.get("termination"))
    except Exception as exc:
        # Avoid dumping API request headers or endpoint error bodies with secrets.
        report.update(passed=False, error_type=type(exc).__name__)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (args.run_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
