import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_critpt as runner
from science_agent.eval import critpt


def dataset():
    return [dict(id=f"CritPt_challenge_{n:02d}", context="A physical system with parameter p.",
                 question="Return the two coefficients.", modality="text-only",
                 code_template="def answer(p):\n    coeffs = ...\n    return coeffs\n") for n in range(1, 71)]


def arguments(path):
    return SimpleNamespace(
        run_dir=Path(path), model="checkpoint-step100", model_path="", base_url="http://model/v1",
        model_api_key="model-secret", eval_api_key="eval-secret", eval_url=critpt.EVALUATION_URL,
        workers=2, temperature=0.6, top_p=1.0, max_tokens=122880, max_context_tokens=122880,
        max_llm_calls=100, task_timeout=9000, code_max_tokens=8192, code_attempts=2,
        code_timeout=600, eval_timeout=7200, retry_failed=False, resubmit=False,
    )


def code_response(code="def answer(p):\n    return [1, 2]", finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": code}}],
            "usage": {"completion_tokens": 12}}


class CritPtTests(unittest.TestCase):
    def setUp(self):
        stdout = patch("sys.stdout", io.StringIO())
        stdout.start()
        self.addCleanup(stdout.stop)

    def test_dataset_requires_complete_unique_ids_and_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            def write(rows):
                path.write_text("\n".join(json.dumps(row) for row in rows))
            rows = dataset()
            write(rows)
            self.assertEqual(len(critpt.load_dataset(path)), 70)
            for bad in (rows[:-1], rows[:-1] + [rows[0]], [{**row, "modality": "image"} for row in rows]):
                write(bad)
                with self.assertRaises(ValueError):
                    critpt.load_dataset(path)

    def test_entire_final_region_and_multiple_boxes_are_preserved(self):
        self.assertEqual(critpt.extract_final_answer("<think>draft</think><answer>\\boxed{1}\n\\boxed{2}</answer>"), "\\boxed{1}\n\\boxed{2}")
        self.assertEqual(critpt.extract_final_answer("<think>\\boxed{99}</think>Final Answer:\n\\boxed{1}\n\\boxed{2}"), "\\boxed{1}\n\\boxed{2}")
        self.assertEqual(critpt.extract_final_answer("<think>\\boxed{99}</think>\\boxed{1} and \\boxed{2}"), "\\boxed{1}\n\\boxed{2}")
        self.assertEqual(critpt.extract_final_answer("still reasoning without a conclusion"), "")
        self.assertEqual(critpt.extract_final_answer("<think><answer>draft</answer></think>still reasoning"), "")

    def test_prompt_includes_background_question_and_output_contract(self):
        row = dataset()[0]
        text = critpt.solver_prompt(row)
        for part in (row["context"], row["question"], row["code_template"]):
            self.assertIn(part, text)
        self.assertNotIn("reference_answer", text)

    def test_code_validation_checks_signature_placeholders_and_imports_without_execution(self):
        template = "import sympy as sp\ndef answer(p):\n    return ...\n"
        critpt.validate_code("import sympy as sp\ndef answer(p):\n    return 1-p**2", template)
        for bad in ("def answer(q): return 1", "def answer(p): return ...", "def answer(p): pass",
                    "import os\ndef answer(p): return 1", "def answer(p) -> int: return 1",
                    "def answer(p): return 1\ndef answer(p): return 2"):
            with self.subTest(code=bad), self.assertRaises(ValueError):
                critpt.validate_code(bad, template)
        self.assertEqual(critpt.extract_code("notes\n```python\ndef answer(p): return 1\n```"), "def answer(p): return 1")
        with self.assertRaises(ValueError):
            critpt.extract_code("```python\nx=1\n```\n```python\ny=2\n```")

    def test_conversion_retries_only_formatting_and_keeps_usage_from_failed_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory)
            request = Mock(side_effect=[code_response(finish="length"), code_response()])
            result = runner.convert_answer(dataset()[0], "coefficients 1 and 2", args, request)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["attempts"]), 2)
            self.assertEqual(request.call_count, 2)
            for call in request.call_args_list:
                payload = call.args[1]
                self.assertEqual(payload["model"], args.model)
                self.assertEqual(payload["max_tokens"], 8192)
                self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
                self.assertIn("coefficients 1 and 2", payload["messages"][0]["content"])
                self.assertNotIn(dataset()[0]["context"], payload["messages"][0]["content"])
            self.assertEqual(sum(a["usage"]["completion_tokens"] for a in result["attempts"]), 24)

    def test_resume_keeps_rollout_immutable_while_retrying_failed_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows = arguments(directory), dataset()[:1]
            solve = Mock(return_value={"prediction": "1, 2", "messages": [], "completion_tokens": 20})
            runner.run_rollouts(rows, args, solve)
            request = Mock(return_value=code_response("def answer(p): return ..."))
            runner.run_conversions(rows, args, request)
            self.assertEqual(request.call_count, 2)
            rollout = args.run_dir / "rollouts" / "Challenge_1_main.json"
            original = rollout.read_bytes()
            args.retry_failed = True
            runner.run_rollouts(rows, args, solve)
            request = Mock(return_value=code_response())
            runner.run_conversions(rows, args, request)
            self.assertEqual(solve.call_count, 1)
            self.assertEqual(rollout.read_bytes(), original)
            self.assertEqual(request.call_count, 1)

    def test_no_final_answer_is_saved_and_never_sent_to_converter(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows = arguments(directory), dataset()[:1]
            runner.run_rollouts(rows, args, Mock(return_value={"prediction": "", "truncated": True}))
            request = Mock()
            runner.run_conversions(rows, args, request)
            request.assert_not_called()
            runner.summarize(rows, args)
            summary = json.loads((args.run_dir / "summary.json").read_text())
            self.assertEqual(summary["no_final_answer"], 1)
            self.assertEqual(summary["truncated"], 1)

    def test_manifest_refuses_changed_dataset_model_or_conversion_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows = arguments(directory), dataset()
            runner.initialize_run(args, rows, {})
            runner.initialize_run(args, rows, {})
            self.assertNotIn(args.model_api_key, (args.run_dir / "manifest.json").read_text())
            changed = copy.deepcopy(rows)
            changed[0]["question"] += " modified"
            with self.assertRaisesRegex(ValueError, "dataset_sha256"):
                runner.initialize_run(args, changed, {})
            args.code_max_tokens += 1
            with self.assertRaisesRegex(ValueError, "code_max_tokens"):
                runner.initialize_run(args, rows, {})

    def test_full_pipeline_submits_exactly_70_and_does_not_resubmit_saved_result(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows = arguments(directory), dataset()
            runner.initialize_run(args, rows, {})
            solve = Mock(return_value={"prediction": "1, 2", "messages": [], "completion_tokens": 20})
            runner.run_rollouts(rows, args, solve)
            conversion = Mock(return_value=code_response())
            runner.run_conversions(rows, args, conversion)
            official = Mock(return_value={"accuracy": 0.5, "timeout_rate": 0, "judge_error_count": 0})
            runner.submit(rows, args, official)
            payload = official.call_args.args[1]
            self.assertEqual(len(payload["submissions"]), 70)
            self.assertEqual({s["problem_id"] for s in payload["submissions"]}, critpt.PROBLEM_IDS)
            self.assertTrue(payload["submissions"][0]["generation_config"]["use_python"])
            self.assertNotIn(args.model_api_key, json.dumps(payload))
            self.assertNotIn(args.eval_api_key, json.dumps(payload))
            runner.submit(rows, args, official)
            self.assertEqual(official.call_count, 1)
            runner.summarize(rows, args)
            summary = json.loads((args.run_dir / "summary.json").read_text())
            self.assertEqual(summary["solver_completion_tokens"], 1400)
            self.assertEqual(summary["conversion_completion_tokens"], 840)
            self.assertEqual(solve.call_count, 70)

    def test_batch_rejects_missing_failed_duplicate_and_stale_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows = arguments(directory), dataset()
            with self.assertRaises(FileNotFoundError):
                critpt.build_batch(rows, args.run_dir, args.model, {})
            runner.run_rollouts(rows, args, Mock(return_value={"prediction": "1, 2"}))
            runner.run_conversions(rows, args, Mock(return_value=code_response()))
            with self.assertRaisesRegex(ValueError, "70"):
                critpt.build_batch(rows[:-1] + [rows[0]], args.run_dir, args.model, {})
            path = args.run_dir / "rollouts" / "Challenge_1_main.json"
            critpt.atomic_json(path, {"status": "completed", "prediction": "different"})
            with self.assertRaisesRegex(ValueError, "different final answer"):
                critpt.build_batch(rows, args.run_dir, args.model, {})
            critpt.atomic_json(path, {"status": "failed", "prediction": ""})
            with self.assertRaisesRegex(ValueError, "no final answer"):
                critpt.build_batch(rows, args.run_dir, args.model, {})

    def test_submit_rejects_incomplete_batch_before_http_or_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows = arguments(directory), dataset()
            runner.initialize_run(args, rows, {})
            request = Mock()
            with self.assertRaises(FileNotFoundError):
                runner.submit(rows, args, request)
            request.assert_not_called()
            self.assertFalse((args.run_dir / "submission_attempt.json").exists())

    def test_uncertain_submission_is_not_automatically_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rows = arguments(directory), dataset()
            runner.initialize_run(args, rows, {})
            payload = {"submissions": [{"problem_id": "unused"}], "batch_metadata": {}}
            request = Mock(side_effect=TimeoutError("result lost"))
            with patch.object(runner, "build_batch", return_value=payload):
                with self.assertRaises(TimeoutError):
                    runner.submit(rows, args, request)
                with self.assertRaisesRegex(ValueError, "already attempted"):
                    runner.submit(rows, args, request)
            self.assertEqual(request.call_count, 1)
            self.assertFalse((args.run_dir / "evaluation.json").exists())

    def test_solver_budget_and_complete_extraction_use_existing_agent_loop(self):
        calls = []
        class Base:
            def __init__(self, **kwargs): self.max_output_tokens = 100
            def count_tokens(self, messages, tools=None): return 5
            def call_server(self, messages, planning_port, **kwargs):
                calls.append(kwargs["max_output_tokens"])
                return SimpleNamespace(content="<answer>\\boxed{1}\n\\boxed{2}</answer>"), {"completion_tokens": 6}
            def _run(self, data):
                for _ in range(3):
                    response, usage = self.call_server([{"role": "user", "content": "q"}], None)
                return {"prediction": self._extract_terminal_answer(response.content)}
        module = SimpleNamespace(MultiTurnReactAgent=Base)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"react_agent_openai_api_tools": module}):
            args = arguments(directory)
            args.input = Path(directory) / "data.jsonl"
            args.max_tokens = 10
            result = runner.make_solver(args)(dataset()[0])
            self.assertEqual(calls, [10, 4])
            self.assertTrue(result["trajectory_budget_exhausted"])
            self.assertEqual(result["prediction"], "")
            args.max_tokens = 100
            result = runner.make_solver(args)(dataset()[0])
            self.assertEqual(result["prediction"], "\\boxed{1}\n\\boxed{2}")

    def test_solver_reserves_boundary_token_for_reported_context_overflows(self):
        calls = []
        class Base:
            def __init__(self, **kwargs): self.max_output_tokens = 122880
            def count_tokens(self, messages, tools=None): return local_input
            def call_server(self, messages, planning_port, **kwargs):
                requested = kwargs["max_output_tokens"]
                if requested + backend_input > 122880:
                    raise RuntimeError("backend rejects context overflow")
                calls.append(requested)
                return SimpleNamespace(content="<answer>1</answer>"), {"completion_tokens": 1}
            def _run(self, data):
                response, _ = self.call_server([{"role": "user", "content": "q"}], None)
                return {"prediction": self._extract_terminal_answer(response.content)}
        module = SimpleNamespace(MultiTurnReactAgent=Base)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"react_agent_openai_api_tools": module}):
            args = arguments(directory)
            args.input = Path(directory) / "data.jsonl"
            for backend_input in (16821, 21935, 5036):
                local_input = backend_input - 1
                with self.subTest(backend_input=backend_input):
                    result = runner.make_solver(args)(dataset()[0])
                    self.assertEqual(result["prediction"], "1")
                    self.assertEqual(calls[-1], 122880 - backend_input)
            local_input = 122879
            result = runner.make_solver(args)(dataset()[0])
            self.assertTrue(result["trajectory_budget_exhausted"])
            self.assertEqual(len(calls), 3)
            with patch.dict("os.environ"):
                runtime = runner.configure_environment(args)
                self.assertEqual(runtime["LLM_AUTO_ADJUST_MAX_OUTPUT_TOKENS_ON_CONTEXT_ERROR"], "1")
                self.assertEqual(runtime["LLM_CONTEXT_ERROR_MIN_OUTPUT_TOKENS"], "1")

    @unittest.skipUnless(importlib.util.find_spec("qwen_agent"), "Requires the normal harness qwen-agent runtime")
    def test_real_react_loop_keeps_python_observation_and_all_final_results(self):
        sys.path.insert(0, str(runner.ROOT / "inference"))
        import react_agent_openai_api_tools as tools_module
        import react_agent_qwen_api as base
        call = SimpleNamespace(type="function", id="python-1", function=SimpleNamespace(
            name="PythonInterpreter", arguments='{"code": "print(1, 2)"}'))
        responses = [
            (SimpleNamespace(content="Check the coefficients.", reasoning_content="", tool_calls=[call]), {"completion_tokens": 10}),
            (SimpleNamespace(content="<answer>\\boxed{1}\n\\boxed{2}</answer>", reasoning_content="", tool_calls=None), {"completion_tokens": 10}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            args = arguments(directory)
            args.input = Path(directory) / "data.jsonl"
            with patch.dict("os.environ", {"SCIENCE_AGENT_TOOLS": "search,visit,python,google_scholar",
                                           "SERVED_MODEL_NAME_FALLBACKS": ""}), \
                    patch.object(base.MultiTurnReactAgent, "count_tokens", return_value=32), \
                    patch.object(base.MultiTurnReactAgent, "call_server", side_effect=responses) as model, \
                    patch.object(tools_module, "call_runtime_tool", return_value="1 2\n") as python:
                result = runner.make_solver(args)(dataset()[0])
            self.assertEqual(result["prediction"], "\\boxed{1}\n\\boxed{2}")
            self.assertEqual(result["termination"], "answer")
            self.assertEqual(result["accounted_completion_tokens"], 20)
            self.assertEqual(model.call_count, 2)
            python.assert_called_once()
            self.assertEqual([m["role"] for m in result["messages"]], ["system", "user", "assistant", "tool", "assistant"])
            self.assertEqual(result["messages"][3]["content"], "1 2\n")


if __name__ == "__main__":
    unittest.main()
