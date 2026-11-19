"""Checkpoint evidence boundaries, delayed feedback, and rollout lifecycle."""

import ast
import copy
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inference"))
import progress_verifier as pv


def verdict(value="pass"):
    return dict(verdict=value, progress="substantive", issue="", evidence="shown computation", suggested_check="")


class ProgressVerifierTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        env = patch.dict(os.environ, {"PROGRESS_VERIFIER_LOG_DIR": directory.name,
                                    "PROGRESS_VERIFIER_CONCURRENCY": "2",
                                    "PROGRESS_VERIFIER_MAX_CHECKPOINTS": "8"})
        env.start()
        self.addCleanup(env.stop)
        self.question = {"role": "user", "content": [
            {"type": "text", "text": "Compute the area."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}}]}
        self.messages = [{"role": "system", "content": "solve"}, self.question]

    def manager(self, verify):
        manager = pv.ProgressVerifier(self.question, verify=verify)
        self.addCleanup(manager.close)
        return manager

    def checkpoint(self, text="<progress>area is 4</progress>", **fields):
        self.messages.append({"role": "assistant", "content": text, **fields})
        return text

    def complete(self, manager):
        # Waiting is restricted to tests, never the production rollout.
        for future in list(manager.pending.values()):
            future.result(timeout=2)

    def test_nonblocking_with_pending_checks_and_exact_evidence_boundaries(self):
        gate = threading.Event()
        started = threading.Event()
        inputs = []

        def verify(payload):
            inputs.append(copy.deepcopy(payload))
            started.set()
            if not gate.wait(2):
                raise TimeoutError()
            return verdict("fail")

        manager = self.manager(verify)
        try:
            text = self.checkpoint("calculated 2*2 <progress>area is 4</progress>next calculation",
                                   reasoning="using the supplied side length",
                                   tool_calls=[{"id": "next", "function": {"name": "python"}}])
            manager.submit(self.messages, text, 1)
            self.assertTrue(started.wait(1))
            self.assertEqual(manager.poll(), [])
            self.assertEqual(inputs[0]["question_message"], self.question)
            self.assertNotIn("tool_calls", inputs[0]["segment"][-1])
            self.assertNotIn("next calculation", json.dumps(inputs[0]))
            self.messages.append({"role": "tool", "tool_call_id": "next", "content": "8"})
            text = self.checkpoint("<progress>volume is 8</progress>")
            manager.submit(self.messages, text, 2)
            payloads = [json.loads(line)["payload"] for line in manager.log_path.read_text().splitlines()
                        if json.loads(line)["event"] == "progress_checkpoint_submitted"]
            second = payloads[1]
            self.assertEqual(second["previous_checkpoint"]["status"], "pending")
            self.assertEqual(second["segment"][0]["tool_calls"][0]["id"], "next")
            self.assertEqual(second["segment"][0]["content"], "next calculation")
            self.assertEqual(second["segment"][1]["content"], "8")
            self.assertNotIn("using the supplied side length", json.dumps(second["segment"]))
        finally:
            gate.set()
            self.complete(manager)
        feedback = manager.poll()
        self.assertEqual(len(feedback), 2)
        self.assertEqual(manager.poll(), [])

    def test_submit_does_not_consume_ready_feedback(self):
        manager = self.manager(lambda payload: verdict("fail"))
        text = self.checkpoint()
        manager.submit(self.messages, text, 1)
        self.complete(manager)
        text = self.checkpoint("<progress>a second conclusion</progress>")
        manager.submit(self.messages, text, 2)
        self.complete(manager)
        feedback = manager.poll()
        self.assertEqual(len(feedback), 2)
        first = json.loads(feedback[0]["content"].split(">", 1)[1].split("</", 1)[0])
        self.assertEqual(first["checkpoint_id"], 1)
        self.assertEqual(first["result"]["verdict"], "fail")

    def test_feedback_filter_preserves_all_results_and_delivers_only_actionable_checks(self):
        cases = [
            (verdict("pass"), False),
            (verdict("fail"), True),
            (verdict("uncertain"), False),
            (dict(verdict("uncertain"), issue="Side length unreadable",
                  suggested_check="Read the side length from the original image"), True),
            (dict(verdict("uncertain"), suggested_check="Read the original image"), False),
        ]
        for result, should_deliver in cases:
            with self.subTest(result=result):
                manager = self.manager(lambda payload, result=result: result)
                text = self.checkpoint()
                manager.submit(self.messages, text, 1)
                self.complete(manager)
                feedback = manager.poll()
                self.assertEqual(bool(feedback), should_deliver)
                self.assertEqual(manager.poll(), [])
                record = manager.close()["checkpoints"][0]
                self.assertEqual(record["result"], result)
                self.assertEqual(record["delivered"], should_deliver)
                events = [json.loads(line) for line in manager.log_path.read_text().splitlines()]
                self.assertEqual(sum(e["event"] == "progress_feedback_suppressed" for e in events),
                                 int(not should_deliver))

    def test_endpoint_and_schema_failures_are_unavailable_not_failed_reasoning(self):
        def timeout(payload):
            raise TimeoutError("secret credentials must not be logged")
        for verify in (timeout, lambda payload: {"verdict": "pass"}):
            manager = self.manager(verify)
            text = self.checkpoint()
            manager.submit(self.messages, text, 1)
            self.complete(manager)
            result = manager.close()["checkpoints"][0]["result"]
            self.assertEqual(result["verdict"], "unavailable")
            self.assertEqual(manager.poll(), [])
            self.assertNotIn("secret credentials", manager.log_path.read_text())

    def test_capacity_and_invalid_tags_preserve_unchecked_evidence(self):
        with patch.dict(os.environ, {"PROGRESS_VERIFIER_MAX_INPUT_BYTES": "10"}):
            manager = self.manager(lambda payload: verdict())
        for text in ("<progress></progress>", "<progress>a</progress><progress>b</progress>",
                     "<progress>oversize</progress>"):
            manager.submit(self.messages + [{"role": "assistant", "content": text}], text, 1)
        self.assertEqual(manager.records, [])
        self.assertEqual(manager.boundary, 2)
        manager.max_bytes = 2000000
        manager.max_checkpoints = 1
        text = self.checkpoint()
        manager.submit(self.messages, text, 1)
        self.complete(manager)
        text = self.checkpoint()
        manager.submit(self.messages, text, 2)
        self.assertEqual(len(manager.records), 1)

    def test_close_does_not_wait_and_late_verdict_is_persisted(self):
        gate = threading.Event()
        entered = threading.Event()
        def verify(payload):
            entered.set()
            gate.wait(2)
            return verdict("fail")
        manager = self.manager(verify)
        text = self.checkpoint()
        manager.submit(self.messages, text, 1)
        self.assertTrue(entered.wait(1))
        try:
            before = time.monotonic()
            report = manager.close()
            self.assertLess(time.monotonic() - before, 0.5)
            self.assertEqual(report["checkpoints"][0]["status"], "pending")
        finally:
            gate.set()
            self.complete(manager)
        events = [json.loads(line) for line in manager.log_path.read_text().splitlines()]
        self.assertTrue(any(e["event"] == "progress_verification_completed" for e in events))
        self.assertFalse(any(e["event"] == "progress_feedback_delivered" for e in events))

    def test_scope_is_disabled_by_default_and_closes_on_exception(self):
        @pv.progress_verifier_scope
        def disabled():
            self.assertIsNone(pv.start_progress_verifier(self.question))
            return {"prediction": "4"}
        with patch.dict(os.environ, {"SCIENCE_AGENT_PROGRESS_VERIFIER": "0"}):
            self.assertEqual(disabled(), {"prediction": "4"})
        manager = self.manager(lambda payload: verdict())
        @pv.progress_verifier_scope
        def broken():
            pv.start_progress_verifier(self.question)
            raise RuntimeError("policy failed")
        with patch.dict(os.environ, {"SCIENCE_AGENT_PROGRESS_VERIFIER": "1"}), \
                patch.object(pv, "ProgressVerifier", return_value=manager):
            with self.assertRaisesRegex(RuntimeError, "policy failed"):
                broken()
        self.assertTrue(manager.closed)
        self.assertIsNone(pv._ACTIVE_VERIFIER.get())

    def test_api_contract_uses_independent_client_and_preserves_image_question(self):
        with patch.dict(os.environ, {"PROGRESS_VERIFIER_BASE_URL": "https://verifier.test/v1",
                                     "PROGRESS_VERIFIER_MODEL": "verifier-model",
                                     "PROGRESS_VERIFIER_API_KEY": "verifier-key",
                                     "PROGRESS_VERIFIER_EXTRA_BODY": '{"thinking":{"type":"disabled"}}'}):
            manager = pv.ProgressVerifier(self.question)
            self.addCleanup(manager.close)
            client = MagicMock()
            client.return_value.__enter__.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(verdict())))])
            with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=client)}):
                text = self.checkpoint()
                manager.submit(self.messages, text, 1)
                self.complete(manager)
            self.assertEqual(manager.close()["checkpoints"][0]["result"]["verdict"], "pass")
            self.assertEqual(client.call_args.kwargs["base_url"], "https://verifier.test/v1")
            self.assertEqual(client.call_args.kwargs["api_key"], "verifier-key")
            request = client.return_value.__enter__.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(request["messages"][1], self.question)
            self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})
            self.assertNotIn("verifier-key", manager.log_path.read_text())

    def test_actual_base_loop_runs_tools_before_verifier_and_delivers_feedback_next_turn(self):
        # Execute the real loop and new mixin without model/tokenizer dependencies.
        # No loop behavior is reimplemented in this test.
        base_source = ROOT / "inference/react_agent_qwen_api.py"
        tree = ast.parse(base_source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiTurnReactAgent")
        run = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_run"))
        run.decorator_list = []
        run.returns = None
        for arg in run.args.args:
            arg.annotation = None
        namespace = dict(copy=copy, os=os, time=time, json_parser=json,
                         MAX_LLM_CALL_PER_RUN=5, MAX_TOOL_CALLS_PER_RUN=0,
                         AGENT_EVAL_FORCE_FINALIZE=False, AGENT_CONTEXT_COMPACTION=False,
                         TRAJECTORY_MAX_TOKENS=100000,
                         _parse_served_model_fallbacks=lambda *a: [],
                         build_initial_user_message=lambda item, question, **kw: {"role": "user", "content": question},
                         merge_usage_dict=lambda *a: False, debug_print=lambda *a: None,
                         terminal_print=lambda *a: None, emit_trace=lambda *a, **kw: None,
                         _tool_arguments_hash=lambda *a: "hash", _python_tool_code_preview=lambda *a: {},
                         take_recovery_notice=lambda: None, tool_content_to_text=str)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[run], type_ignores=[])),
                     str(base_source), "exec"), namespace)
        mixin_source = ROOT / "inference/react_agent_qwen_api_progress.py"
        mixin = next(n for n in ast.parse(mixin_source.read_text()).body
                     if isinstance(n, ast.ClassDef) and n.name == "ProgressVerifierMixin")
        mix_namespace = dict(copy=copy, os=os, _RUN_STATE=ContextVar("test_progress_state", default=None),
                             PROGRESS_INSTRUCTION=pv.PROGRESS_INSTRUCTION,
                             PROGRESS_RE=pv.PROGRESS_RE,
                             progress_verifier_scope=pv.progress_verifier_scope,
                             start_progress_verifier=pv.start_progress_verifier)
        exec(compile(ast.Module(body=[mixin], type_ignores=[]), str(mixin_source), "exec"), mix_namespace)
        gate = threading.Event()
        entered = threading.Event()
        payloads = []
        def verify(payload):
            payloads.append(payload)
            entered.set()
            gate.wait(2)
            return verdict("fail")
        manager = self.manager(verify)
        call = SimpleNamespace(type="function", id="tool-1",
                               function=SimpleNamespace(name="python", arguments='{"code":"print(4)"}'))

        class Base:
            _run = namespace["_run"]
            served_model_name = "test-policy"
            _served_model_fallbacks_raw = ""
            llm_local_path = "unused"
            task_timeout_sec = 60
            def _build_system_prompt(self):
                return "Original system prompt"
            def _configure_task_tools(self, *a, **kw):
                return ""
            def _set_fallback_message_lists(self, *a):
                pass
            def _observation_masked_view(self, messages):
                return messages
            def _active_tool_schemas_for_current_task(self):
                return []
            def call_server(self, messages, *a, **kw):
                requests.append(copy.deepcopy(messages))
                return next(responses), {}
            def _extract_response_text_fields(self, response):
                return "", response.content
            def _recover_text_tool_calls(self, content, reasoning, round_idx):
                return None, content, reasoning
            def _normalize_serialized_tool_calls(self, calls):
                return calls
            def _build_assistant_request_message(self, reasoning, content, calls, output):
                return {"role": "assistant", "content": content, "tool_calls": calls}
            def custom_call_tool(self, *a):
                self_outer.assertTrue(entered.wait(1))
                self_outer.assertEqual(manager.poll(), [])
                gate.set()
                self_outer.complete(manager)
                return "4"
            def _extract_terminal_answer(self, content):
                return "4" if "<answer>" in content else ""
            def _compact_at_soft_limit(self, messages):
                return messages, ""
            def count_tokens(self, messages):
                return 10
            def _attach_usage_fields(self, result, *a):
                return result

        Agent = type("Agent", (mix_namespace["ProgressVerifierMixin"], Base), {})
        self_outer = self
        requests = []
        responses = iter([SimpleNamespace(content="computed <progress>area is 4</progress>", tool_calls=[call]),
                          SimpleNamespace(content="<progress>final area is 4</progress><answer>4</answer>", tool_calls=None)])
        with patch.dict(os.environ, {"SCIENCE_AGENT_PROGRESS_VERIFIER": "1"}), \
                patch.object(pv, "ProgressVerifier", return_value=manager):
            result = Agent()._run({"item": {"question": "compute", "answer": "SECRET_GROUND_TRUTH"}})
        self.assertEqual(result["prediction"], "4")
        self.assertEqual(len(requests), 2)
        self.assertTrue(requests[0][0]["content"].startswith("Original system prompt"))
        self.assertIn("<progress>", requests[0][0]["content"])
        self.assertEqual([m["role"] for m in requests[1]],
                         ["system", "user", "assistant", "tool", "user"])
        self.assertIn("<checkpoint_feedback>", requests[1][-1]["content"])
        self.assertEqual(result["progress_verifier"]["checkpoints"][0]["status"], "completed")
        self.assertNotIn("SECRET_GROUND_TRUTH", json.dumps(payloads))
        self.assertTrue(manager.closed)
        self.assertEqual(len(result["progress_verifier"]["checkpoints"]), 1)
        audit = [json.loads(line) for line in manager.log_path.read_text().splitlines()]
        self.assertTrue(any(event.get("reason") == "no_same_turn_tool_call" for event in audit))
        with patch.dict(os.environ, {"SCIENCE_AGENT_PROGRESS_VERIFIER": "0"}):
            self.assertEqual(Agent()._build_system_prompt(), "Original system prompt")


if __name__ == "__main__":
    unittest.main()
