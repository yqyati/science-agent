"""Exercise the isolated async adapter without loading policy-model dependencies."""

import ast
import asyncio
import importlib.util
import json
import logging
import os
import sys
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

HARNESS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_ROOT / "inference"))
import sandbox_gateway as gateway
import judge_client

ADAPTER = HARNESS_ROOT.parent.parent / "labs-molt/examples/python/agents/science_legacy_react_sandbox.py"


def load_adapter(prediction="5003"):
    tree = ast.parse(ADAPTER.read_text())
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) or
             isinstance(node, ast.ClassDef) and node.name == "ScienceLegacyReactAgent"]
    namespace = {
        "asyncio": asyncio, "json": json, "os": os, "re": __import__("re"),
        "SCIENCE_AGENT_ROOT": HARNESS_ROOT, "logger": logging.getLogger(__name__),
        "judge_answer": judge_client.judge_answer,
        "ChatAgent": object, "ChatContext": object, "Result": SimpleNamespace,
        "torch": SimpleNamespace(tensor=lambda value, **kw: value, float32=None),
        "sandbox_run_scope": gateway.sandbox_run_scope,
        "shared_sandbox_run": gateway.shared_sandbox_run,
        "AgentSandboxRun": lambda: gateway.AgentSandboxRun(),
        "MultiTurnReactAgent": lambda **kw: SimpleNamespace(
            _run=lambda *args: {"prediction": prediction, "messages": []}),
        "find_boxed_spans": lambda text: [],
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ADAPTER), "exec"), namespace)
    return namespace


def context():
    return SimpleNamespace(label={"answer": "5003"}, model_name="policy", prompt="What is the largest prime factor of 5003?",
                           sampling_params=SimpleNamespace(), base_url="http://policy", api_key="policy-key", messages=[])


def response(score=1):
    return {"status_code": 200, "text": json.dumps({"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"score": score, "correct": bool(score), "reason": "matched"})}}]})}


@unittest.skipUnless(ADAPTER.is_file(), "Requires the isolated Molt Sandbox adapter checkout")
class JudgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "SCIENCE_AGENT_JUDGE_BASE_URL": "http://judge/v1", "SCIENCE_AGENT_JUDGE_MODEL": "o3-mini",
            "SCIENCE_AGENT_JUDGE_API_KEY": "judge-key", "API_KEY": "fallback-key",
            "SCIENCE_AGENT_REWARD_MODE": "judge", "SCIENCE_AGENT_NEXT_HLE_EVAL_PROTOCOL": "0",
            "SCIENCE_AGENT_JUDGE_TIMEOUT": "7", "SCIENCE_AGENT_JUDGE_CONFIG": "",
            "SCIENCE_AGENT_JUDGE_FALLBACK_BASE_URL": "", "SCIENCE_AGENT_JUDGE_THINKING": "disabled",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.namespace = load_adapter()
        self.run = Mock()
        self.run.run_http.return_value = response()
        self.factory = Mock(return_value=self.run)
        factory_patch = patch.object(gateway, "AgentSandboxRun", self.factory)
        factory_patch.start()
        client_patch = patch.object(judge_client, "AgentSandboxRun", self.factory)
        client_patch.start()
        self.addCleanup(client_patch.stop)
        self.addCleanup(factory_patch.stop)
        self.agent = self.namespace["ScienceLegacyReactAgent"]()

    async def test_judge_transport_auth_and_cleanup(self):
        result = await self.agent.run(context())
        self.assertEqual(result.reward, 1)
        self.assertEqual(result.info["science_judge_valid"], 1)
        self.assertEqual(result.info["science_judge_backend"], "gateway_sandbox")
        payload = self.run.run_http.call_args.args[0]
        self.assertEqual(payload["url"], "http://judge/v1/chat/completions")
        self.assertEqual(payload["headers"]["Authorization"], "Bearer judge-key")
        self.assertEqual(payload["json"]["model"], "o3-mini")
        self.assertEqual(self.run.run_http.call_args.kwargs["timeout"], 7)
        self.run.close.assert_called_once()

    async def test_retry_reuses_run_and_preserves_incorrect_score(self):
        self.run.run_http.side_effect = [
            {"status_code": 599, "text": "connection timed out"}, response(0)]
        with patch.object(asyncio, "sleep", return_value=None) as sleep:
            result = await self.agent.run(context())
        self.assertEqual(result.reward, 0)
        self.assertEqual(result.info["science_judge_valid"], 1)
        self.assertEqual(self.run.run_http.call_count, 2)
        sleep.assert_awaited_once_with(1)
        self.factory.assert_called_once()
        self.run.close.assert_called_once()

    async def test_failed_judge_retains_status_and_closes(self):
        replacements = [Mock()]
        runs = [self.run, *replacements]
        self.factory.side_effect = runs
        for run in runs:
            run.run_http.return_value = {"status_code": 503, "text": "service unavailable"}
        with patch.object(asyncio, "sleep", return_value=None) as sleep:
            result = await self.agent.run(context())
        self.assertEqual(result.reward, -1)
        self.assertEqual(result.score, -1)
        self.assertEqual(result.info["science_judge_valid"], 0)
        self.assertIn("503", result.info["science_judge_error"])
        self.assertEqual(result.info["science_judge_response"], "service unavailable")
        self.assertEqual(self.run.run_http.call_count, 7)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [1, 5, 10, 30, 60, 180] * 2)
        self.assertEqual(result.info["science_judge_attempts"], 14)
        self.assertEqual(result.info["science_judge_sandbox_restarts"], 1)
        self.assertEqual(self.factory.call_count, 2)
        for run in runs:
            self.assertEqual(run.run_http.call_count, 7)
            run.close.assert_called_once()

    async def test_judge_recovers_on_last_retry_without_regenerating_answer(self):
        generate = Mock(return_value={"prediction": "5003", "messages": []})
        self.namespace["MultiTurnReactAgent"] = lambda **kw: SimpleNamespace(_run=generate)
        self.run.run_http.side_effect = [{"status_code": 599, "text": "connection refused"}] * 6 + [response()]
        with patch.object(asyncio, "sleep", return_value=None) as sleep:
            result = await self.agent.run(context())
        self.assertEqual(result.reward, 1)
        self.assertEqual(result.info["science_judge_valid"], 1)
        self.assertEqual(self.run.run_http.call_count, 7)
        self.assertEqual(sleep.await_count, 6)
        generate.assert_called_once()
        self.factory.assert_called_once()
        self.run.close.assert_called_once()

    async def test_replacement_preserves_answer_and_accepts_incorrect_grade(self):
        generate = Mock(return_value={"prediction": "5003", "messages": []})
        self.namespace["MultiTurnReactAgent"] = lambda **kw: SimpleNamespace(_run=generate)
        replacement = Mock()
        replacement.run_http.return_value = response(0)
        self.factory.side_effect = [self.run, replacement]
        self.run.run_http.return_value = {"status_code": 599, "text": "connection refused"}
        with patch.object(asyncio, "sleep", return_value=None):
            result = await self.agent.run(context())
        self.assertEqual(result.reward, 0)
        self.assertEqual(result.info["science_judge_valid"], 1)
        self.assertEqual(result.info["science_prediction"], "5003")
        self.assertEqual(result.info["science_judge_attempts"], 8)
        self.assertEqual(result.info["science_judge_sandbox_restarts"], 1)
        generate.assert_called_once()
        self.assertEqual(self.factory.call_count, 2)
        self.assertEqual(replacement.run_http.call_args, self.run.run_http.call_args)
        self.run.close.assert_called_once()
        replacement.close.assert_called_once()

    async def test_replacement_cleanup_failure_preserves_valid_grade(self):
        replacement = Mock()
        self.factory.side_effect = [self.run, replacement]
        self.run.run_http.side_effect = gateway.SandboxStateError("pending recovery notice")
        replacement.run_http.return_value = response()
        replacement.close.side_effect = RuntimeError("cleanup failed")
        result = await self.agent.run(context())
        self.assertEqual(result.reward, 1)
        self.assertEqual(result.info["science_judge_sandbox_restarts"], 1)
        self.assertEqual(result.info["science_judge_attempts"], 2)
        self.run.close.assert_called_once()
        replacement.close.assert_called_once()

    async def test_cancel_replacement_waits_for_request_and_closes_both_runs(self):
        started, release = Event(), Event()
        replacement = Mock()
        def request(*args, **kwargs):
            started.set()
            release.wait(5)
            return response()
        replacement.run_http.side_effect = request
        self.factory.side_effect = [self.run, replacement]
        self.run.run_http.return_value = {"status_code": 599, "text": "connection refused"}
        with patch.object(asyncio, "sleep", return_value=None):
            task = asyncio.create_task(self.agent.run(context()))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                task.cancel()
                replacement.close.assert_not_called()
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.run.close.assert_called_once()
        replacement.close.assert_called_once()

    async def test_invalid_json_is_retried_and_cleanup_failure_preserves_grade(self):
        self.run.run_http.side_effect = [{"status_code": 200, "text": "invalid JSON"}, response()]
        self.run.close.side_effect = RuntimeError("delete unavailable")
        with patch.object(asyncio, "sleep", return_value=None):
            result = await self.agent.run(context())
        self.assertEqual(result.reward, 1)
        self.run.close.assert_called_once()

    async def test_empty_answer_and_f1_do_not_create_judge_sandbox(self):
        self.namespace["MultiTurnReactAgent"] = lambda **kw: SimpleNamespace(_run=lambda *args: {"prediction": ""})
        result = await self.agent.run(context())
        self.assertEqual(result.reward, 0)
        self.run.run_http.assert_not_called()
        self.run.ensure.assert_not_called()
        self.namespace["MultiTurnReactAgent"] = lambda **kw: SimpleNamespace(_run=lambda *args: {"prediction": "5003"})
        with patch.dict(os.environ, {"SCIENCE_AGENT_REWARD_MODE": "f1"}):
            result = await self.agent.run(context())
        self.assertEqual(result.reward, 1)
        self.run.run_http.assert_not_called()
        self.run.ensure.assert_not_called()

    async def test_budget_exhausted_placeholder_skips_judge_and_marks_truncated(self):
        self.namespace["MultiTurnReactAgent"] = lambda **kw: SimpleNamespace(_run=lambda *args: {
            "prediction": "your answer", "trajectory_budget_exhausted": True, "generation_truncated": False})
        result = await self.agent.run(context())
        self.assertEqual(result.reward, 0)
        self.assertTrue(result.truncated)
        self.assertEqual(result.info["science_prediction"], "")
        self.assertEqual(result.info["science_final_answer_valid"], 0)
        self.assertEqual(result.info["science_generation_truncated"], 0)
        self.assertEqual(result.info["science_trajectory_budget_exhausted"], 1)
        self.assertEqual(result.info["science_termination"], "trajectory budget exhausted without valid final answer")
        self.run.run_http.assert_not_called()
        self.run.ensure.assert_not_called()

    async def test_budget_exhausted_real_answer_is_graded(self):
        self.namespace["MultiTurnReactAgent"] = lambda **kw: SimpleNamespace(_run=lambda *args: {
            "prediction": "5003", "trajectory_budget_exhausted": True, "generation_truncated": False})
        result = await self.agent.run(context())
        self.assertEqual(result.reward, 1)
        self.assertTrue(result.truncated)
        self.assertEqual(result.info["science_final_answer_valid"], 1)
        self.run.run_http.assert_called_once()

    async def test_generation_truncation_is_distinct_from_budget_exhaustion(self):
        self.namespace["MultiTurnReactAgent"] = lambda **kw: SimpleNamespace(_run=lambda *args: {
            "prediction": "5003", "generation_truncated": True, "trajectory_budget_exhausted": False})
        with patch.dict(os.environ, {"SCIENCE_AGENT_REWARD_MODE": "f1"}):
            result = await self.agent.run(context())
        self.assertTrue(result.truncated)
        self.assertEqual(result.info["science_generation_truncated"], 1)
        self.assertEqual(result.info["science_trajectory_budget_exhausted"], 0)

    async def test_cancel_waits_for_inflight_request_before_cleanup(self):
        started, release = Event(), Event()

        def request(*args, **kwargs):
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test release timed out")
            return response()

        self.run.run_http.side_effect = request
        task = asyncio.create_task(self.agent.run(context()))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            task.cancel()
            await asyncio.sleep(0.02)
            self.run.close.assert_not_called()
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.run.close.assert_called_once()

    async def test_concurrent_judges_own_distinct_runs_with_fallback_auth(self):
        other = Mock()
        other.run_http.return_value = response()
        self.factory.side_effect = [self.run, other]
        with patch.dict(os.environ, {"SCIENCE_AGENT_JUDGE_API_KEY": ""}):
            results = await asyncio.gather(self.agent.run(context()), self.agent.run(context()))
        self.assertEqual([result.reward for result in results], [1, 1])
        for run in (self.run, other):
            self.assertEqual(run.run_http.call_args.args[0]["headers"]["Authorization"], "Bearer fallback-key")
            run.close.assert_called_once()


class SmokeJudgeTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("sandbox_smoke", HARNESS_ROOT / "sandbox_smoke.py")
        self.smoke = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.smoke)
        self.run = Mock()
        self.run.run_python.side_effect = ["42", "41"]
        self.run.run_http.return_value = response()
        patches = [
            patch.dict(os.environ, {
                "SCIENCE_AGENT_TOOL_HTTP_BACKEND": "gateway_sandbox",
                "SCIENCE_AGENT_JUDGE_BASE_URL": "http://judge/v1",
                "SCIENCE_AGENT_JUDGE_MODEL": "o3-mini", "API_KEY": "test-key",
                "SCIENCE_AGENT_JUDGE_CONFIG": "", "SCIENCE_AGENT_JUDGE_FALLBACK_BASE_URL": "",
                "SCIENCE_AGENT_JUDGE_THINKING": "disabled",
            }),
            patch.object(sys, "argv", ["sandbox_smoke.py", "--judge", "--attempts", "1"]),
            patch.object(self.smoke, "shared_sandbox_run", return_value=self.run),
            patch.object(self.smoke, "close_shared_sandbox_run"),
            patch.object(self.smoke, "tool_http_request", return_value=Mock(
                status_code=200, backend="gateway_sandbox", text="Example Domain")),
            patch.object(self.smoke.time, "sleep"),
            patch.object(asyncio, "sleep", return_value=None),
            patch.object(judge_client, "AgentSandboxRun", return_value=self.run),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_first_success_does_not_wait(self):
        self.smoke.main()
        self.run.run_http.assert_called_once()
        self.smoke.time.sleep.assert_not_called()
        self.smoke.close_shared_sandbox_run.assert_called_once()

    def test_last_retry_recovers_after_transport_and_response_failures(self):
        self.run.run_http.side_effect = [
            RuntimeError("gateway unavailable"), {"status_code": 599, "text": "connection refused"},
            {"status_code": 200, "text": "invalid JSON"},
        ] * 2 + [response()]
        self.smoke.main()
        self.assertEqual(self.run.run_http.call_count, 7)
        self.smoke.time.sleep.assert_not_called()
        self.smoke.close_shared_sandbox_run.assert_called_once()

    def test_exhausted_retries_still_fail_preflight_and_clean_up(self):
        self.run.run_http.return_value = {"status_code": 599, "text": "connection refused"}
        with self.assertRaisesRegex(AssertionError, "Judge HTTP 599"):
            self.smoke.main()
        self.assertEqual(self.run.run_http.call_count, 14)
        self.smoke.time.sleep.assert_not_called()
        self.smoke.close_shared_sandbox_run.assert_called_once()

    def test_whole_preflight_retries_with_fresh_instance_after_startup_failure(self):
        first = Mock()
        first.ensure.side_effect = gateway.SandboxGatewayError("command HTTP 403")
        with patch.object(sys, "argv", ["sandbox_smoke.py", "--judge"]), \
                patch.object(self.smoke, "shared_sandbox_run", side_effect=[first, self.run, self.run]):
            self.smoke.main()
        first.run_python.assert_not_called()
        self.run.run_http.assert_called_once()
        self.assertEqual(self.smoke.close_shared_sandbox_run.call_count, 2)
        self.smoke.time.sleep.assert_called_once_with(5)

    def test_whole_preflight_stops_after_three_failed_attempts(self):
        self.run.ensure.side_effect = gateway.SandboxGatewayError("command HTTP 403")
        with patch.object(sys, "argv", ["sandbox_smoke.py", "--judge"]):
            with self.assertRaisesRegex(gateway.SandboxGatewayError, "command HTTP 403"):
                self.smoke.main()
        self.assertEqual(self.run.ensure.call_count, 3)
        self.assertEqual(self.smoke.close_shared_sandbox_run.call_count, 3)
        self.assertEqual([call.args[0] for call in self.smoke.time.sleep.call_args_list], [5, 15])

    def test_cleanup_failure_does_not_start_another_instance(self):
        self.run.ensure.side_effect = gateway.SandboxGatewayError("command HTTP 403")
        self.smoke.close_shared_sandbox_run.side_effect = RuntimeError("deletion unconfirmed")
        with patch.object(sys, "argv", ["sandbox_smoke.py", "--judge"]):
            with self.assertRaisesRegex(SystemExit, "cleanup failed"):
                self.smoke.main()
        self.run.ensure.assert_called_once()
        self.smoke.time.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
