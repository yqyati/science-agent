"""Exercise the isolated async adapter without loading policy-model dependencies."""

import ast
import asyncio
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

ADAPTER = HARNESS_ROOT.parent.parent / "labs-molt/examples/python/agents/science_legacy_react_sandbox.py"


def load_adapter(prediction="5003"):
    tree = ast.parse(ADAPTER.read_text())
    nodes = [node for node in tree.body if
             isinstance(node, ast.FunctionDef) or
             isinstance(node, ast.ClassDef) and node.name == "ScienceLegacyReactAgent"]
    namespace = {
        "asyncio": asyncio, "json": json, "os": os, "re": __import__("re"),
        "SCIENCE_AGENT_ROOT": HARNESS_ROOT, "logger": logging.getLogger(__name__),
        "ChatAgent": object, "ChatContext": object, "Result": SimpleNamespace,
        "torch": SimpleNamespace(tensor=lambda value, **kw: value, float32=None),
        "sandbox_run_scope": gateway.sandbox_run_scope,
        "shared_sandbox_run": gateway.shared_sandbox_run,
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
            "SCIENCE_AGENT_JUDGE_TIMEOUT": "7",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.namespace = load_adapter()
        self.run = Mock()
        self.run.run_http.return_value = response()
        self.factory = Mock(return_value=self.run)
        factory_patch = patch.object(gateway, "AgentSandboxRun", self.factory)
        factory_patch.start()
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
        self.run.run_http.return_value = {"status_code": 503, "text": "service unavailable"}
        with patch.object(asyncio, "sleep", return_value=None) as sleep:
            result = await self.agent.run(context())
        self.assertEqual(result.reward, -1)
        self.assertEqual(result.score, -1)
        self.assertEqual(result.info["science_judge_valid"], 0)
        self.assertIn("503", result.info["science_judge_error"])
        self.assertEqual(result.info["science_judge_response"], "service unavailable")
        self.assertEqual(self.run.run_http.call_count, 9)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [1, 5, 10, 30, 60, 300, 300, 300])
        self.run.close.assert_called_once()

    async def test_judge_recovers_on_last_retry_without_regenerating_answer(self):
        generate = Mock(return_value={"prediction": "5003", "messages": []})
        self.namespace["MultiTurnReactAgent"] = lambda **kw: SimpleNamespace(_run=generate)
        self.run.run_http.side_effect = [{"status_code": 599, "text": "connection refused"}] * 8 + [response()]
        with patch.object(asyncio, "sleep", return_value=None) as sleep:
            result = await self.agent.run(context())
        self.assertEqual(result.reward, 1)
        self.assertEqual(result.info["science_judge_valid"], 1)
        self.assertEqual(self.run.run_http.call_count, 9)
        self.assertEqual(sleep.await_count, 8)
        generate.assert_called_once()
        self.factory.assert_called_once()
        self.run.close.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()
