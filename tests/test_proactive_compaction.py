"""Exercise real budget/rollout methods without model, network, or sandbox dependencies."""

import ast
import copy
import json
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "inference/react_agent_qwen_api.py"
tree = ast.parse(SOURCE.read_text())
agent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiTurnReactAgent")
methods = [copy.deepcopy(n) for n in agent.body if isinstance(n, ast.FunctionDef)
           and n.name in {"_compact_at_soft_limit", "_run"}]
for method in methods:
    method.decorator_list = []
cls = ast.ClassDef(name="Agent", bases=[], keywords=[], body=methods, decorator_list=[])
namespace = {
    "copy": copy, "os": os, "time": time, "AGENT_CONTEXT_COMPACTION": True,
    "AGENT_CONTEXT_COMPACT_THRESHOLD_TOKENS": 56, "TRAJECTORY_MAX_TOKENS": 64,
    "AGENT_EVAL_FORCE_FINALIZE": False, "MAX_LLM_CALL_PER_RUN": 10,
    "MAX_TOOL_CALLS_PER_RUN": 0, "CONTEXT_MANAGEMENT_STRATEGY": "none",
    "terminal_print": Mock(), "debug_print": Mock(), "OpenAI": Mock(),
    "build_initial_user_message": lambda *a, **kw: {"role": "user", "content": "Q"},
    "_parse_served_model_fallbacks": lambda *a: (),
    "merge_usage_dict": lambda target, usage: bool(usage),
    "json_parser": json, "emit_trace": Mock(),
    "_tool_arguments_hash": lambda s: "test",
    "_python_tool_code_preview": lambda *a: {},
    "take_recovery_notice": lambda: "", "tool_content_to_text": str,
}
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)


class ProactiveCompactionTests(unittest.TestCase):
    def setUp(self):
        self.compactor = SimpleNamespace(_compact_messages=Mock(return_value=[{"role": "user", "content": "summary"}]))
        self.env = patch.dict(namespace, {"_get_context_compactor": lambda: self.compactor})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.agent = namespace["Agent"]()
        a = self.agent
        a._context_compaction_count = 0
        a._observation_masked_view = lambda m: m
        a.count_tokens = lambda m: sum(len(x.get("content", "")) for x in m)
        a._get_api_key = lambda: "test"
        a._get_api_base = lambda: "http://unused.invalid"
        a.request_timeout = a.task_timeout_sec = 10
        a.served_model_name = "policy"
        a.served_model_name_actual = None
        a._served_model_fallbacks_raw = ""
        a.llm_local_path = "policy"
        a._active_tool_schemas_for_current_task = lambda: []
        a._configure_task_tools = lambda *args, **kw: ""
        a._build_system_prompt = lambda: "S"
        a._set_fallback_message_lists = lambda *args: None
        a._extract_response_text_fields = lambda r: ("", r.content)
        a._recover_text_tool_calls = lambda c, r, i: (None, c, r)
        a._normalize_serialized_tool_calls = lambda t: t
        a._build_assistant_request_message = lambda r, c, t, *args: {"role": "assistant", "content": c}
        a._extract_terminal_answer = lambda c: "42" if "<answer>42</answer>" in c else ""
        a._last_substantive_assistant_content = lambda m: m[-1]["content"]
        a._attach_usage_fields = lambda result, *args: result
        a._attempt_forced_finalization = Mock(return_value=({"content": "<answer>42</answer>"}, {}))

    def run_task(self, outputs):
        self.agent.call_server = Mock(side_effect=[
            (c if hasattr(c, "content") else SimpleNamespace(content=c, tool_calls=None), {})
            for c in outputs
        ])
        return self.agent._run({"item": {"question": "Q", "answer": "42"}})

    def test_disabled_and_below_threshold_preserve_context(self):
        for enabled, length in [(False, 60), (True, 55)]:
            with self.subTest(enabled=enabled), patch.dict(namespace, {"AGENT_CONTEXT_COMPACTION": enabled}):
                messages = [{"role": "user", "content": "x" * length}]
                result, reason = self.agent._compact_at_soft_limit(messages)
                self.assertIs(result, messages)
                self.assertEqual(reason, "")
        self.compactor._compact_messages.assert_not_called()

    def test_compact_once_then_finalize_at_second_crossing(self):
        result = self.run_task(["x" * 56, "y" * 56])
        self.assertEqual(result["prediction"], "42")
        self.assertEqual(result["context_compaction_count"], 1)
        self.assertIn("after compaction", result["termination"])
        self.compactor._compact_messages.assert_called_once()
        # Next generation sees the compacted context, but diagnostic history retains the first turn.
        self.assertEqual(self.agent.call_server.call_args_list[1].args[0][0]["content"], "summary")
        self.assertEqual(result["messages"][2]["content"], "x" * 56)
        self.agent._attempt_forced_finalization.assert_called_once()

    def test_final_answer_takes_precedence_over_compaction(self):
        result = self.run_task(["x" * 60 + "<answer>42</answer>"])
        self.assertEqual(result["termination"], "answer")
        self.compactor._compact_messages.assert_not_called()
        self.agent._attempt_forced_finalization.assert_not_called()

    def test_failure_and_ineffective_summary_finalize_without_retry_loop(self):
        for summary in [None, [{"role": "user", "content": "z" * 100}]]:
            with self.subTest(summary=summary):
                self.compactor._compact_messages.reset_mock()
                self.compactor._compact_messages.return_value = summary
                result = self.run_task(["x" * 56])
                self.assertEqual(result["prediction"], "42")
                self.assertEqual(self.agent.call_server.call_count, 1)
                self.compactor._compact_messages.assert_called_once()

    def test_reduced_but_still_above_threshold_finalizes_with_shorter_context(self):
        self.compactor._compact_messages.return_value = [{"role": "user", "content": "z" * 56}]
        result = self.run_task(["x" * 60])
        self.assertEqual(result["prediction"], "42")
        context = self.agent._attempt_forced_finalization.call_args.args[0]
        self.assertEqual(context[0]["content"], "z" * 56)

    def test_counter_resets_for_next_task(self):
        self.run_task(["x" * 56, "<answer>42</answer>"])
        self.run_task(["y" * 56, "<answer>42</answer>"])
        self.assertEqual(self.compactor._compact_messages.call_count, 2)

    def test_disabled_keeps_original_no_answer_exit(self):
        with patch.dict(namespace, {"AGENT_CONTEXT_COMPACTION": False}):
            result = self.run_task(["x" * 60])
        self.assertEqual(result["prediction"], "")
        self.assertEqual(result["termination"], "answer not found")
        self.compactor._compact_messages.assert_not_called()
        self.agent._attempt_forced_finalization.assert_not_called()

    def test_tool_result_is_checked_before_next_generation(self):
        tool = SimpleNamespace(id="call1", type="function",
                               function=SimpleNamespace(name="python", arguments="{}"))
        self.agent.custom_call_tool = Mock(return_value="t" * 56)
        turn = SimpleNamespace(content="", tool_calls=[tool])
        result = self.run_task([turn, "<answer>42</answer>"])
        self.assertEqual(result["prediction"], "42")
        self.compactor._compact_messages.assert_called_once()
        self.assertEqual(self.agent.call_server.call_args_list[1].args[0][0]["content"], "summary")

    def test_malformed_answer_tag_does_not_block_finalization(self):
        self.compactor._compact_messages.return_value = None
        result = self.run_task(["x" * 56 + "<answer>"])
        self.assertEqual(result["prediction"], "42")
        self.agent._attempt_forced_finalization.assert_called_once()

    def test_error_recovery_compaction_uses_same_task_allowance(self):
        self.agent._context_compaction_count = 1
        _, reason = self.agent._compact_at_soft_limit([{"role": "user", "content": "x" * 56}])
        self.assertIn("after compaction", reason)
        self.compactor._compact_messages.assert_not_called()


if __name__ == "__main__":
    unittest.main()
