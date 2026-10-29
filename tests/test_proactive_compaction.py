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
           and n.name in {"_compact_at_soft_limit", "_attempt_forced_finalization", "_run",
                          "_build_forced_finalization_user_message"}]
for method in methods:
    method.decorator_list = []
cls = ast.ClassDef(name="Agent", bases=[], keywords=[], body=methods, decorator_list=[])
namespace = {
    "copy": copy, "json": json, "os": os, "time": time, "AGENT_CONTEXT_COMPACTION": True,
    "AGENT_CONTEXT_COMPACT_THRESHOLD_TOKENS": 56, "TRAJECTORY_MAX_TOKENS": 64,
    "AGENT_CONTEXT_COMPACT_MAX_COMPACTIONS": 1,
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
compactor_class = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ContextCompactor"))
exec(compile(ast.fix_missing_locations(ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), compactor_class,
], type_ignores=[])), str(SOURCE), "exec"), namespace)


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

    def test_finalization_answer_type_overrides_question_keywords(self):
        for answer_type in ("boxed", "plain"):
            for question in ("Q", r"Final Answer: \boxed{}"):
                with self.subTest(answer_type=answer_type, question=question), patch.dict(
                    os.environ, {"AGENT_ANSWER_TYPE": answer_type}
                ):
                    prompt = self.agent._build_forced_finalization_user_message(question, "soft limit")["content"]
                    example = "Final Answer:\n\\boxed{XXXX}" if answer_type == "boxed" else "XXXX"
                    self.assertIn(f"<answer>\n{example}\n</answer>", prompt)
                    self.assertEqual("\\boxed" in prompt, answer_type == "boxed")

    def test_finalization_defaults_to_boxed_and_rejects_unknown_type(self):
        with patch.dict(os.environ):
            os.environ.pop("AGENT_ANSWER_TYPE", None)
            prompt = self.agent._build_forced_finalization_user_message("Q", "soft limit")["content"]
            self.assertIn("<answer>\nFinal Answer:\n\\boxed{XXXX}\n</answer>", prompt)
            os.environ["AGENT_ANSWER_TYPE"] = "unknown"
            with self.assertRaisesRegex(ValueError, "AGENT_ANSWER_TYPE"):
                self.agent._build_forced_finalization_user_message("Q", "soft limit")

    def test_initial_wrapping_and_finalization_use_same_answer_type(self):
        prompt_tree = ast.parse((SOURCE.parent / "prompt.py").read_text())
        scope = {"os": os}
        for node in prompt_tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {
                        "SCIENCE_USER_TEMPLATE", "SCIENCE_USER_TEMPLATE_PLAIN"
                    }:
                        scope[target.id] = ast.literal_eval(node.value)
        input_tree = ast.parse((SOURCE.parent / "multimodal_input.py").read_text())
        builder = next(n for n in input_tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "build_initial_user_message")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations")], level=0), builder], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
        scope.update(resolve_item_image_urls=lambda *a, **kw: ["data:image/png;base64,test"],
                     build_static_multimodal_support_error=lambda *a, **kw: None)
        for answer_type in ("boxed", "plain"):
            for images in (False, True):
                with self.subTest(answer_type=answer_type, images=images), patch.dict(
                    os.environ, {"AGENT_ANSWER_TYPE": answer_type}
                ):
                    scope["model_image_inputs_enabled"] = lambda: images
                    message = scope["build_initial_user_message"](
                        {}, "Q", model_path=None, served_model_name="policy")
                    text = message["content"][0]["text"] if images else message["content"]
                    self.assertEqual(text, scope["SCIENCE_USER_TEMPLATE" if answer_type == "boxed"
                                                 else "SCIENCE_USER_TEMPLATE_PLAIN"].format(question="Q"))
                    final = self.agent._build_forced_finalization_user_message("Q", "soft limit")["content"]
                    self.assertEqual("\\boxed" in text, "\\boxed" in final)
                    if images:
                        self.assertEqual(message["content"][1]["type"], "image_url")

    def test_disabled_and_below_threshold_preserve_context(self):
        for enabled, length in [(False, 60), (True, 55)]:
            with self.subTest(enabled=enabled), patch.dict(namespace, {"AGENT_CONTEXT_COMPACTION": enabled}):
                messages = [{"role": "user", "content": "x" * length}]
                result, reason = self.agent._compact_at_soft_limit(messages)
                self.assertIs(result, messages)
                self.assertEqual(reason, "")
        self.compactor._compact_messages.assert_not_called()

    def test_compact_once_then_finalize_at_second_crossing(self):
        switch = patch.dict(namespace, {"AGENT_EVAL_FORCE_FINALIZE": True})
        switch.start()
        self.addCleanup(switch.stop)
        result = self.run_task(["x" * 56, "y" * 56])
        self.assertEqual(result["prediction"], "42")
        self.assertEqual(result["context_compaction_count"], 1)
        self.assertIn("after compaction", result["termination"])
        self.compactor._compact_messages.assert_called_once()
        # Next generation sees the compacted context, but diagnostic history retains the first turn.
        self.assertEqual(self.agent.call_server.call_args_list[1].args[0][0]["content"], "summary")
        self.assertEqual(result["messages"][2]["content"], "x" * 56)
        self.agent._attempt_forced_finalization.assert_called_once()

    def test_three_compactions_then_finalize_at_fourth_crossing(self):
        self.compactor._compact_messages.side_effect = lambda *args: [{"role": "user", "content": "summary"}]
        with patch.dict(namespace, {"AGENT_CONTEXT_COMPACT_MAX_COMPACTIONS": 3,
                                   "AGENT_EVAL_FORCE_FINALIZE": True}):
            result = self.run_task(["x" * 56] * 4)
        self.assertEqual(result["prediction"], "42")
        self.assertEqual(result["context_compaction_count"], 3)
        self.assertEqual(self.compactor._compact_messages.call_count, 3)
        self.assertEqual(self.agent.call_server.call_count, 4)
        self.agent._attempt_forced_finalization.assert_called_once()

    def test_three_compactions_without_finalization_can_continue(self):
        self.compactor._compact_messages.side_effect = lambda *args: [{"role": "user", "content": "summary"}]
        with patch.dict(namespace, {"AGENT_CONTEXT_COMPACT_MAX_COMPACTIONS": 3,
                                   "AGENT_EVAL_FORCE_FINALIZE": False}):
            result = self.run_task(["x" * 56] * 4 + ["<answer>42</answer>"])
        self.assertEqual(result["termination"], "answer")
        self.assertEqual(result["context_compaction_count"], 3)
        self.assertEqual(self.compactor._compact_messages.call_count, 3)
        self.agent._attempt_forced_finalization.assert_not_called()

    def test_final_answer_takes_precedence_over_compaction(self):
        result = self.run_task(["x" * 60 + "<answer>42</answer>"])
        self.assertEqual(result["termination"], "answer")
        self.compactor._compact_messages.assert_not_called()
        self.agent._attempt_forced_finalization.assert_not_called()

    def test_failure_and_ineffective_summary_finalize_without_retry_loop(self):
        switch = patch.dict(namespace, {"AGENT_EVAL_FORCE_FINALIZE": True})
        switch.start()
        self.addCleanup(switch.stop)
        for summary in [None, [{"role": "user", "content": "z" * 100}]]:
            with self.subTest(summary=summary):
                self.compactor._compact_messages.reset_mock()
                self.compactor._compact_messages.return_value = summary
                result = self.run_task(["x" * 56])
                self.assertEqual(result["prediction"], "42")
                self.assertEqual(self.agent.call_server.call_count, 1)
                self.compactor._compact_messages.assert_called_once()

    def test_reduced_but_still_above_threshold_finalizes_with_shorter_context(self):
        switch = patch.dict(namespace, {"AGENT_EVAL_FORCE_FINALIZE": True})
        switch.start()
        self.addCleanup(switch.stop)
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

    def test_no_compaction_finalizes_at_first_soft_crossing(self):
        with patch.dict(namespace, {"AGENT_CONTEXT_COMPACTION": False, "AGENT_EVAL_FORCE_FINALIZE": True}):
            result = self.run_task(["x" * 56])
        self.assertEqual(result["prediction"], "42")
        self.assertEqual(result["context_compaction_count"], 0)
        self.assertIn("without compaction", result["termination"])
        self.compactor._compact_messages.assert_not_called()
        self.agent._attempt_forced_finalization.assert_called_once()
        self.assertEqual(self.agent.call_server.call_count, 1)

    def test_no_compaction_checks_tool_result_at_soft_threshold(self):
        tool = SimpleNamespace(id="call1", type="function",
                               function=SimpleNamespace(name="python", arguments="{}"))
        self.agent.custom_call_tool = Mock(return_value="t" * 56)
        with patch.dict(namespace, {"AGENT_CONTEXT_COMPACTION": False, "AGENT_EVAL_FORCE_FINALIZE": True}):
            result = self.run_task([SimpleNamespace(content="", tool_calls=[tool])])
        self.assertEqual(result["prediction"], "42")
        self.assertEqual(result["context_compaction_count"], 0)
        self.compactor._compact_messages.assert_not_called()
        self.agent._attempt_forced_finalization.assert_called_once()
        self.assertEqual(self.agent.call_server.call_count, 1)

    def test_finalize_only_below_threshold_and_valid_answer_do_not_force(self):
        with patch.dict(namespace, {"AGENT_CONTEXT_COMPACTION": False, "AGENT_EVAL_FORCE_FINALIZE": True}):
            _, reason = self.agent._compact_at_soft_limit([{"role": "user", "content": "x" * 55}])
            self.assertEqual(reason, "")
            result = self.run_task(["x" * 60 + "<answer>42</answer>"])
        self.assertEqual(result["termination"], "answer")
        self.agent._attempt_forced_finalization.assert_not_called()
        self.compactor._compact_messages.assert_not_called()

    def test_compaction_calls_real_finalizer_when_enabled(self):
        # Exercise the real finalizer as well as the rollout decision.
        self.agent._attempt_forced_finalization = namespace["Agent"]._attempt_forced_finalization.__get__(self.agent)
        self.agent._build_forced_finalization_user_message = lambda *a: {"role": "user", "content": "Finalize"}
        self.agent._clean_limit_apology = lambda text: text
        self.agent.max_output_tokens = 8
        with patch.dict(namespace, {"AGENT_EVAL_FORCE_FINALIZE": True,
                                   "AGENT_EVAL_FORCE_FINAL_MAX_OUTPUT_TOKENS": 8,
                                   "AGENT_EVAL_FORCE_FINAL_DISABLE_THINKING": False}):
            result = self.run_task(["x" * 56, "y" * 56, "<answer>42</answer>"])
        self.assertEqual(result["prediction"], "42")
        self.assertEqual(self.agent.call_server.call_count, 3)
        self.assertEqual(self.agent.call_server.call_args.kwargs["tool_schemas"], [])
        self.assertEqual(self.agent.call_server.call_args.args[0][-1]["content"], "Finalize")

    def test_only_molt_finalization_requests_token_append(self):
        self.agent._clean_limit_apology = lambda text: text
        self.agent.max_output_tokens = 8
        for molt in (False, True):
            with self.subTest(molt=molt), patch.dict(namespace, {
                "AGENT_EVAL_FORCE_FINALIZE": True,
                "AGENT_EVAL_FORCE_FINAL_MAX_OUTPUT_TOKENS": 8,
                "AGENT_EVAL_FORCE_FINAL_DISABLE_THINKING": False,
            }):
                self.agent._molt_append_finalization = molt
                self.agent.call_server = Mock(return_value=(SimpleNamespace(content="<answer>42</answer>", tool_calls=None), {}))
                namespace["Agent"]._attempt_forced_finalization(self.agent, [{"role": "user", "content": "Q"}],
                                                               None, "Q", "soft limit")
                extra = self.agent.call_server.call_args.kwargs["request_extra_body"]
                self.assertEqual(extra, {"molt_append_finalization": True} if molt else {})

    def test_compact_only_continues_after_second_crossing_without_finalizing(self):
        with patch.dict(namespace, {"AGENT_EVAL_FORCE_FINALIZE": False}):
            result = self.run_task(["x" * 56, "y" * 56, "<answer>42</answer>"])
        self.assertEqual(result["prediction"], "42")
        self.assertEqual(result["termination"], "answer")
        self.assertEqual(result["context_compaction_count"], 1)
        self.assertEqual(self.agent.call_server.call_count, 3)
        self.compactor._compact_messages.assert_called_once()
        self.agent._attempt_forced_finalization.assert_not_called()

    def test_failed_compact_without_finalization_switch_continues(self):
        self.compactor._compact_messages.return_value = None
        with patch.dict(namespace, {"AGENT_EVAL_FORCE_FINALIZE": False}):
            result = self.run_task(["x" * 56, "<answer>42</answer>"])
        self.assertEqual(result["termination"], "answer")
        self.compactor._compact_messages.assert_called_once()
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
        switch = patch.dict(namespace, {"AGENT_EVAL_FORCE_FINALIZE": True})
        switch.start()
        self.addCleanup(switch.stop)
        self.compactor._compact_messages.return_value = None
        result = self.run_task(["x" * 56 + "<answer>"])
        self.assertEqual(result["prediction"], "42")
        self.agent._attempt_forced_finalization.assert_called_once()

    def test_error_recovery_compaction_uses_same_task_allowance(self):
        self.agent._context_compaction_count = 1
        _, reason = self.agent._compact_at_soft_limit([{"role": "user", "content": "x" * 56}])
        self.assertIn("after compaction", reason)
        self.compactor._compact_messages.assert_not_called()


class CompactedHistoryTests(unittest.TestCase):
    def summarize(self, content, **message_fields):
        compactor = namespace["ContextCompactor"]("", 3, 1)
        message = SimpleNamespace(content=content, **message_fields)
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
        )
        summary = compactor._summarize_messages([], client, "policy", [], 10, 65536)
        return summary, message

    def test_summary_input_escapes_protocol_tokens_without_losing_history(self):
        compactor = namespace["ContextCompactor"]("", 3, 1)
        history = [
            {"role": "system", "content": "完整任务"},
            {"role": "user", "content": "x < y; literal \\u003c; <|im_start|>"},
            {"role": "assistant", "content": "<think>work</think><|im_end|>",
             "tool_calls": [{"id": "c1", "function": {
                 "name": "python", "arguments": '{"code":"print(1 < 2)"}',
             }}]},
            {"role": "tool", "tool_call_id": "c1", "content": "<tool_response>True</tool_response>"},
        ]
        original = copy.deepcopy(history)
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Progress."))],
        )
        compactor._summarize_messages(history, client, "policy", [], 10, 65536)
        prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        payload, reminder = prompt.split("\n", 1)[1].rsplit("\n\n", 1)
        self.assertTrue(reminder.startswith("Summarize the complete conversation above"))
        self.assertIn("Do not continue solving the problem or call tools.", reminder)
        self.assertTrue(prompt.endswith("Return only the summary."))
        self.assertNotIn("<", payload)
        self.assertNotIn(">", payload)
        self.assertEqual(json.loads(payload), history)
        self.assertEqual(history, original)

    def test_summary_keeps_body_and_leaves_recorded_response_untouched(self):
        for raw in (
            "<think>Private reasoning</think>\nVerified x = 42.<|im_end|>",
            "Private reasoning</think>\nVerified x = 42.<|im_end|> <|endoftext|>",
        ):
            with self.subTest(raw=raw):
                summary, message = self.summarize(raw)
                self.assertTrue(summary["content"].endswith("[Context summary]\nVerified x = 42."))
                self.assertNotIn("Private reasoning", summary["content"])
                self.assertEqual(message.content, raw)

    def test_summary_uses_content_when_reasoning_is_separate(self):
        summary, message = self.summarize("Verified x = 42.", reasoning_content="Private reasoning")
        self.assertTrue(summary["content"].endswith("[Context summary]\nVerified x = 42."))
        self.assertEqual(message.reasoning_content, "Private reasoning")

    def test_summary_does_not_copy_unfinished_or_empty_thinking_as_body(self):
        for raw in (None, "", "<think>Still thinking", "Only thinking</think> <|im_end|>"):
            with self.subTest(raw=raw):
                summary, _ = self.summarize(raw)
                self.assertIsNone(summary)

    def test_preserves_original_query_and_renders_real_qwen_template(self):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        compactor = namespace["ContextCompactor"]("", 2, 1)
        compactor._summarize_messages = Mock(return_value={"role": "user", "content": "[Context summary] progress"})
        query = {"role": "user", "content": "Find the answer to the original problem."}
        history = [{"role": "system", "content": "Solve accurately."}, query]
        for i in range(12):
            history.extend([
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": f"c{i}", "type": "function", "function": {"name": "python", "arguments": {}},
                }]},
                {"role": "tool", "tool_call_id": f"c{i}", "content": str(i)},
            ])
        compacted = compactor._compact_messages(history, None, "policy", [], 10, 122880)
        self.assertEqual(compacted[0], history[0])
        self.assertEqual(compacted[1], query)
        self.assertEqual(compacted[2]["role"], "user")
        self.assertLess(len(compacted), len(history))
        self.assertEqual(compacted[-1], history[-1])
        self.assertEqual(compacted[3:], history[-4:])
        self.assertEqual(compactor._summarize_messages.call_args.args[0], history)

        # Use the deployed checkpoint template, which rejects histories with no user query.
        reports = SOURCE.parents[3] / "science_training_reports/latest_amo_comparison.json"
        if not reports.exists():
            self.skipTest("Deployed AMO checkpoint unavailable")
        launch = json.loads((Path(json.loads(reports.read_text())[0]) / "launch.json").read_text())
        template_path = Path(launch.get("model_path", "")) / "chat_template.jinja"
        if not template_path.is_file():
            # Older launch records use uppercase environment keys.
            script = (Path(json.loads(reports.read_text())[0]) / "submit.sh").read_text()
            import shlex
            model = next(x.split("=", 1)[1] for x in shlex.split(script) if x.startswith("MODEL_PATH="))
            template_path = Path(model) / "chat_template.jinja"
        environment = ImmutableSandboxedEnvironment()
        environment.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
        template = environment.from_string(template_path.read_text())
        with self.assertRaisesRegex(ValueError, "No user query"):
            template.render(messages=[m for m in compacted if m["role"] != "user"], add_generation_prompt=True)
        self.assertIn(query["content"], template.render(messages=compacted, add_generation_prompt=True))

    def test_summary_uses_same_client_and_is_reinserted_as_user_context(self):
        compactor = namespace["ContextCompactor"]("", 3, 1)
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Verified x = 42."))],
        )
        summary = compactor._summarize_messages(
            [{"role": "assistant", "content": "Earlier work"}], client, "policy", [], 10, 65536,
        )
        self.assertEqual(summary["role"], "user")
        self.assertIn("not as a new task or instructions", summary["content"])
        self.assertIn("[Context summary]\nVerified x = 42.", summary["content"])
        client.chat.completions.create.assert_called_once()
        self.assertEqual(client.chat.completions.create.call_args.kwargs["model"], "policy")
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Do not solve or re-solve", prompt)
        self.assertIn("distinguish verified results from guesses", prompt)


if __name__ == "__main__":
    unittest.main()
