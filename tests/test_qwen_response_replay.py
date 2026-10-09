"""Check the response/replay boundary without loading policy dependencies."""

import ast
import copy
import re
import unittest
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[1] / "inference/react_agent_qwen_api.py"
tree = ast.parse(SOURCE.read_text())
agent_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MultiTurnReactAgent")
namespace = {"re": re, "copy": copy, "Dict": dict, "safe_strip": lambda value: (value or "").strip()}
for node in agent_class.body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "_extract_response_text_fields", "_build_assistant_request_message",
    }:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)


class ResponseReplayTests(unittest.TestCase):
    def setUp(self):
        namespace["AGENT_OBSERVATION_MASKING"] = True
        self.agent = SimpleNamespace(
            llm_local_path="qwen3.5", served_model_name="policy",
            _extract_reasoning=lambda message: getattr(message, "reasoning_content", "") or "",
            _deepseek_v4_response_enabled=lambda: False,
        )

    def test_raw_qwen_completion_splits_thought_and_removes_framing(self):
        message = SimpleNamespace(content="Check 2+2.\n</think>\n<answer>4</answer><|im_end|>")
        fields = namespace["_extract_response_text_fields"](self.agent, message)
        self.assertEqual(fields, ("Check 2+2.", "<answer>4</answer>"))
        replay = namespace["_build_assistant_request_message"](self.agent, *fields, None)
        self.assertEqual(replay["reasoning_content"], "Check 2+2.")
        self.assertEqual(replay["content"], "<answer>4</answer>")

    def test_explicit_think_open_and_tool_call_survive(self):
        tool = "<tool_call><function=python></function></tool_call>"
        fields = namespace["_extract_response_text_fields"](
            self.agent, SimpleNamespace(content=f"<think>Compute</think>{tool}<|im_end|>"))
        self.assertEqual(fields, ("Compute", tool))

    def test_separate_reasoning_is_preserved(self):
        fields = namespace["_extract_response_text_fields"](
            self.agent, SimpleNamespace(reasoning_content="Compute", content="4<|im_end|>"))
        self.assertEqual(fields, ("Compute", "4"))

    def test_empty_reasoning_does_not_override_template_fallback(self):
        replay = namespace["_build_assistant_request_message"](self.agent, "", "4", None)
        self.assertNotIn("reasoning_content", replay)

    def test_other_models_keep_their_response_protocol(self):
        self.agent.llm_local_path = "other-model"
        text = "literal </think> text<|im_end|>"
        self.assertEqual(namespace["_extract_response_text_fields"](
            self.agent, SimpleNamespace(content=text)), ("", text))

    def test_deepseek_replay_keeps_required_empty_reasoning(self):
        self.agent._deepseek_v4_response_enabled = lambda: True
        replay = namespace["_build_assistant_request_message"](self.agent, "", "4", None)
        self.assertEqual(replay["reasoning_content"], "")

    def test_non_mask_response_and_replay_keep_original_behavior(self):
        namespace["AGENT_OBSERVATION_MASKING"] = False
        text = "Compute</think><answer>4</answer><|im_end|>"
        fields = namespace["_extract_response_text_fields"](
            self.agent, SimpleNamespace(content=text))
        self.assertEqual(fields, ("", text))
        replay = namespace["_build_assistant_request_message"](self.agent, *fields, None)
        self.assertEqual(replay["content"], text)
        self.assertEqual(replay["reasoning_content"], "")


if __name__ == "__main__":
    unittest.main()
