"""Check the response/replay boundary without loading policy dependencies."""

import ast
import copy
import __future__
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[1] / "inference/react_agent_qwen_api.py"
tree = ast.parse(SOURCE.read_text())
agent_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MultiTurnReactAgent")
namespace = {
    "re": re, "copy": copy, "json": json, "os": os,
    "safe_strip": lambda value: (value or "").strip(),
    "_parse_served_model_fallbacks": lambda *_: (),
    "get_env_float": lambda *_, default: default,
    "get_env_int": lambda *_, default: default,
    "get_env_bool": lambda *_, default: default,
    "TASK_TIMEOUT_SEC": 600,
}
for node in agent_class.body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "__init__", "_extract_response_text_fields", "_build_assistant_request_message",
    }:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)


class ResponseReplayTests(unittest.TestCase):
    def setUp(self):
        namespace["AGENT_OBSERVATION_MASKING"] = True
        self.agent = SimpleNamespace(
            _extract_reasoning=lambda message: getattr(message, "reasoning_content", "") or "",
            _deepseek_v4_response_enabled=lambda: False,
        )
        namespace["__init__"](self.agent, llm={"model": "qwen3.5", "served_model_name": "policy", "generate_cfg": {}})

    def test_checkpoint_config_controls_replay_independently_of_directory_name(self):
        text = "Compute</think><tool_call><function=python></function></tool_call><|im_end|>"
        with tempfile.TemporaryDirectory() as tmp:
            for directory, model_type, expected in [
                ("global_step300/model", "qwen3_5_moe", ("Compute", "<tool_call><function=python></function></tool_call>")),
                ("qwen-named-other-model", "llama", ("", text)),
            ]:
                with self.subTest(model_type=model_type):
                    path = Path(tmp) / directory
                    path.mkdir(parents=True)
                    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
                    namespace["__init__"](self.agent, llm={"model": str(path), "served_model_name": "policy", "generate_cfg": {}})
                    fields = namespace["_extract_response_text_fields"](self.agent, SimpleNamespace(content=text))
                    self.assertEqual(fields, expected)
                    replay = namespace["_build_assistant_request_message"](self.agent, *fields, None)
                    self.assertEqual(replay["content"], expected[1])

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
        namespace["__init__"](self.agent, llm={"model": "other-model", "generate_cfg": {}})
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
