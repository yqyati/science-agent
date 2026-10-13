"""Check budget accounting without importing model or tool backends."""

import ast
import copy
import hashlib
import json
import unittest
from collections import UserDict
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "inference/react_agent_qwen_api.py"
tree = ast.parse(SOURCE.read_text())
nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and
         node.name in {"count_tokens", "_count_image_tokens", "estimate_request_input_tokens", "_json_for_token_estimate"}]
namespace = {"copy": copy, "hashlib": hashlib, "json": json, "Mapping": Mapping, "_get_token_encoding": lambda: None}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)


class ContextTokenTests(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Solve the problem in this image."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "abcd" * 100000}},
            ]},
            {"role": "assistant", "content": "", "tool_calls": [{
                "type": "function", "id": "call-1",
                "function": {"name": "python", "arguments": '{"code":"print(1+1)"}'},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": "2"},
        ]
        self.tools = [{"type": "function", "function": {"name": "python"}}]
        self.agent = SimpleNamespace(
            _context_tokenizer_unavailable=False,
            _context_tokenizer=Mock(),
            _active_tool_schemas_for_current_task=lambda: self.tools,
            _count_image_tokens=Mock(return_value=0),
            _context_processor=None,
            _context_image_tokens={},
            llm_local_path="local-qwen-model",
        )

    def test_qwen_argument_mapping_and_batch_encoding_preserve_api_messages(self):
        original = copy.deepcopy(self.messages)

        def template(messages, **kwargs):
            arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
            self.assertEqual(dict(arguments.items()), {"code": "print(1+1)"})
            self.assertEqual(kwargs["tools"], self.tools)
            return UserDict(input_ids=list(range(137)), attention_mask=[1] * 137)

        self.agent._context_tokenizer.apply_chat_template.side_effect = template
        with patch.dict(namespace, {"estimate_request_input_tokens": Mock(side_effect=AssertionError("unexpected fallback"))}):
            self.assertEqual(namespace["count_tokens"](self.agent, self.messages), 137)
        self.assertEqual(self.messages, original)

    def test_existing_mapping_and_token_id_list_remain_supported(self):
        self.messages[1]["tool_calls"][0]["function"]["arguments"] = {"code": "print(1+1)"}
        self.agent._context_tokenizer.apply_chat_template.return_value = list(range(93))
        self.assertEqual(namespace["count_tokens"](self.agent, self.messages), 93)

    def test_explicit_empty_tools_override_the_active_tool_set(self):
        self.agent._context_tokenizer.apply_chat_template.return_value = list(range(93))
        self.assertEqual(namespace["count_tokens"](self.agent, self.messages, []), 93)
        self.assertNotIn("tools", self.agent._context_tokenizer.apply_chat_template.call_args.kwargs)

    def test_fallback_ignores_image_encoding_length_but_keeps_text_and_tools(self):
        original = copy.deepcopy(self.messages)
        estimate = namespace["estimate_request_input_tokens"]
        large = estimate(self.messages, self.tools)
        small_messages = copy.deepcopy(self.messages)
        small_messages[0]["content"][1]["image_url"]["url"] = "data:image/png;base64,YQ=="
        self.assertEqual(large, estimate(small_messages, self.tools))
        self.assertLess(large, 1000)
        self.assertGreater(large, estimate(self.messages))
        small_messages[0]["content"][0]["text"] += " extra evidence" * 100
        self.assertGreater(estimate(small_messages, self.tools), large)
        self.assertEqual(self.messages, original)

    def test_token_encoding_fallback_receives_no_image_payload(self):
        seen = []

        def encode(text):
            seen.append(json.loads(text))
            return list(range(71))

        with patch.dict(namespace, {"_get_token_encoding": lambda: SimpleNamespace(encode=encode)}):
            self.assertEqual(namespace["estimate_request_input_tokens"](self.messages, self.tools), 71)
        self.assertEqual(seen[0]["messages"][0]["content"][1]["image_url"]["url"], "<image>")
        self.assertEqual(seen[0]["tools"], self.tools)
        self.assertEqual(seen[0]["messages"][2]["content"], "2")

    def test_unavailable_or_failing_tokenizer_cannot_exhaust_budget_with_base64(self):
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable):
                self.agent._context_tokenizer_unavailable = unavailable
                self.agent._context_tokenizer = None if unavailable else Mock()
                if not unavailable:
                    self.agent._context_tokenizer.apply_chat_template.side_effect = TypeError("invalid template")
                self.assertLess(namespace["count_tokens"](self.agent, self.messages), 1000)

    def test_malformed_tool_arguments_fall_back_without_mutating_messages(self):
        self.messages[1]["tool_calls"][0]["function"]["arguments"] = "{invalid json"
        original = copy.deepcopy(self.messages)
        self.assertLess(namespace["count_tokens"](self.agent, self.messages), 1000)
        self.agent._context_tokenizer.apply_chat_template.assert_not_called()
        self.assertEqual(self.messages, original)

    def test_vision_tokens_replace_placeholders_instead_of_double_counting(self):
        self.agent._count_image_tokens.return_value = 384
        self.agent._context_processor = SimpleNamespace(image_token_id=42)
        self.agent._context_tokenizer.apply_chat_template.return_value = UserDict(input_ids=[[0, 42, 3, 42, 5]])
        self.assertEqual(namespace["count_tokens"](self.agent, self.messages), 387)

    def test_fallback_still_includes_computed_vision_tokens(self):
        self.agent._count_image_tokens.return_value = 384
        self.agent._context_tokenizer = None
        self.agent._context_tokenizer_unavailable = True
        expected = namespace["estimate_request_input_tokens"](self.messages, self.tools) + 384
        self.assertEqual(namespace["count_tokens"](self.agent, self.messages), expected)

    def test_image_cache_avoids_reloading_but_counts_each_present_image(self):
        processor = SimpleNamespace(merge_size=2, get_number_of_image_patches=Mock(return_value=32))
        factory = Mock(return_value=SimpleNamespace(image_processor=processor, image_token_id=42))
        load_image = Mock(return_value=nullcontext(SimpleNamespace(height=128, width=64)))
        modules = {
            "transformers": SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=factory)),
            "transformers.image_utils": SimpleNamespace(load_image=load_image),
        }
        with patch.dict("sys.modules", modules):
            count = namespace["_count_image_tokens"]
            self.assertEqual(count(self.agent, self.messages), 8)
            self.assertEqual(count(self.agent, self.messages), 8)
            self.assertEqual(count(self.agent, [self.messages[0], self.messages[0]]), 16)
            self.assertEqual(count(self.agent, self.messages[1:]), 0)
        factory.assert_called_once()
        load_image.assert_called_once()
        processor.get_number_of_image_patches.assert_called_once_with(128, 64, {})

    def test_image_count_failure_cannot_silently_use_a_text_only_budget(self):
        self.agent._count_image_tokens.side_effect = ValueError("image cannot be counted")
        with self.assertRaisesRegex(ValueError, "image cannot be counted"):
            namespace["count_tokens"](self.agent, self.messages)
        self.agent._context_tokenizer.apply_chat_template.assert_not_called()


if __name__ == "__main__":
    unittest.main()
