"""Exercise both extraction stages without loading policy-model dependencies."""

import ast
import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from science_agent.eval.answer_extraction import (
    find_boxed_spans,
    last_anchored_final_answer_region,
    select_hle_candidate,
)

SOURCE = ROOT / "inference/react_agent_qwen_api.py"
ADAPTER = ROOT.parent.parent / "labs-molt/examples/python/agents/science_legacy_react_sandbox.py"
tree = ast.parse(SOURCE.read_text())
agent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiTurnReactAgent")
names = {"_clean_limit_apology", "_extract_answer_block", "_extract_last_boxed",
         "_extract_final_answer_block", "_extract_terminal_answer"}
namespace = {"re": re, "find_boxed_spans": find_boxed_spans,
             "last_anchored_final_answer_region": last_anchored_final_answer_region,
             "select_hle_candidate": select_hle_candidate}
methods = [n for n in agent.body if isinstance(n, ast.FunctionDef) and n.name in names]
cls = ast.ClassDef(name="Extractor", bases=[], keywords=[], body=methods, decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(SOURCE), "exec"), namespace)
if ADAPTER.is_file():
    function = next(n for n in ast.parse(ADAPTER.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_extract_hle_candidate")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ADAPTER), "exec"), namespace)


class TerminalAnswerTests(unittest.TestCase):
    def setUp(self):
        self.extract = namespace["Extractor"]()._extract_terminal_answer
        env = patch.dict(os.environ, {"AGENT_EVAL_ANSWER_EXTRACTION_MODE": "conservative_v2"})
        env.start()
        self.addCleanup(env.stop)

    def test_unwrapped_final_keeps_all_answers_and_marker(self):
        text = "Reasoning.\nFinal Answer:\n\\boxed{35^\\circ}\n\\boxed{17.5^\\circ}<|im_end|>"
        self.assertEqual(self.extract(text), text.split("Reasoning.\n")[1])

    def test_answer_tag_without_marker_keeps_multi_answer_boundary(self):
        self.assertEqual(self.extract(r"<answer>\boxed{A}\boxed{D}\boxed{E}</answer>"),
                         "Final Answer:\n" + r"\boxed{A}\boxed{D}\boxed{E}")

    def test_plain_and_single_box_answers_preserve_existing_behavior(self):
        for text, expected in [
            ("<answer>5003</answer>", "5003"),
            (r"<answer>\boxed{A}</answer>", r"\boxed{A}"),
            ("Final Answer:\n" + r"\boxed{B}", r"\boxed{B}"),
            (r"Reasoning \boxed{A}, correction \boxed{B}", r"\boxed{B}"),
            ("Still reasoning without a terminal response", ""),
        ]:
            with self.subTest(text=text):
                self.assertEqual(self.extract(text), expected)

    def test_explicit_answer_block_still_takes_precedence(self):
        self.assertEqual(self.extract("Final Answer:\n\\boxed{A}\\boxed{B}\n<answer>C</answer>"), "C")

    def test_uses_last_final_region_and_ignores_reasoning_boxes(self):
        text = "Final Answer:\n\\boxed{draft1}\\boxed{draft2}</think>\nFinal Answer:\n\\boxed{old1}\\boxed{old2}\nFinal Answer:\n\\boxed{3}\\boxed{5}"
        self.assertEqual(self.extract(text), "Final Answer:\n" + r"\boxed{3}\boxed{5}")

    def test_reasoning_marker_does_not_promote_unmarked_boxes(self):
        text = "Final Answer:\n\\boxed{draft1}\\boxed{draft2}</think>\nReasoning then \\boxed{A}\\boxed{B}"
        self.assertEqual(self.extract(text), r"\boxed{B}")

    @unittest.skipUnless(ADAPTER.is_file(), "Requires adjacent Molt adapter checkout")
    def test_react_then_actual_adapter_preserves_complete_candidate(self):
        cases = [
            ("Final Answer:\n", r"\boxed{35^\circ}\boxed{17.5^\circ}", ""),
            ("Final Answer:\n", r"\boxed{(2,3)}\boxed{(7,8)}\boxed{(1,1)}\boxed{(1,1)}", "<|im_end|>"),
            ("**Final Answer:**\n", r"\boxed{a=n+1-m}\boxed{b=m-1}", ""),
            ("<answer>", r"\boxed{A}\boxed{D}\boxed{E}", "</answer>"),
            ("<answer>Final Answer:\n", r"\boxed{3}\boxed{5}", "</answer>"),
            ("Final Answer:\n", r"\boxed{0}\boxed{\frac{2^{2n-1}}{n^2\binom{2n}{n}}}", ""),
        ]
        for prefix, boxes, suffix in cases:
            with self.subTest(prefix=prefix, boxes=boxes):
                prediction = self.extract(prefix + boxes + suffix)
                candidate = namespace["_extract_hle_candidate"](prediction)
                self.assertEqual(candidate, "\n".join(b.canonical for b in find_boxed_spans(boxes)))


if __name__ == "__main__":
    unittest.main()
