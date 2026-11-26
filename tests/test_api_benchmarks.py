"""Offline protocol checks: no model imports, API calls or GPU use."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_hle_api as evaluation
from science_agent.eval.api_benchmarks import extract_prediction


class BenchmarkAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, benchmark, raw):
        source = self.root / f'{benchmark}.jsonl'
        source.write_text(json.dumps(raw) + '\n')
        return evaluation.parse_args(['--benchmark', benchmark, '--input', str(source),
                                      '--expected-rows', '1', '--run-dir', str(self.root / benchmark),
                                      '--stage', 'prepare'])

    def test_imo_uses_original_prompt_and_all_references(self):
        args = self.args('imo_answerbench', {'id': 'one', 'question': 'Q', 'answer': ['01', '1'],
                                            'prompt': [{'role': 'user', 'content': 'Original IMO prompt'}]})
        row = evaluation.load_rows(args)[0]
        self.assertEqual(row['question'], 'Original IMO prompt')
        self.assertEqual(json.loads(row['answer']), ['01', '1'])
        self.assertEqual(args.dataset_base_dir, args.input.parent)

    def test_fso_preserves_multi_part_reference_and_units(self):
        args = self.args('fso', {'id': 'one', 'question': 'Q', 'answer': ['1 m', '2 s']})
        row = evaluation.load_rows(args)[0]
        self.assertEqual(row['answer'], '1 m\n2 s')
        self.assertIn('Keep labels and units clear.', row['question'])
        text = r'Hidden</think><answer>Length: \boxed{1} m; time: \boxed{2} s</answer>'
        self.assertEqual(extract_prediction('fso', text), text.split('</think>')[1])
        self.assertIn('unit conversions', evaluation.judge_messages(row, '1 m', 'fso')[1]['content'])

    def test_math_keeps_all_boxes_but_cmt_keeps_full_response(self):
        text = r'Ignore \boxed{9}</think>First \boxed{1}, second \boxed{2}.'
        self.assertEqual(extract_prediction('amobench', text), '\\boxed{1}\n\\boxed{2}')
        self.assertEqual(extract_prediction('imo_answerbench', text), '\\boxed{1}\n\\boxed{2}')
        self.assertEqual(extract_prediction('cmt_benchmark', text), text.split('</think>')[1])

    def test_cmt_parameters_functions_and_image_rejection(self):
        raw = {'id': 'one', 'question': 'Q', 'answer': ['2'], 'parameters': 'a=1', 'functions': 'f(x)=x'}
        args = self.args('cmt_benchmark', raw)
        row = evaluation.load_rows(args)[0]
        self.assertIn('Parameters (if any):\na=1', row['question'])
        self.assertIn('Functions (if any):\nf(x)=x', row['question'])
        args = self.args('cmt_benchmark', {**raw, 'image_question': True})
        with self.assertRaisesRegex(ValueError, 'Image input'):
            evaluation.load_rows(args)

    def test_duplicate_validation_before_limit(self):
        args = self.args('amobench', {'id': 'one', 'question': 'Q', 'answer': ['1']})
        args.input.write_text(args.input.read_text() * 2)
        args.expected_rows, args.limit = 2, 1
        with self.assertRaisesRegex(ValueError, 'unique'):
            evaluation.load_rows(args)

    def test_prepare_requires_no_solver_credentials(self):
        args = self.args('amobench', {'id': 'one', 'question': 'Q', 'answer': ['1']})
        with patch.dict(os.environ, {}, clear=True):
            evaluation.main(['--benchmark', 'amobench', '--input', str(args.input),
                             '--expected-rows', '1', '--run-dir', str(args.run_dir), '--stage', 'prepare'])
        self.assertEqual(len(json.loads((args.run_dir / 'dataset.json').read_text())), 1)
        self.assertNotIn('torch', sys.modules)

    def test_api_roundtrip_uses_math_protocol_and_resume(self):
        args = self.args('amobench', {'id': 'one', 'question': 'Q', 'answer': ['secret-reference']})
        args.stage, args.model = 'all', 'api-model'
        row = evaluation.load_rows(args)[0]
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(evaluation.configure(args)['AGENT_EVAL_FORCE_FINALIZE'], '0')
        agent = SimpleNamespace(_run=Mock(return_value={'prediction': r'work</think>\boxed{1} and \boxed{2}',
                                                       'messages': [], 'truncated': False}),
                                last_messages=[], context_tokens_estimated=True)
        calls = []
        async def judge(messages, **kwargs):
            calls.append(messages)
            self.assertFalse(kwargs['require_score'])
            return {'judge_result': {'correct': True, 'score': 1}}
        for _ in range(2):
            asyncio.run(evaluation.run_question(row, args, lambda: agent, judge, []))
        agent._run.assert_called_once()
        item = agent._run.call_args.args[0]['item']
        self.assertNotIn('answer', item)
        self.assertNotIn('secret-reference', json.dumps(item))
        self.assertEqual(len(calls), 1)
        self.assertIn('mathematical competition answers', calls[0][0]['content'])
        grade = json.loads(evaluation.artifact(args, 'grades', row).read_text())
        self.assertEqual(grade['prediction'], '\\boxed{1}\n\\boxed{2}')
        self.assertEqual(evaluation.summarize([row], args)['pass1_overall'], 1)


if __name__ == '__main__':
    unittest.main()
