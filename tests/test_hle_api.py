"""API HLE tests without network, GPU, Qwen Agent or Molt dependencies."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('eval_hle_api', ROOT / 'eval_hle_api.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class HLEAPITests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / 'hle.jsonl'
        self.source.write_text(json.dumps({'id': 'one', 'question': 'What is 6*7?', 'answer': '42'}) + '\n')
        self.args = evaluation.parse_args(['--run-dir', str(self.root / 'run'), '--stage', 'summary',
                                          '--input', str(self.source), '--expected-rows', '1'])
        self.args.stage = 'all'
        self.args.model = 'DeepSeek-V4.1-Flash-1017'
        self.args.base_url = 'http://solver/v1'
        self.row = evaluation.load_rows(self.args)[0]
        self.agent = SimpleNamespace(
            _run=Mock(return_value={'prediction': r'\boxed{42}', 'messages': [
                {'role': 'assistant', 'tool_calls': [{'id': 'call1'}]}],
                'truncated': False, 'termination': 'answer', 'completion_tokens': 12}),
            last_messages=[], context_tokens_estimated=True)
        self.create = Mock(return_value=self.agent)
        self.judge_calls = []

    async def judge(self, messages, **kwargs):
        self.judge_calls.append(messages)
        return {'judge_result': {'correct': False, 'score': 0}}

    def run_question(self, judge=None):
        return asyncio.run(evaluation.run_question(self.row, self.args, self.create, judge or self.judge, ['secret-key']))

    def grade(self):
        return json.loads(evaluation.artifact(self.args, 'grades', self.row).read_text())

    def test_defaults(self):
        self.assertEqual((self.args.workers, self.args.temperature, self.args.top_p), (16, 1.0, .95))
        self.assertEqual((self.args.max_context_tokens, self.args.soft_context_tokens), (122880, 110592))
        with patch.dict(os.environ, {}, clear=True):
            settings = evaluation.configure(self.args)
            self.assertEqual(settings['AGENT_CONTEXT_COMPACTION'], '0')
            self.assertEqual(settings['AGENT_EVAL_FORCE_FINALIZE'], '1')
            self.assertEqual(settings['AGENT_ANSWER_TYPE'], 'boxed')

    def test_references_not_sent_to_solver(self):
        self.run_question()
        self.assertNotIn('answer', self.agent._run.call_args.args[0]['item'])
        self.assertEqual(self.judge_calls[0], evaluation.judge_messages(self.row, r'\boxed{42}'))
        self.assertIn('Ground truth answer:\n42', self.judge_calls[0][1]['content'])

    def test_wrong_grade_cached(self):
        self.run_question()
        self.run_question()
        self.create.assert_called_once()
        self.assertEqual(len(self.judge_calls), 1)
        self.assertEqual(self.grade()['score'], 0)
        self.assertEqual(self.grade()['status'], 'graded')

    def test_rollout_saved_before_judge(self):
        async def check(messages, **kwargs):
            self.assertTrue(evaluation.artifact(self.args, 'rollouts', self.row).exists())
            return await self.judge(messages, **kwargs)
        self.run_question(check)

    def test_judge_only_does_not_regenerate(self):
        self.args.stage = 'rollout'
        self.run_question()
        self.assertFalse(evaluation.artifact(self.args, 'grades', self.row).exists())
        self.args.stage = 'judge'
        self.run_question()
        self.create.assert_called_once()
        self.assertEqual(len(self.judge_calls), 1)

    def test_invalid_judge_retries_saved_answer(self):
        async def invalid(*args, **kwargs):
            return {'judge_result': None, 'judge_error': 'upstream failed'}
        self.run_question(invalid)
        self.assertEqual(self.grade()['score'], -1)
        self.args.retry_failed = True
        self.args.stage = 'judge'
        self.run_question()
        self.create.assert_called_once()
        self.assertEqual(self.grade()['status'], 'graded')

    def test_empty_skips_judge(self):
        self.agent._run.return_value['prediction'] = ''
        self.run_question()
        self.assertFalse(self.judge_calls)
        self.assertEqual(self.grade()['status'], 'empty_answer')

    def test_solver_error_saved_and_redacted(self):
        self.agent._run.side_effect = RuntimeError('Authorization: secret-key')
        self.run_question()
        result = json.loads(evaluation.artifact(self.args, 'rollouts', self.row).read_text())
        self.assertIn('<redacted>', result['error'])
        self.assertNotIn('secret-key', result['error'])
        self.assertFalse(self.judge_calls)

    def test_summary(self):
        self.run_question()
        summary = evaluation.summarize([self.row, {**self.row, 'question_id': 'two'}], self.args)
        self.assertEqual((summary['completed'], summary['pending'], summary['correct']), (1, 1, 0))
        self.assertEqual(summary['judge_errors'], 0)
        self.assertEqual(summary['tool_calls'], 1)
        self.assertEqual(summary['completion_tokens'], 12)

    def test_duplicates_checked_before_limit(self):
        self.source.write_text(self.source.read_text() * 2)
        self.args.expected_rows = 2
        self.args.limit = 1
        with self.assertRaises(ValueError):
            evaluation.load_rows(self.args)

    def test_core_overrides_rejected(self):
        with self.assertRaises(SystemExit):
            evaluation.parse_args(['--run-dir', str(self.root), '--stage', 'summary',
                                  '--request-extra-body', '{"messages": []}'])

    def test_plain_candidate(self):
        self.assertEqual(evaluation.hle_candidate('5003'), '5003')
        self.assertEqual(evaluation.hle_candidate('Your final answer'), '')

    def test_budget_and_integer_effort(self):
        class Base:
            def __init__(self, **kwargs): self._context_tokenizer = None
            def count_tokens(self, messages, tool_schemas=None): return 110000
            def call_server(self, messages, planning_port, **kwargs): return kwargs, {}
        self.args.request_extra_body = {'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 20}}
        with patch.dict(sys.modules, react_agent_openai_api_tools=SimpleNamespace(MultiTurnReactAgent=Base)):
            agent = evaluation.make_agent_factory(self.args)()
        response, _ = agent.call_server([], None, tool_schemas=[])
        self.assertEqual(response['max_output_tokens'], 12879)
        self.assertEqual(response['request_extra_body']['chat_template_kwargs']['reasoning_effort'], 20)
        response, _ = agent.call_server([], None, request_extra_body={'chat_template_kwargs': {'thinking': False}})
        self.assertFalse(response['request_extra_body']['chat_template_kwargs']['thinking'])
        self.assertEqual(agent._get_api_base(), 'http://solver/v1')
        self.assertEqual(agent._deepseek_v4_profile_model_name('deepseek-v4-flash-max'), '')

    def test_full_entry_scope_cleanup_manifest_and_resume(self):
        sys.path.insert(0, str(ROOT / 'inference'))
        import sandbox_gateway as gateway
        runs, solve_runs, judge_runs = [], [], []
        class Run:
            def __init__(self):
                self.closed = False
                runs.append(self)
            def close(self):
                self.closed = True
        result = self.agent._run.return_value
        def solve(*args):
            solve_runs.append(gateway.shared_sandbox_run())
            return dict(result)
        async def judge(*args, **kwargs):
            judge_runs.append(gateway.shared_sandbox_run())
            return {'judge_result': {'correct': True, 'score': 1}}
        self.agent._run.side_effect = solve
        chain = [{'name': 'test', 'transport': 'direct', 'model': 'test', 'api_key': 'judge-secret'}]
        argv = ['--run-dir', str(self.args.run_dir), '--input', str(self.source), '--expected-rows', '1',
                '--base-url', 'http://solver/v1', '--model', 'test']
        async def inline(func, *args, **kwargs):
            return func(*args, **kwargs)
        with patch.object(asyncio, 'to_thread', inline), \
             patch.dict(os.environ, {'HLE_API_KEY': 'solver-secret'}), \
             patch.dict(sys.modules, judge_client=SimpleNamespace(judge_answer=judge, load_judge_chain=lambda **kw: chain)), \
             patch.object(gateway, 'AgentSandboxRun', Run), \
             patch.object(evaluation, 'make_agent_factory', return_value=self.create):
            evaluation.main(argv)
            evaluation.main(argv)
        self.assertEqual(solve_runs, judge_runs)
        self.assertEqual(len(solve_runs), 1)
        self.assertTrue(all(r.closed for r in runs))
        manifest = (self.args.run_dir / 'manifest.json').read_text()
        self.assertNotIn('solver-secret', manifest)
        self.assertNotIn('judge-secret', manifest)
        self.assertEqual(json.loads((self.args.run_dir / 'summary.json').read_text())['correct'], 1)

    def test_images_preserved_without_qwen_processor(self):
        class Base:
            def __init__(self, **kwargs): self._context_tokenizer = None
            def count_tokens(self, messages, tool_schemas=None):
                self.counted_messages = messages
                return 100
        with patch.dict(sys.modules, react_agent_openai_api_tools=SimpleNamespace(MultiTurnReactAgent=Base)):
            agent = evaluation.make_agent_factory(self.args)()
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Q'},
                     {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}}]}]
        self.assertEqual(agent.count_tokens(messages), 4196)
        self.assertTrue(agent.image_context_estimated)
        self.assertEqual(len(messages[0]['content']), 2)
        self.assertEqual(len(agent.counted_messages[0]['content']), 1)


if __name__ == '__main__':
    unittest.main()
