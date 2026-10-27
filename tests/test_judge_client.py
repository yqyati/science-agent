"""Exercise ordered Judge routing, retries and recovery without network or GPUs."""
import ast
import asyncio
import json
import logging
import math
import tempfile
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class SandboxGatewayError(RuntimeError):
    pass


class SandboxStateError(SandboxGatewayError):
    pass


class JudgeThinkingFallbackTests(unittest.TestCase):
    def run_judge(self, responses, *, protocol=True, thinking="enabled", recovery=False, retries=(), direct_fallback=False, chain=None):
        source = Path(__file__).resolve().parents[1] / "inference/judge_client.py"
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        namespace = dict(asyncio=asyncio, json=json, os=os, math=math, Path=Path,
                         logger=logging.getLogger(__name__))
        if chain is None:
            chain = [{"name": "primary", "transport": "sandbox", "base_url": "http://test",
                      "model": "test", "api_key_env": "API_KEY", "retry_delays": list(retries),
                      "sandbox_replacements": int(recovery)}]
            if direct_fallback:
                chain.append({"name": "fallback", "transport": "direct", "base_url": "http://direct/v1",
                              "model": "DeepSeek-V4.1-Flash-1017"})
        calls = []
        sandbox_calls, closed = [], []
        instances = []
        direct_options = []

        class DirectClient:
            def __init__(self, **kwargs):
                direct_options.append(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, *, json, headers):
                calls.append(__import__("json").loads(__import__("json").dumps(json)))
                direct_options[-1].update(url=url, headers=headers)
                response = responses[len(calls) - 1]
                if isinstance(response, Exception):
                    raise response
                return SimpleNamespace(status_code=response["status_code"], text=response["text"])

        class Run:
            def __init__(self):
                self.index = len(instances)
                instances.append(self)

            def run_http(self, request, timeout):
                calls.append(json.loads(json.dumps(request["json"])))
                sandbox_calls.append(self.index)
                response = responses[len(calls) - 1]
                if isinstance(response, Exception):
                    raise response
                return response

            def close(self):
                closed.append(self.index)

        namespace["initial_run"] = Run()
        namespace.update(AgentSandboxRun=Run, SandboxGatewayError=SandboxGatewayError, SandboxStateError=SandboxStateError,
                         httpx=SimpleNamespace(AsyncClient=DirectClient), shared_sandbox_run=lambda: namespace["initial_run"])
        exec(compile(ast.Module(nodes, type_ignores=[]), str(source), "exec"), namespace)

        async def run_inline(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "judges.json"
            config.write_text(json.dumps(chain))
            with patch.dict(os.environ, SCIENCE_AGENT_JUDGE_CONFIG=str(config),
                            SCIENCE_AGENT_JUDGE_THINKING=thinking, API_KEY="test"), \
                    patch.object(asyncio, "to_thread", run_inline):
                result = asyncio.run(namespace["judge_answer"](
                    [{"role": "user", "content": "candidate"}], require_score=not protocol))
        result.update(sandbox_calls=sandbox_calls, closed=closed, direct_options=direct_options)
        return result, calls

    @staticmethod
    def response(content, *, status=200):
        return {"status_code": status, "text": json.dumps({"choices": [
            {"finish_reason": "stop", "message": {"content": content}}]})}

    def test_valid_false_is_not_a_fallback(self):
        result, calls = self.run_judge([self.response('{"correct": false}')])
        self.assertEqual(result["judge_result"]["score"], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["judge_thinking_fallbacks"], 0)

    def test_invalid_output_falls_back_once_in_both_protocols(self):
        for protocol in (True, False):
            for invalid in ('', 'not json', '[]', '{"correct": "false"}'):
                with self.subTest(protocol=protocol, invalid=invalid):
                    result, calls = self.run_judge([self.response(invalid),
                        self.response('{"correct": true, "score": 1}')], protocol=protocol)
                    self.assertEqual(result["judge_result"]["score"], 1)
                    self.assertEqual([r["thinking"]["type"] for r in calls], ["enabled", "disabled"])
                    self.assertEqual(result["judge_attempts"], 2)
                    self.assertEqual(result["judge_thinking_fallbacks"], 1)

    def test_failed_fallback_remains_ungraded(self):
        result, calls = self.run_judge([self.response(''), self.response('')])
        self.assertIsNone(result["judge_result"])
        self.assertEqual(len(calls), 2)
        self.assertIn('JSONDecodeError', result["judge_error"])

    def test_transport_error_does_not_trigger_thinking_fallback(self):
        result, calls = self.run_judge([self.response('', status=503)])
        self.assertIsNone(result["judge_result"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["judge_thinking_fallbacks"], 0)

    def test_disabled_thinking_has_no_second_attempt(self):
        result, calls = self.run_judge([self.response('')], thinking="disabled")
        self.assertIsNone(result["judge_result"])
        self.assertEqual(len(calls), 1)

    def test_pending_recovery_switches_sandbox_without_retrying_notice(self):
        result, calls = self.run_judge([SandboxStateError('pending recovery notice'),
            self.response('{"correct": true}')], recovery=True, retries=(0,) * 6)
        self.assertEqual(result['sandbox_calls'], [0, 1])
        self.assertEqual(result['closed'], [1])
        self.assertEqual(result['judge_result']['score'], 1)
        self.assertEqual([r['messages'] for r in calls], [calls[0]['messages']] * 2)

    def test_two_connection_failures_switch_sandbox(self):
        error = SandboxGatewayError('proxy HTTP 502 (attempt 3/3): Sandbox upstream connection failed')
        result, _ = self.run_judge([error, error, self.response('{"correct": false}')],
                                   recovery=True, retries=(0,) * 6)
        self.assertEqual(result['sandbox_calls'], [0, 0, 1])
        self.assertEqual(result['closed'], [1])
        self.assertEqual(result['judge_result']['score'], 0)

    def test_api_error_breaks_connection_failure_streak(self):
        error = SandboxGatewayError('proxy HTTP 502: Sandbox upstream connection failed')
        result, _ = self.run_judge([error, self.response('', status=503), error,
            self.response('{"correct": true}')], recovery=True, retries=(0,) * 6)
        self.assertEqual(result['sandbox_calls'], [0, 0, 0, 0])
        self.assertEqual(result['judge_result']['score'], 1)

    def test_unusable_replacements_are_closed_and_remain_ungraded(self):
        result, _ = self.run_judge([SandboxStateError('pending notice')] * 2,
                                   recovery=True, retries=(0,) * 6)
        self.assertEqual(result['sandbox_calls'], [0, 1])
        self.assertEqual(result['closed'], [1])
        self.assertIsNone(result['judge_result'])

    def test_direct_fallback_runs_only_after_all_sandboxes_fail(self):
        for protocol, correct in ((True, False), (False, True)):
            with self.subTest(protocol=protocol):
                result, calls = self.run_judge([SandboxStateError('pending notice')] * 2 + [
                    self.response(json.dumps({'correct': correct, 'score': int(correct)}))],
                    protocol=protocol, recovery=True, direct_fallback=True)
                self.assertEqual(result['sandbox_calls'], [0, 1])
                self.assertEqual(result['closed'], [1])
                self.assertEqual(result['judge_backend'], 'direct_fallback')
                self.assertEqual(result['judge_result']['score'], int(correct))
                self.assertEqual(result['judge_error'], '')
                self.assertIn('pending notice', result['judge_primary_error'])
                self.assertEqual(calls[-1]['model'], 'DeepSeek-V4.1-Flash-1017')
                self.assertEqual(calls[-1]['chat_template_kwargs'], {'thinking': True, 'reasoning_effort': 20})
                self.assertNotIn('thinking', calls[-1])
                self.assertEqual([r['messages'] for r in calls], [calls[0]['messages']] * 3)
                options = result['direct_options'][0]
                self.assertFalse(options['trust_env'])
                self.assertEqual(options['headers'], {'Authorization': 'Bearer EMPTY'})
                self.assertEqual(options['url'], 'http://direct/v1/chat/completions')

    def test_valid_sandbox_result_never_calls_direct_fallback(self):
        result, calls = self.run_judge([self.response('{"correct": false}')], recovery=True, direct_fallback=True)
        self.assertEqual(result['judge_result']['score'], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['direct_options'], [])

    def test_direct_invalid_thinking_output_retries_without_thinking(self):
        result, calls = self.run_judge([SandboxStateError('pending notice')] * 2 + [
            self.response(''), self.response('{"correct": true}')], recovery=True, direct_fallback=True)
        self.assertEqual(result['judge_result']['score'], 1)
        self.assertEqual(calls[-1]['chat_template_kwargs'], {'thinking': False})

    def test_direct_failure_keeps_answer_ungraded(self):
        for response in (TimeoutError('direct timeout'), self.response('', status=503), self.response('')):
            with self.subTest(response=response):
                replies = [SandboxStateError('pending notice')] * 2 + [response, self.response('')]
                result, _ = self.run_judge(replies, recovery=True, direct_fallback=True)
                self.assertIsNone(result['judge_result'])
                self.assertTrue(result['judge_error'])
                self.assertEqual(result['closed'], [1])


    def test_ordered_mixed_chain_and_history(self):
        chain = [
            {"name": "first", "transport": "direct", "base_url": "http://first/v1", "model": "first"},
            {"name": "second", "transport": "sandbox", "base_url": "http://second/v1", "model": "second",
             "sandbox_replacements": 0, "retry_delays": []},
            {"name": "third", "transport": "direct", "base_url": "http://third/v1", "model": "third",
             "thinking": "disabled"},
        ]
        result, calls = self.run_judge([self.response('', status=503), self.response('', status=502),
                                      self.response('{"correct": true}')], chain=chain)
        self.assertEqual([c["model"] for c in calls], ["first", "second", "third"])
        self.assertEqual(result["sandbox_calls"], [0])
        self.assertEqual(result["judge_index"], 2)
        self.assertEqual([h["valid"] for h in result["history"]], [False, False, True])
        self.assertEqual(result["judge_result"]["score"], 1)
        self.assertEqual([c["messages"] for c in calls], [calls[0]["messages"]] * 3)

    def test_direct_only_does_not_use_sandbox(self):
        chain = [{"name": "local", "transport": "direct", "base_url": "http://local/v1", "model": "local",
                  "reasoning_effort": 35, "max_tokens": 300, "timeout": 9}]
        result, calls = self.run_judge([self.response('{"correct": false}')], chain=chain)
        self.assertEqual(result["sandbox_calls"], [])
        self.assertEqual(result["closed"], [])
        self.assertEqual(result["judge_backend"], "direct")
        self.assertEqual(calls[0]["chat_template_kwargs"]["reasoning_effort"], 35)
        self.assertEqual(calls[0]["max_tokens"], 300)
        self.assertEqual(result["direct_options"][0]["timeout"], 9)

    def test_direct_retry_before_next_service(self):
        chain = [{"name": "retry", "transport": "direct", "base_url": "http://local/v1", "model": "retry",
                  "retry_delays": [0]}]
        result, calls = self.run_judge([self.response('', status=503), self.response('{"correct": true}')], chain=chain)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["judge_result"]["score"], 1)

    def test_transport_and_thinking_format_are_independent(self):
        chain = [{"name": "plain", "transport": "direct", "base_url": "http://local/v1", "model": "plain",
                  "thinking_format": "none"}]
        result, calls = self.run_judge([self.response('{"correct": false}')], chain=chain)
        self.assertNotIn("thinking", calls[0])
        self.assertNotIn("chat_template_kwargs", calls[0])
        self.assertEqual(result["judge_attempts"], 1)

    def test_malformed_config_rejected_before_requests(self):
        valid = {"transport": "direct", "base_url": "http://local/v1", "model": "local"}
        cases = [[], {}, [dict(valid, transport="automatic")], [dict(valid, reasoning_effort=101)],
                 [dict(valid, reasoning_effort=True)], [dict(valid, retry_delays=[181])],
                 [dict(valid, sandbox_replacements=1)], [dict(valid, api_key_env="MISSING_JUDGE_TEST_KEY")],
                 [dict(valid, max_tokens=0)], [dict(valid, timeout=float('nan'))],
                 [dict(valid, typo="x")], [dict(valid, name="same"), dict(valid, name="same")]]
        for chain in cases:
            with self.subTest(chain=chain), self.assertRaises(ValueError):
                self.run_judge([], chain=chain)



    def test_later_sandbox_service_uses_and_closes_fresh_run(self):
        chain = [{"name": name, "transport": "sandbox", "base_url": "http://judge/v1", "model": name,
                  "sandbox_replacements": 0, "retry_delays": []} for name in ("first", "second")]
        result, calls = self.run_judge([self.response('', status=503), self.response('{"correct": true}')], chain=chain)
        self.assertEqual(result["sandbox_calls"], [0, 1])
        self.assertEqual(result["closed"], [1])
        self.assertEqual(result["sandbox_restarts"], 1)
        self.assertEqual([c["model"] for c in calls], ["first", "second"])

    def test_per_service_auth_and_vendor_thinking_on_direct_transport(self):
        chain = [{"transport": "direct", "base_url": "http://judge/v1", "model": "vendor",
                  "api_key_env": "API_KEY", "thinking_format": "thinking", "thinking": "enabled"}]
        result, calls = self.run_judge([self.response('{"correct": false}')], chain=chain)
        self.assertEqual(result["direct_options"][0]["headers"], {"Authorization": "Bearer test"})
        self.assertEqual(calls[0]["thinking"], {"type": "enabled"})
        self.assertNotIn("chat_template_kwargs", calls[0])

    def test_legacy_environment_defaults(self):
        source = Path(__file__).resolve().parents[1] / "inference/judge_client.py"
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "load_judge_chain")
        namespace = dict(json=json, os=os, math=math, Path=Path)
        exec(compile(ast.Module([node], type_ignores=[]), str(source), "exec"), namespace)
        env = {"SCIENCE_AGENT_JUDGE_BASE_URL": "http://primary/v1", "SCIENCE_AGENT_JUDGE_MODEL": "primary",
               "API_KEY": "legacy", "SCIENCE_AGENT_JUDGE_THINKING": "enabled"}
        with patch.dict(os.environ, env, clear=True):
            chain = namespace["load_judge_chain"](default_max_tokens=6000)
            self.assertEqual([e["transport"] for e in chain], ["sandbox", "direct"])
            self.assertEqual(chain[0]["api_key"], "legacy")
            self.assertEqual(chain[0]["sandbox_replacements"], 1)
            self.assertEqual(chain[0]["retry_delays"], [1, 5, 10, 30, 60, 180])
            self.assertEqual(chain[1]["reasoning_effort"], 20)
            self.assertEqual(chain[1]["max_tokens"], 6000)
            with patch.dict(os.environ, SCIENCE_AGENT_JUDGE_FALLBACK_BASE_URL=""):
                self.assertEqual(len(namespace["load_judge_chain"]()), 1)


if __name__ == "__main__":
    unittest.main()
