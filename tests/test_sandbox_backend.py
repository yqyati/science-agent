import ast
import asyncio
import io
import json
import os
import sys
import unittest
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))

import sandbox_gateway as gateway
import sandbox_dns
import sandbox_http_worker as http_worker
import sandbox_python_worker as python_worker
import tool_http_client as transport
RealGatewayClient = gateway.SandboxGatewayClient


class SandboxCreateRetryTests(unittest.TestCase):
    def test_transient_http_failure_recovers_without_changing_payload(self):
        client = RealGatewayClient("http://gateway", "secret-token", "resource")
        for status in (403, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                error = HTTPError("http://gateway", status, "temporary", {}, io.BytesIO(b"temporary"))
                with patch.object(gateway.urllib.request, "urlopen", side_effect=[
                    error, io.BytesIO(b'{"id":"created"}'),
                ]) as request, patch.object(gateway.time, "sleep") as sleep:
                    with self.assertLogs(gateway.__name__, level="WARNING") as logs:
                        self.assertEqual(client.create(image="image", env={"API_KEY": "secret-key"}), "created")
                self.assertEqual(request.call_count, 2)
                sleep.assert_called_once_with(1)
                first, second = [call.args[0] for call in request.call_args_list]
                self.assertEqual(first.data, second.data)
                self.assertEqual(json.loads(first.data)["env"], {"API_KEY": "secret-key"})
                self.assertNotIn("secret-key", str(logs.output))
                self.assertNotIn("secret-token", str(logs.output))

    def test_network_failures_recover_with_bounded_backoff(self):
        client = RealGatewayClient("http://gateway", "token")
        with patch.object(gateway.urllib.request, "urlopen", side_effect=[
            URLError(ConnectionRefusedError("refused")), TimeoutError("timed out"),
            URLError("temporary DNS failure"), io.BytesIO(b'{"id":"created"}'),
        ]) as request, patch.object(gateway.time, "sleep") as sleep:
            self.assertEqual(client.create(image="image"), "created")
        self.assertEqual(request.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 5, 10])

    def test_create_stops_after_four_failures(self):
        client = RealGatewayClient("http://gateway", "token")
        with patch.object(gateway.urllib.request, "urlopen", side_effect=TimeoutError("timed out")) as request, \
                patch.object(gateway.time, "sleep") as sleep:
            with self.assertRaisesRegex(gateway.SandboxGatewayError, "create failed after 4 attempts"):
                client.create(image="image")
        self.assertEqual(request.call_count, 4)
        self.assertEqual(sleep.call_count, 3)

    def test_permanent_http_failures_are_not_retried(self):
        client = RealGatewayClient("http://gateway", "token")
        for status in (400, 401, 404, 409, 422):
            with self.subTest(status=status):
                error = HTTPError("http://gateway", status, "permanent", {}, io.BytesIO(b"permanent"))
                with patch.object(gateway.urllib.request, "urlopen", side_effect=error) as request, \
                        patch.object(gateway.time, "sleep") as sleep:
                    with self.assertRaises(gateway.SandboxGatewayError):
                        client.create(image="image")
                self.assertEqual(request.call_count, 1)
                sleep.assert_not_called()

    def test_missing_id_does_not_trigger_another_creation(self):
        client = RealGatewayClient("http://gateway", "token")
        with patch.object(gateway.urllib.request, "urlopen", return_value=io.BytesIO(b'{}')) as request:
            with self.assertRaisesRegex(gateway.SandboxGatewayError, "no sandbox id"):
                client.create(image="image")
        self.assertEqual(request.call_count, 1)

    def test_other_posts_are_not_replayed(self):
        client = RealGatewayClient("http://gateway", "token")
        error = HTTPError("http://gateway", 503, "temporary", {}, io.BytesIO(b"temporary"))
        with patch.object(gateway.urllib.request, "urlopen", side_effect=error) as request, \
                patch.object(gateway.time, "sleep") as sleep:
            with self.assertRaises(gateway.SandboxGatewayError):
                client._request("POST", "/v1/sandboxes/existing/commands", {"command": "work"})
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()


class FakeClient:
    def __init__(self, *_args):
        self.deleted = []
        self.uploads = []
        self.globals = {}

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return "sandbox-1"

    def wait_running(self, *_args):
        pass

    def upload_file_bytes(self, _sid, path, _data):
        self.uploads.append(path)

    def command(self, *_args, **_kwargs):
        return ""

    _shell_quote = staticmethod(gateway.SandboxGatewayClient._shell_quote)

    def proxy_http(self, *_args, **_kwargs):
        return 200, "ok"

    def proxy_json(self, _sid, port, _path, payload, **_kwargs):
        if _path == '/prepare-vision':
            return {'ready': True}
        if port == 18080:
            from contextlib import redirect_stdout
            output = io.StringIO()
            with redirect_stdout(output):
                exec(payload["code"], self.globals)
            return {"stdout": output.getvalue()}
        return {"status_code": 200, "text": "ok"}

    def delete(self, sid):
        self.deleted.append(sid)


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "OPEN_SANDBOX_GATEWAY_TOKEN": "test-token",
            "SCIENCE_AGENT_TOOL_HTTP_BACKEND": "gateway_sandbox",
        })
        self.env.start()
        self.client = patch.object(gateway, "SandboxGatewayClient", FakeClient)
        self.client.start()

    def tearDown(self):
        gateway.close_shared_sandbox_run()
        self.client.stop()
        self.env.stop()

    def test_python_and_http_share_run_and_cleanup_on_error(self):
        captured = []

        @gateway.sandbox_run_scope
        def run_agent():
            run = gateway.shared_sandbox_run()
            captured.append(run)
            self.assertEqual(run.run_python("x=40\nprint(x+2)"), "42\n")
            self.assertEqual(run.run_python("print(x+1)"), "41\n")
            response = transport.tool_http_request(
                prefix="JINA", method="GET", url="https://example.com",
                trace_layer="test", trace_event="test",
            )
            self.assertEqual(response.backend, "gateway_sandbox")
            self.assertEqual(len(run.client.uploads), 3)
            raise ValueError("test error")

        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "test error"):
                run_agent()
        self.assertIsNot(captured[0], captured[1])
        for run in captured:
            self.assertEqual(run.client.deleted, ["sandbox-1"])

    def test_concurrent_agents_have_distinct_runs(self):
        barrier = Barrier(2)

        @gateway.sandbox_run_scope
        def run_agent():
            run = gateway.shared_sandbox_run()
            barrier.wait(timeout=5)
            self.assertIs(gateway.shared_sandbox_run(), run)
            return run

        with ThreadPoolExecutor(max_workers=2) as pool:
            runs = list(pool.map(lambda _: run_agent(), range(2)))
        self.assertIsNot(runs[0], runs[1])

    def test_async_task_reuses_tool_sandbox_for_judge_and_cleans_up(self):
        captured = []

        @gateway.sandbox_run_scope
        def run_tools():
            run = gateway.shared_sandbox_run()
            run.run_python("print(42)")
            return run

        @gateway.sandbox_run_scope
        async def run_task(fail):
            run = await asyncio.to_thread(run_tools)
            captured.append(run)
            self.assertIs(run, gateway.shared_sandbox_run())
            self.assertEqual(run.client.deleted, [])
            await asyncio.to_thread(run.run_http, {"url": "https://judge"})
            self.assertEqual(run.client.deleted, [])
            if fail:
                raise ValueError("judge failure")

        asyncio.run(run_task(False))
        with self.assertRaisesRegex(ValueError, "judge failure"):
            asyncio.run(run_task(True))
        self.assertIsNot(captured[0], captured[1])
        for run in captured:
            self.assertEqual(run.client.deleted, ["sandbox-1"])
        self.assertIsNone(gateway._SHARED_RUN.get())

    def test_concurrent_async_tasks_keep_distinct_sandboxes(self):
        barrier = Barrier(2)

        @gateway.sandbox_run_scope
        def run_tools():
            run = gateway.shared_sandbox_run()
            run.ensure()
            barrier.wait(timeout=5)
            return run

        @gateway.sandbox_run_scope
        async def run_task():
            run = await asyncio.to_thread(run_tools)
            self.assertIs(run, gateway.shared_sandbox_run())
            return run

        async def run_both():
            return await asyncio.gather(run_task(), run_task())

        runs = asyncio.run(run_both())
        self.assertIsNot(runs[0], runs[1])
        for run in runs:
            self.assertEqual(run.client.deleted, ["sandbox-1"])

    def test_molt_adapter_keeps_sandbox_until_judge_finishes(self):
        from types import SimpleNamespace
        source = Path(__file__).resolve().parents[3] / "labs-molt/examples/python/agents/science_legacy_react_sandbox.py"
        if not source.is_file():
            self.skipTest("Molt adapter integration requires the sibling labs-molt checkout")
        tree = ast.parse(source.read_text())
        agent_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ScienceLegacyReactAgent")
        method = agent_class.body[0]
        captured = []

        class ToolAgent:
            def __init__(self, **kwargs):
                pass

            @gateway.sandbox_run_scope
            def _run(self, *args):
                run = gateway.shared_sandbox_run()
                captured.append(run)
                run.run_python("print(42)")
                return {"prediction": "42", "messages": []}

        def judge(run, payload, **kwargs):
            self.assertIs(run, captured[-1])
            self.assertEqual(run.client.deleted, [])
            self.assertEqual(payload["json"]["model"], "test-judge")
            return {"status_code": 200, "text": json.dumps({"choices": [{"message": {
                "content": '{"score": 1, "correct": true}',
            }, "finish_reason": "stop"}]})}

        namespace = {
            "sandbox_run_scope": gateway.sandbox_run_scope,
            "shared_sandbox_run": gateway.shared_sandbox_run,
            "asyncio": asyncio, "os": os, "json": json,
            "ChatContext": object, "Result": lambda **kwargs: SimpleNamespace(**kwargs),
            "MultiTurnReactAgent": ToolAgent, "SCIENCE_AGENT_ROOT": source.parent,
            "_text": lambda value: "" if value is None else str(value),
            "torch": SimpleNamespace(tensor=lambda value, **kwargs: value, float32=None),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        ctx = SimpleNamespace(label={"answer": "42"}, model_name="test-policy", base_url="http://policy",
                              api_key="test-key", prompt="Question", messages=[], sampling_params=None)
        with patch.dict(os.environ, {"SCIENCE_AGENT_JUDGE_BASE_URL": "http://judge/v1",
                                     "SCIENCE_AGENT_JUDGE_MODEL": "test-judge", "API_KEY": "test-key",
                                     "SCIENCE_AGENT_REWARD_MODE": "judge", "SCIENCE_AGENT_NEXT_HLE_EVAL_PROTOCOL": "0"}):
            with patch.object(gateway.AgentSandboxRun, "run_http", judge):
                result = asyncio.run(namespace["run"](None, ctx))
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].client.deleted, ["sandbox-1"])

    def test_delete_failure_keeps_id_for_retry(self):
        run = gateway.shared_sandbox_run()
        run.ensure()
        with patch.object(gateway.time, "sleep"), patch.object(run.client, "delete", side_effect=RuntimeError("delete failed")):
            with self.assertRaises(RuntimeError):
                run.close()
        self.assertEqual(run.sandbox_id, "sandbox-1")
        run.close()
        self.assertEqual(run.sandbox_id, "")

    def test_delete_missing_sandbox_completes_cleanup_without_retry(self):
        run = gateway.AgentSandboxRun()
        run.client = RealGatewayClient("http://gateway", "test-token")
        run.sandbox_id = "missing-sandbox"
        run._python_started = run._http_started = True
        body = b'{"detail":{"code":"NOT_FOUND","message":"sandbox not found"}}'
        error = HTTPError("http://gateway", 404, "Not Found", {}, io.BytesIO(body))
        with patch.object(gateway.urllib.request, "urlopen", side_effect=error) as request:
            run.close()
        self.assertEqual(request.call_count, 1)
        self.assertEqual(run.sandbox_id, "")
        self.assertFalse(run._python_started or run._http_started)

    def test_delete_does_not_ignore_other_http_errors(self):
        client = RealGatewayClient("http://gateway", "test-token")
        for status, body in [(404, b"route not found"), (403, b"sandbox not found"), (500, b"server error")]:
            with self.subTest(status=status):
                error = HTTPError("http://gateway", status, "error", {}, io.BytesIO(body))
                with patch.object(gateway.urllib.request, "urlopen", side_effect=error):
                    with self.assertRaises(gateway.SandboxGatewayError):
                        client.delete("sid")

    def test_cleanup_failure_preserves_result_or_original_error(self):
        result = {"prediction": "5003", "reward": 1.0, "messages": ["trajectory"]}
        original_error = ValueError("generation failed")
        for asynchronous in (False, True):
            for fail in (False, True):
                with self.subTest(asynchronous=asynchronous, fail=fail):
                    captured = []

                    def body():
                        run = gateway.shared_sandbox_run()
                        run.ensure()
                        run.client.delete = unittest.mock.Mock(side_effect=RuntimeError("delete failed"))
                        captured.append(run)
                        if fail:
                            raise original_error
                        return result

                    async def async_body():
                        return body()

                    wrapped = gateway.sandbox_run_scope(async_body if asynchronous else body)
                    with patch.object(gateway.time, "sleep"), self.assertLogs(gateway.__name__, level="ERROR") as logs:
                        if fail:
                            with self.assertRaises(ValueError) as raised:
                                asyncio.run(wrapped()) if asynchronous else wrapped()
                            self.assertIs(raised.exception, original_error)
                        else:
                            actual = asyncio.run(wrapped()) if asynchronous else wrapped()
                            self.assertIs(actual, result)
                    self.assertEqual(captured[0].client.delete.call_count, 3)
                    self.assertTrue(any("Sandbox cleanup failed id=sandbox-1" in log for log in logs.output))
                    self.assertIsNone(gateway._SHARED_RUN.get())

    def test_agent_cleanup_retries_transient_delete_failure(self):
        captured = []

        @gateway.sandbox_run_scope
        def run_agent():
            run = gateway.shared_sandbox_run()
            run.ensure()
            captured.append(run)
            run.client.delete = unittest.mock.Mock(side_effect=[RuntimeError("temporary"), None])

        with patch.dict(os.environ, {"PYTHON_SESSION_BACKEND": "gateway_sandbox"}), patch.object(gateway.time, "sleep"):
            run_agent()
        self.assertEqual(captured[0].client.delete.call_count, 2)
        self.assertEqual(captured[0].sandbox_id, "")

    def test_gateway_lookup_retries_403_and_returns_recovered_response(self):
        client = RealGatewayClient("http://gateway", "test-token")
        error = HTTPError("http://gateway", 403, "Forbidden", {}, io.BytesIO(b"awaiting authorization"))
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"endpoint":"http://worker"}'
        with patch.object(gateway.urllib.request, "urlopen", side_effect=[error, response]) as request, patch.object(gateway.time, "sleep"):
            self.assertEqual(client.endpoint("sid")[0], "http://worker")
            self.assertEqual(request.call_count, 2)

    def test_gateway_lookup_reports_error_only_after_three_failures(self):
        client = RealGatewayClient("http://gateway", "test-token")
        errors = [HTTPError("http://gateway", 403, "Forbidden", {}, io.BytesIO(b"awaiting authorization")) for _ in range(3)]
        with patch.object(gateway.urllib.request, "urlopen", side_effect=errors) as request, patch.object(gateway.time, "sleep"):
            with self.assertRaisesRegex(gateway.SandboxGatewayError, "attempt 3/3"):
                client.endpoint("sid")
            self.assertEqual(request.call_count, 3)

    def test_python_proxy_retries_connection_failure_but_not_execution_timeout(self):
        client = RealGatewayClient("http://gateway", "test-token")
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"stdout":"executed once"}'
        errors = [HTTPError("http://worker", 502, "Bad Gateway", {}, io.BytesIO(b"Sandbox upstream connection failed")) for _ in range(2)]
        with patch.object(client, "endpoint", return_value=("http://worker", {})), patch.object(gateway.time, "sleep"):
            with patch.object(gateway.urllib.request, "urlopen", side_effect=errors + [response]) as request:
                self.assertEqual(client.proxy_json("sid", 18080, "/execute", {"code":"print(1)"})["stdout"], "executed once")
                self.assertEqual(request.call_count, 3)
            with patch.object(gateway.urllib.request, "urlopen", side_effect=TimeoutError("response timed out")) as request:
                with self.assertRaises(TimeoutError):
                    client.proxy_json("sid", 18080, "/execute", {"code":"print(1)"})
                self.assertEqual(request.call_count, 1)
            error = HTTPError("http://worker", 502, "Bad Gateway", {}, io.BytesIO(b"worker response interrupted"))
            with patch.object(gateway.urllib.request, "urlopen", side_effect=error) as request:
                with self.assertRaises(gateway.SandboxGatewayError):
                    client.proxy_json("sid", 18080, "/execute", {"code":"print(1)"})
                self.assertEqual(request.call_count, 1)

    def test_agent_finish_marks_budget_and_rejects_placeholder(self):
        from types import SimpleNamespace
        source = Path(__file__).resolve().parents[1] / "inference/react_agent_qwen_api.py"
        tree = ast.parse(source.read_text())
        finish = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "finish")
        namespace = {
            "response_was_truncated": False, "trajectory_budget_exhausted": True,
            "usage_totals": {}, "has_usage": False,
            "self": SimpleNamespace(_attach_usage_fields=lambda result, *args: result),
        }
        exec(compile(ast.Module(body=[finish], type_ignores=[]), str(source), "exec"), namespace)
        result = namespace["finish"]({"prediction": "Your Answer", "termination": "old"})
        self.assertEqual(result["prediction"], "")
        self.assertTrue(result["truncated"])
        self.assertFalse(result["generation_truncated"])
        self.assertFalse(result["final_answer_valid"])
        result = namespace["finish"]({"prediction": "32", "termination": "finalized"})
        self.assertEqual(result["prediction"], "32")
        self.assertEqual(result["termination"], "finalized")
        self.assertTrue(result["final_answer_valid"])
        namespace["trajectory_budget_exhausted"] = False
        result = namespace["finish"]({"prediction": "32"})
        self.assertFalse(result["truncated"])

    def test_disabled_scope_preserves_existing_context(self):
        existing = gateway.shared_sandbox_run()

        @gateway.sandbox_run_scope
        def run_agent():
            return gateway.shared_sandbox_run()

        with patch.dict(os.environ, {"PYTHON_SESSION_BACKEND": "fusion", "SCIENCE_AGENT_TOOL_HTTP_BACKEND": "requests"}):
            self.assertIs(run_agent(), existing)
        self.assertEqual(existing.sandbox_id, "")

    def test_startup_failure_deletes_created_instance(self):
        run = gateway.shared_sandbox_run()
        with patch.object(FakeClient, "wait_running", side_effect=gateway.SandboxGatewayError("failed startup")):
            with self.assertRaisesRegex(gateway.SandboxGatewayError, "failed startup"):
                run.ensure()
        self.assertEqual(run.client.deleted, ["sandbox-1"])
        self.assertEqual(run.sandbox_id, "")

    def test_explicit_network_policy_is_sent_on_creation(self):
        run = gateway.shared_sandbox_run()
        with patch.dict(os.environ, {"SCIENCE_AGENT_SANDBOX_NETWORK_DEFAULT_ACTION": "allow"}):
            run.ensure()
        self.assertEqual(run.client.create_kwargs["network_policy"], {
            "defaultAction": "deny", "egress": [
                {"action": "allow", "target": "0.0.0.0/0"},
                {"action": "allow", "target": "::/0"},
            ],
        })

    def test_dns_fallback_verifies_before_writing_and_restores_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            resolver = Path(directory) / 'resolv.conf'
            original = 'search svc.cluster.local\nnameserver 169.254.25.10\noptions ndots:5\n'
            resolver.write_text(original)
            with patch.object(sandbox_dns, 'Path', return_value=resolver):
                with patch.object(sandbox_dns, 'resolver_works', return_value=False):
                    with self.assertRaisesRegex(RuntimeError, 'No fallback DNS'):
                        sandbox_dns.configure(['1.1.1.1'])
                self.assertEqual(resolver.read_text(), original)
                with patch.object(sandbox_dns, 'resolver_works', return_value=True):
                    with patch.object(sandbox_dns.socket, 'getaddrinfo', side_effect=OSError('resolution failed')):
                        with self.assertRaises(OSError):
                            sandbox_dns.configure(['1.1.1.1'])
                    self.assertEqual(resolver.read_text(), original)
                    with patch.object(sandbox_dns.socket, 'getaddrinfo', return_value=[(0, 0, 0, '', ('93.184.215.14', 443))]):
                        result = sandbox_dns.configure(['1.1.1.1'])
                self.assertEqual(result['dns_servers'], ['1.1.1.1'])
                self.assertIn('nameserver 1.1.1.1', resolver.read_text())
                self.assertIn('search svc.cluster.local', resolver.read_text())

    def test_ca_bundle_is_valid_and_injected_into_sandbox(self):
        with patch.dict(os.environ, {'SCIENCE_AGENT_SANDBOX_CA_BUNDLE': 'auto'}):
            bundle = gateway.AgentSandboxRun._ca_bundle()
            self.assertIn(b'-----BEGIN CERTIFICATE-----', bundle)
            gateway.ssl.create_default_context(cadata=bundle.decode('ascii'))
            run = gateway.shared_sandbox_run()
            run.ensure()
        self.assertEqual(run.client.create_kwargs['env']['SSL_CERT_FILE'], '/tmp/science-agent-ca.pem')
        self.assertIn('/tmp/science-agent-ca.pem', run.client.uploads)

    def test_large_upload_is_split_into_bounded_commands(self):
        client = object.__new__(RealGatewayClient)
        with patch.object(client, 'command') as command:
            client.upload_file_bytes('test', '/tmp/ca.pem', b'x' * (200 * 1024))
        self.assertEqual(command.call_count, 5)
        calls = [call.args[1] for call in command.call_args_list]
        self.assertTrue(all(len(text) < 70000 for text in calls))
        self.assertIn(' > ', calls[0])
        self.assertTrue(all(' >> ' in text for text in calls[1:]))

    def test_visual_worker_returns_crop_plot_and_persistent_state(self):
        import base64
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.png'
            Image.new('RGB', (20, 16), 'red').save(path)
            data = python_worker.execute_payload({'visual': True, 'code':
                'from PIL import Image\n'
                f'img = Image.open({str(path)!r})\n'
                'display(img.crop((0,0,5,6)))\n'
                'import matplotlib.pyplot as plt\nplt.plot([1,2],[3,4])\nplt.show()\n'
                'print("visual_measurement_summary: crop 5x6")'})
            self.assertEqual(data['error'], '')
            self.assertEqual(len(data['images']), 2)
            crop = Image.open(io.BytesIO(base64.b64decode(data['images'][0]['data'])))
            self.assertEqual(crop.size, (5,6))
            self.assertIn('crop 5x6', data['stdout'])
            subsequent = python_worker.execute_payload({'code': 'print(img.size)'})
            self.assertEqual(subsequent['stdout'], '(20, 16)\n')
            tail = python_worker.execute_payload({'visual': True, 'code': 'img.crop((0,0,2,3))'})
            self.assertEqual(len(tail['images']), 1)

    def test_visual_upload_deduplicates_within_run(self):
        import base64
        files = {'input_images/test.png': base64.b64encode(b'image').decode()}
        run = gateway.shared_sandbox_run()
        run.run_visual('print(1)', files)
        run.run_visual('print(2)', files)
        self.assertEqual(run.client.uploads.count('/workspace/input_images/test.png'), 1)
        with self.assertRaises(gateway.SandboxGatewayError):
            run.run_visual('print(3)', {'../outside.png': files['input_images/test.png']})

    def test_vision_installs_only_missing_packages_and_checks_errors(self):
        from types import SimpleNamespace
        with patch.object(python_worker.importlib.util, 'find_spec', side_effect=lambda name: None if name == 'matplotlib' else object()):
            with patch.object(python_worker.importlib, 'import_module'):
                with patch.object(python_worker.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as install:
                    result = python_worker.prepare_vision({'install': True})
                    self.assertTrue(result['ready'])
                    self.assertEqual(result['installed'], ['matplotlib'])
                    self.assertEqual(install.call_args.args[0][-1], 'matplotlib')
                with patch.object(python_worker.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stderr='download failed', stdout='')):
                    self.assertIn('download failed', python_worker.prepare_vision({'install': True})['error'])
            with patch.object(python_worker.subprocess, 'run') as install:
                self.assertIn('Missing Vision dependencies', python_worker.prepare_vision({'install': False})['error'])
                install.assert_not_called()

    def test_http_worker_preserves_upstream_status_and_body(self):
        handler = object.__new__(http_worker.Handler)
        handler.path = "/request"
        payload = json.dumps({"url": "https://example.com"}).encode()
        handler.headers = {"Content-Length": str(len(payload))}
        handler.rfile, handler.wfile = io.BytesIO(payload), io.BytesIO()
        handler.send_response = lambda *_: None
        handler.send_header = lambda *_: None
        handler.end_headers = lambda: None
        handler.send_error = lambda *args: self.fail(str(args))
        error = HTTPError("https://example.com", 429, "rate limited", {}, io.BytesIO(b'{"error":"rate limited"}'))
        with patch.object(http_worker, "urlopen", side_effect=error):
            handler.do_POST()
        result = json.loads(handler.wfile.getvalue())
        self.assertEqual(result["status_code"], 429)
        self.assertEqual(result["text"], '{"error":"rate limited"}')

    def test_readpage_summary_uses_sandbox_transport(self):
        # Extract the actual method to avoid optional qwen_agent dependencies.
        source = Path(__file__).resolve().parents[1] / "inference/tool_visit_mcp_r1.py"
        tree = ast.parse(source.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "call_server")
        namespace = {
            "os": os, "json": json, "time": __import__("time"),
            "SUMMARY_REQUEST_TIMEOUT_SEC": 30, "SUMMARY_RETRY_SLEEP_SEC": 0,
            "SUMMARY_CACHE_FILE": "unused", "successful_calls_summarize": 0,
            "failed_calls_summarize": 0, "emit_trace": lambda *a, **k: None,
            "is_text_blocked": lambda *a, **k: False,
            "write_cache_log": lambda *a: None,
            "OpenAI": lambda **k: self.fail("host LLM used"),
        }
        calls = []

        def request(**kwargs):
            calls.append(kwargs)
            return transport.ToolHttpResponse(200, json.dumps({"choices": [{"message": {"content": '{"summary":"ok"}'}}]}), "gateway_sandbox", 0)

        namespace["tool_http_request"] = request
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        with patch.dict(os.environ, {"API_BASE": "http://llm/v1", "API_KEY": "test", "SUMMARY_MODEL_NAME": "test-model"}):
            result = namespace["call_server"](None, [{"role": "user", "content": "Summarize"}])
        self.assertEqual(json.loads(result)["summary"], "ok")
        self.assertEqual(calls[0]["url"], "http://llm/v1/chat/completions")
        self.assertEqual(calls[0]["json_payload"]["model"], "test-model")

        # The existing host SDK route still handles non-gateway mode.
        from types import SimpleNamespace
        sdk_calls = []

        def sdk_create(**kwargs):
            sdk_calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary":"host"}'))])

        namespace["OpenAI"] = lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=sdk_create)))
        namespace["httpx"] = SimpleNamespace(Client=lambda **kwargs: None)
        with patch.dict(os.environ, {"SCIENCE_AGENT_TOOL_HTTP_BACKEND": "requests", "SUMMARY_MODEL_NAME": "host-model"}):
            result = namespace["call_server"](None, [{"role": "user", "content": "Summarize"}])
        self.assertEqual(json.loads(result)["summary"], "host")
        self.assertEqual(sdk_calls[0]["model"], "host-model")
        self.assertEqual(len(calls), 1)

    def test_extractive_visit_skips_tokenizer_and_summary(self):
        from types import SimpleNamespace
        source = Path(__file__).resolve().parents[1] / 'inference/tool_visit_mcp_r1.py'
        tree = ast.parse(source.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'readpage_jina')
        namespace = {
            'os': os, 'time': __import__('time'), 'VISIT_SERVER_MAX_RETRIES': 1,
            'emit_trace': lambda *a, **k: None,
            '_is_blocked_visit_response': lambda *a: False,
            'is_text_blocked': lambda *a, **k: False,
            '_extractive_visit_return': lambda url, goal, content: 'evidence: ' + content,
            'truncate_to_tokens': lambda *a, **k: self.fail('tokenizer called'),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        tool = SimpleNamespace(html_readpage_jina=lambda url: 'page text', call_server=lambda *a, **k: self.fail('summary called'))
        with patch.dict(os.environ, {'SCIENCE_AGENT_READ_PAGE_MODE': 'extractive'}):
            self.assertEqual(namespace['readpage_jina'](tool, 'https://example.com', 'goal'), 'evidence: page text')


if __name__ == "__main__":
    unittest.main()
