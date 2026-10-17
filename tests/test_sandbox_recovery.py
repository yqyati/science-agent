"""Recovery semantics with real Python execution and a fake management plane."""
import io
import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import sandbox_gateway as gateway
import sandbox_python_worker as worker
import tool_http_client as transport
from sandbox_recovery import RecoveryJournal


class Client:
    _shell_quote = staticmethod(gateway.SandboxGatewayClient._shell_quote)

    def __init__(self):
        self.alive = True
        self.calls = []
        self.files = {}
        self.namespace = {}
        self.results = {}
        self.lose_response = False

    def proxy_json(self, sid, port, path, payload, timeout=180):
        if not self.alive:
            raise gateway.SandboxInactiveError("inactive")
        if path == "/request":
            return {"status_code": 200, "page": {"path": "pages/test/content.md"},
                    "recovery_files": {"pages/test/content.md": "original evidence", "pages/test/source.json": "{}"}}
        self.calls.append(dict(payload))
        output = io.StringIO()
        from contextlib import redirect_stdout
        with redirect_stdout(output):
            try:
                exec(payload["code"], self.namespace)
                error = ""
            except Exception as exc:
                error = type(exc).__name__
        result = {"stdout": output.getvalue(), "stderr": "", "error": error, "images": []}
        response = {"state": "completed", "result": result}
        self.results[payload["operation_id"]] = response
        if self.lose_response:
            self.lose_response = False
            raise TimeoutError("response lost")
        return response

    def proxy_http(self, sid, port, path, timeout=10):
        if not self.alive:
            raise gateway.SandboxInactiveError("inactive")
        return 200, json.dumps(self.results.get(path.split("/")[-1], {"state": "unknown"}))

    def command(self, *args, **kwargs):
        pass

    def upload_file_bytes(self, sid, path, content):
        self.files[path] = content

    def delete(self, sid):
        pass


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {"SCIENCE_AGENT_SANDBOX_RECOVERY": "1",
                                          "SCIENCE_AGENT_SANDBOX_RECOVERY_DIR": self.temp.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.run = gateway.AgentSandboxRun()
        self.client = Client()
        self.run.client = self.client
        self.run.sandbox_id = "old"
        def ensure():
            if not self.run.sandbox_id:
                self.client.alive = True
                self.client.namespace = {}
                self.client.files = {}
                self.run.sandbox_id = "new"
            return self.run.sandbox_id
        self.run.ensure = ensure
        self.run._start_worker = lambda *args: self.run.ensure()

    def test_rebuild_variables_files_and_partial_error_state(self):
        self.run.run_http({"save_page": True})
        self.run.run_python("x = 40")
        self.run.run_python("y = 2; raise ValueError()")
        self.client.alive = False
        with self.assertRaisesRegex(gateway.SandboxGatewayError, "replayed 2"):
            self.run.run_python("x += 100")
        with self.assertRaisesRegex(gateway.SandboxGatewayError, "pending recovery notice"):
            self.run.run_python("x += 100")
        self.assertIn("Environment replaced", self.run.take_recovery_notice())
        self.assertEqual(self.run.run_python("print(x+y)"), "42\n")
        self.assertEqual(self.client.files["/workspace/pages/test/content.md"], b"original evidence")
        self.assertEqual(self.run.take_recovery_notice(), "")
        self.assertEqual([e["kind"] for e in self.run._journal.events[:3]], ["file", "file", "python_start"])

    def test_lost_response_is_queried_not_reexecuted(self):
        self.client.lose_response = True
        self.assertEqual(self.run.run_python("x = 1; print(x)"), "1\n")
        self.assertEqual(len(self.client.calls), 1)
        self.assertIsNone(self.run._pending_python)
        self.assertEqual(self.run.run_python("x += 1; print(x)"), "2\n")

    def test_http_only_request_does_not_wait_for_python(self):
        self.run._pending_python = "previous"
        for state in ("running", "unknown", "completed"):
            with self.subTest(state=state):
                self.client.results["previous"] = {"state": state}
                with patch.object(self.client, "proxy_http", wraps=self.client.proxy_http) as poll:
                    result = self.run.run_http({"method": "POST", "url": "https://search.example", "json": {"q": "test"}})
                self.assertEqual(result["status_code"], 200)
                poll.assert_not_called()
                self.assertEqual(self.run._pending_python, "previous")
                self.assertEqual(self.client.calls, [])

    def test_pending_python_still_blocks_file_writes_and_python(self):
        self.run._pending_python = "previous"
        for call in (lambda: self.run.run_http({"save_page": True}), lambda: self.run.run_python("x = 1")):
            with self.assertRaisesRegex(gateway.SandboxStateError, "outcome is unknown"):
                call()
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.client.files, {})

    def test_recovery_notice_blocks_http_until_consumed(self):
        self.run._recovery_notice.append("Environment replaced.")
        with self.assertRaisesRegex(gateway.SandboxStateError, "pending recovery notice"):
            self.run.run_http({})
        self.assertEqual(self.run.take_recovery_notice(), "Environment replaced.")
        self.assertEqual(self.run.run_http({})["status_code"], 200)

    def test_state_error_never_falls_back_to_host_http(self):
        with patch.dict(os.environ, {"SCIENCE_AGENT_TOOL_HTTP_BACKEND": "gateway_sandbox", "SCIENCE_AGENT_SANDBOX_HTTP_STRICT": "false"}), \
                patch.object(transport, "_sandbox_http_request", side_effect=gateway.SandboxStateError("pending recovery notice")), \
                patch.object(transport, "tool_http_config") as fallback:
            with self.assertRaises(gateway.SandboxStateError):
                transport.tool_http_request(prefix="SERPER", method="POST", url="https://search.example", trace_layer="test", trace_event="test")
        fallback.assert_not_called()

    def test_search_wrappers_return_state_without_retry_or_instructions(self):
        for filename, method_name in (("tool_search_mcp_r1.py", "google_search_with_serp"), ("tool_scholar.py", "google_scholar_with_serp")):
            with self.subTest(tool=filename):
                source = Path(gateway.__file__).with_name(filename)
                method = next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.FunctionDef) and n.name == method_name)
                request = Mock(side_effect=gateway.SandboxStateError("pending recovery notice"))
                clock = Mock()
                clock.time.return_value = 0
                namespace = dict(provider_strategy=lambda: "original", SERPER_KEY="test", SERPER_API_URL="https://search.example", SERPER_SCHOLAR_API_URL="https://scholar.example", _serper_max_retries=lambda: 5, _serper_retry_sleep_sec=lambda: 1, time=clock, tool_http_config=lambda _: dict(backend="gateway_sandbox", connect_timeout=1, read_timeout=1, max_time=1), emit_trace=Mock(), tool_http_request=request, SandboxStateError=gateway.SandboxStateError, successful_calls_search=0, failed_calls_search=0)
                exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
                result = namespace[method_name](None, "test query")
                self.assertIn("unavailable: pending recovery notice", result)
                self.assertNotIn("Please", result)
                request.assert_called_once()
                clock.sleep.assert_not_called()

    def test_unknown_execution_stops_replay(self):
        self.run._get_journal().record({"kind": "python_start", "operation_id": "unknown", "payload": {"code": "x=1"}, "timeout": 1})
        self.run._pending_python = "unknown"
        self.client.alive = False
        with self.assertRaisesRegex(gateway.SandboxGatewayError, "historical_execution_outcome_unknown"):
            self.run.run_python("print(x)")
        self.assertTrue(self.run._recovery_broken)
        self.assertEqual(len(self.client.calls), 0)

    def test_output_divergence_stops_replay(self):
        self.run.run_python("print(__name__)")
        event = next(e for e in self.run._journal.events if e['kind'] == 'python_result')
        event['sha256'] = 'different'
        self.client.alive = False
        with self.assertRaisesRegex(gateway.SandboxGatewayError, "python_replay_output_mismatch"):
            self.run.run_python("x = 1")
        self.assertNotIn('x', self.client.namespace)

    def test_generic_gateway_error_does_not_reset(self):
        with patch.object(self.client, 'proxy_json', side_effect=gateway.SandboxGatewayError('HTTP 503')):
            with self.assertRaises(gateway.SandboxGatewayError):
                self.run.run_python('x = 1')
        self.assertEqual(self.run.sandbox_id, 'old')
        self.assertEqual(self.run._recovery_count, 0)

    def test_recovery_limit(self):
        self.run._recovery_count = 2
        self.client.alive = False
        with self.assertRaisesRegex(gateway.SandboxGatewayError, 'limit reached'):
            self.run.run_http({})

    def test_restart_clears_replay_code_but_keeps_file_backups(self):
        self.run.run_http({"save_page": True})
        self.run.run_python("old = 1")
        self.run._get_journal().record({"kind": "python_session_reset"})
        self.client.namespace.clear()
        self.run.run_python("new = 2")
        self.client.alive = False
        with self.assertRaisesRegex(gateway.SandboxStateError, "Recovery is partial"):
            self.run.run_python("print(new)")
        self.assertNotIn("old", self.client.namespace)
        self.assertEqual(self.client.namespace["new"], 2)
        self.assertEqual(self.client.files["/workspace/pages/test/content.md"], b"original evidence")
        starts = [e for e in self.run._journal.events if e["kind"] == "python_start"]
        self.assertFalse(any("old =" in e["payload"]["code"] for e in starts))

    def test_interrupted_code_is_excluded_from_replay(self):
        self.run.run_python("x = 1")
        journal = self.run._get_journal()
        journal.record({"kind": "python_start", "operation_id": "loop", "payload": {"code": "while True: pass"}, "timeout": 1})
        journal.record({"kind": "python_interrupted", "operation_id": "loop"})
        self.client.alive = False
        with self.assertRaisesRegex(gateway.SandboxStateError, "Recovery is partial"):
            self.run.run_python("print(x)")
        self.assertEqual(self.client.namespace["x"], 1)

    def test_journal_integrity_and_paths(self):
        journal = RecoveryJournal(self.temp.name)
        with self.assertRaises(ValueError):
            journal.file('../outside', b'x')
        journal.file('pages/a', b'content')
        event = journal.events[-1]
        (journal.directory / event['sha256']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            journal.read(event)

    def test_worker_operation_id_deduplicates_and_detects_conflicts(self):
        worker.OPERATIONS.clear()
        worker.GLOBALS.clear()
        payload = {'operation_id': 'test', 'code': 'x = globals().get("x", 0) + 1'}
        worker.execute_operation(payload)
        worker.execute_operation(payload)
        self.assertEqual(worker.GLOBALS['x'], 1)
        with self.assertRaises(ValueError):
            worker.execute_operation(dict(payload, code='x = 3'))

    def test_management_error_is_typed_only_for_inactive(self):
        client = gateway.SandboxGatewayClient('http://gateway', 'token')
        error = HTTPError('http://gateway', 409, 'inactive', {}, io.BytesIO(
            b'{"detail":{"code":"SANDBOX_NOT_ACTIVE"}}'))
        with patch.object(gateway.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaises(gateway.SandboxInactiveError):
                client.endpoint('sid')


if __name__ == '__main__':
    unittest.main()
