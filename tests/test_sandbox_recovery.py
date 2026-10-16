"""Recovery semantics with real Python execution and a fake management plane."""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import sandbox_gateway as gateway
import sandbox_python_worker as worker
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
        with self.assertRaisesRegex(gateway.SandboxGatewayError, "pending"):
            self.run.run_python("x = 1")
        with self.assertRaisesRegex(gateway.SandboxGatewayError, "current request was not executed"):
            self.run.run_python("x += 1")
        self.assertEqual(len(self.client.calls), 1)
        self.run.take_recovery_notice()
        self.assertEqual(self.run.run_python("print(x)"), "1\n")

    def test_unknown_execution_stops_replay(self):
        self.client.lose_response = True
        with self.assertRaises(gateway.SandboxGatewayError):
            self.run.run_python("x = 1")
        self.client.alive = False
        with self.assertRaisesRegex(gateway.SandboxGatewayError, "historical_execution_outcome_unknown"):
            self.run.run_python("print(x)")
        self.assertTrue(self.run._recovery_broken)
        self.assertEqual(len(self.client.calls), 1)

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
