import importlib.util
import json
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
from sandbox_gateway import SandboxGatewayClient

spec = importlib.util.spec_from_file_location("cleanup_sandboxes", ROOT / "cleanup_sandboxes.py")
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class SandboxLedgerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = patch.dict(os.environ, {
            "SCIENCE_AGENT_SANDBOX_LEDGER_DIR": self.directory.name,
            "RUN_NAME": "test-run", "OPEN_SANDBOX_GATEWAY_TOKEN": "secret-test-token",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_parallel_creates_and_failed_delete_keep_pending_records(self):
        def create(i):
            client = SandboxGatewayClient("http://gateway", "secret-test-token", "cluster/pool")
            with patch.object(client, "_request", return_value={"id": f"sid-{i}"}):
                client.create(image="test")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(create, range(12)))
        paths = list(Path(self.directory.name).glob("*.json"))
        self.assertEqual(len(paths), 12)
        self.assertTrue(all(json.loads(p.read_text())["status"] == "created" for p in paths))
        self.assertTrue(all("secret-test-token" not in p.read_text() for p in paths))
        client = SandboxGatewayClient("http://gateway", "token", "cluster/pool")
        with patch.object(client, "_request", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                client.delete("sid-0")
        self.assertTrue(all(json.loads(p.read_text())["status"] == "created" for p in paths))

    def test_cleanup_retries_then_is_idempotent(self):
        client = SandboxGatewayClient("http://gateway", "token", "cluster/pool")
        client._record_lifecycle("sid", "created")
        argv = ["cleanup", "--ledger-dir", self.directory.name, "--run-name", "test-run", "--delete"]
        with patch.object(sys, "argv", argv), patch.object(SandboxGatewayClient, "_request", side_effect=[RuntimeError(), None]) as request, patch.object(cleanup.time, "sleep"):
            self.assertEqual(cleanup.main(), 0)
            self.assertEqual(request.call_count, 2)
            self.assertEqual(cleanup.main(), 0)
            self.assertEqual(request.call_count, 2)

    def test_mismatched_run_prevents_all_deletions(self):
        client = SandboxGatewayClient("http://gateway", "token", "cluster/pool")
        client._record_lifecycle("sid", "created")
        argv = ["cleanup", "--ledger-dir", self.directory.name, "--run-name", "other-run", "--delete"]
        with patch.object(sys, "argv", argv), patch.object(SandboxGatewayClient, "_request") as request:
            with self.assertRaises(SystemExit):
                cleanup.main()
            request.assert_not_called()

    def test_trace_directory_default_and_create_record_failure_cleanup(self):
        client = SandboxGatewayClient("http://gateway", "token", "cluster/pool")
        with patch.dict(os.environ, {"SCIENCE_AGENT_SANDBOX_LEDGER_DIR": "", "SCIENCE_AGENT_TRACE_FILE": str(Path(self.directory.name) / "trace.jsonl")}):
            client._record_lifecycle("sid", "created")
        self.assertEqual(len(list((Path(self.directory.name) / "sandbox_ledger").glob("*.json"))), 1)
        with patch.object(client, "_record_lifecycle", side_effect=OSError("disk full")), patch.object(client, "_request", side_effect=[{"id": "sid-2"}, None]) as request, self.assertLogs("sandbox_gateway", level="ERROR"):
            with self.assertRaises(OSError):
                client.create(image="test")
        self.assertEqual(request.call_args_list[-1].args, ("DELETE", "/v1/sandboxes/sid-2"))


if __name__ == "__main__":
    unittest.main()
