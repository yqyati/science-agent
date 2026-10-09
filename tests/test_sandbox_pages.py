import ast
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
import sandbox_http_worker as worker
import sandbox_gateway as gateway
from tool_url_guard import capture_guard_context, guard_fingerprint


class PageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        previous = os.getcwd()
        os.chdir(self.directory.name)
        self.addCleanup(os.chdir, previous)
        context = capture_guard_context({})
        self.payload = {
            "url": "https://reader/test", "source_url": "https://source/test", "save_page": True,
            "guard_snapshot": context.snapshot.to_dict(), "guard_fingerprint": guard_fingerprint(context),
        }

    def request(self, payload=None):
        handler = object.__new__(worker.Handler)
        handler.path = "/request"
        raw = json.dumps(payload or self.payload).encode()
        handler.headers = {"Content-Length": str(len(raw))}
        handler.rfile, handler.wfile = io.BytesIO(raw), io.BytesIO()
        handler.send_response = lambda *args: None
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        handler.send_error = lambda *args: self.fail(str(args))
        handler.do_POST()
        return json.loads(handler.wfile.getvalue())

    def test_full_page_saved_but_only_metadata_returns_and_cache_reused(self):
        text = "Title: Paper\n\n# Results\n" + "Evidence 42\n" * 3000
        reply = SimpleNamespace(status=200, read=lambda: text.encode())
        reply.__enter__ = lambda self: self
        with patch.object(worker, "urlopen") as fetch:
            fetch.return_value.__enter__.return_value = reply
            result = self.request()
            page = result["page"]
            self.assertEqual(Path(page["path"]).read_text(), text)
            self.assertNotIn("absolute_path", page)
            self.assertFalse(Path(page["path"]).is_absolute())
            self.assertEqual(page["lines"], len(text.splitlines()))
            self.assertLess(len(json.dumps(result)), 2500)
            self.assertNotIn("text", result)
            source = Path(page["path"]).with_name("source.json")
            source.write_text(json.dumps(dict(page, absolute_path=str(Path(page["path"]).resolve()))))
            cached = self.request()
            self.assertNotIn("absolute_path", cached["page"])
            self.assertTrue(cached["page"]["cached"])
            fetch.assert_called_once()
        other = dict(self.payload, source_url="https://source/other")
        second = worker.save_page(other, "Other page")
        self.assertNotEqual(page["path"], second["path"])
        self.assertEqual(Path(page["path"]).read_text(), text)
        self.assertFalse(list(Path("pages").rglob("*.tmp")))

    def test_errors_and_blocked_pages_never_saved(self):
        error = HTTPError(self.payload["url"], 403, "Denied", {}, io.BytesIO(b"Denied"))
        with patch.object(worker, "urlopen", side_effect=error):
            self.assertEqual(self.request()["status_code"], 403)
        self.assertFalse(Path("pages").exists())
        with self.assertRaises(ValueError):
            worker.save_page(self.payload, " ")
        with patch("tool_url_guard.evaluate_guard_value", return_value=SimpleNamespace(blocked=True)):
            self.assertEqual(worker.save_page(self.payload, "blocked text"), {"blocked": True})
        self.assertFalse(Path("pages").exists())

    def test_read_page_file_mode_skips_host_fetch_tokenizer_and_summary(self):
        source = ROOT / "inference/tool_visit_mcp_r1.py"
        tree = ast.parse(source.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "readpage_jina")
        page = {"path": "pages/test/content.md", "absolute_path": "/workspace/pages/test/content.md", "characters": 50000}
        calls = []
        def run_http(payload, **kwargs):
            calls.append(payload)
            return {"status_code": 200, "page": page}
        namespace = {
            "os": os, "json": json, "time": __import__("time"), "emit_trace": lambda *a, **k: None,
            "shared_sandbox_run": lambda: SimpleNamespace(run_http=run_http),
            "current_guard_context": lambda: capture_guard_context({}), "guard_fingerprint": guard_fingerprint,
            "_visit_read_max_retries": lambda: 1, "tool_http_config": lambda prefix: {"max_time": 30},
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        agent = SimpleNamespace(_build_jina_request=lambda url: ("https://reader/test", {}))
        with patch.dict(os.environ, {"SCIENCE_AGENT_READ_PAGE_MODE": "file", "SCIENCE_AGENT_TOOL_HTTP_BACKEND": "gateway_sandbox"}):
            result = json.loads(namespace["readpage_jina"](agent, "https://source/test", "find evidence"))
        self.assertEqual(result["path"], page["path"])
        self.assertNotIn("absolute_path", result)
        self.assertIn("do not prepend /workspace", result["instruction"])
        self.assertTrue(calls[0]["save_page"])
        self.assertIn("guard_snapshot", calls[0])
        self.assertNotIn("tiktoken", source.read_text())

    def test_python_gateway_uses_existing_observation_limit(self):
        source = ROOT / "inference/tool_python.py"
        tree = ast.parse(source.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "call" and any(isinstance(n, ast.Name) and n.id == "SANDBOX_GATEWAY_BACKEND" for n in ast.walk(node)))
        calls = []
        namespace = {
            "normalize_python_code_input": lambda params: params["code"], "python_request_blocked": lambda *args: False,
            "python_path_policy_error": lambda code: None, "PYTHON_SESSION_BACKEND": "gateway_sandbox",
            "SANDBOX_GATEWAY_BACKEND": "gateway_sandbox", "PYTHON_TOOL_TIMEOUT_SEC": 30,
            "shared_sandbox_run": lambda: SimpleNamespace(run_python=lambda *args, **kwargs: "x" * 10000),
            "SandboxGatewayError": gateway.SandboxGatewayError, "PYTHON_TOOL_NO_OUTPUT_MESSAGE": "no output",
            "_sha256_text": lambda text: "hash",
            "_truncate_python_tool_return": lambda text, **kwargs: calls.append(text) or "bounded",
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        result = namespace["call"](SimpleNamespace(), {"code": "print('x')"})
        self.assertEqual(result, "bounded")
        self.assertEqual(len(calls[0]), 10000)


if __name__ == "__main__":
    unittest.main()
