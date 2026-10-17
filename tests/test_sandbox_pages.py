import ast
from contextlib import nullcontext
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

    def test_inline_character_boundary_with_cache_and_recovery(self):
        for length in (5999, 6000, 6001):
            with self.subTest(length=length):
                text = "字" * length
                payload = dict(self.payload, source_url=f"https://source/{length}", recovery_backup=True)
                with patch.object(worker, "urlopen") as fetch:
                    fetch.return_value.__enter__.return_value = SimpleNamespace(status=200, read=lambda: text.encode())
                    first = self.request(payload)
                    cached = self.request(payload)
                fetch.assert_called_once()
                for result in (first, cached):
                    page = result["page"]
                    self.assertEqual(Path(page["path"]).read_text(encoding="utf-8"), text)
                    self.assertEqual(result["recovery_files"][page["path"]], text)
                    self.assertEqual(page["preview"], text[:1000])
                    if length <= 6000:
                        self.assertEqual(result["text"], text)
                    else:
                        self.assertNotIn("text", result)


    def test_recovery_transport_includes_full_original_and_cached_page(self):
        text = "original evidence\n" * 1000
        with patch.object(worker, "urlopen") as fetch:
            fetch.return_value.__enter__.return_value = SimpleNamespace(status=200, read=lambda: text.encode())
            payload = dict(self.payload, recovery_backup=True)
            first = self.request(payload)
            cached = self.request(payload)
        fetch.assert_called_once()
        path = first['page']['path']
        self.assertEqual(first['recovery_files'][path], text)
        self.assertEqual(cached['recovery_files'], first['recovery_files'])
        self.assertEqual(json.loads(first['recovery_files'][str(Path(path).with_name('source.json'))])['source_url'], self.payload['source_url'])

    def test_read_page_file_mode_skips_host_fetch_tokenizer_and_summary(self):
        source = ROOT / "inference/tool_visit_mcp_r1.py"
        tree = ast.parse(source.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "readpage_jina")
        page = {"path": "pages/test/content.md", "absolute_path": "/workspace/pages/test/content.md", "characters": 50000}
        calls = []
        def run_http(payload, **kwargs):
            calls.append(payload)
            result = {"status_code": 200, "page": page}
            if page["characters"] <= 6000:
                result["text"] = "字" * page["characters"]
            return result
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
        with patch.dict(os.environ, {"SCIENCE_AGENT_READ_PAGE_MODE": "file", "SCIENCE_AGENT_TOOL_HTTP_BACKEND": "gateway_sandbox"}):
            for length in (5999, 6000, 6001):
                page["characters"] = length
                returned = namespace["readpage_jina"](agent, "https://source/test", "find evidence")
                if length <= 6000:
                    self.assertEqual(returned, "字" * length)
                else:
                    self.assertEqual(json.loads(returned)["path"], page["path"])

    def test_public_call_preserves_inline_whitespace_and_cache(self):
        source = ROOT / "inference/tool_visit_mcp_r1.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Visit")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"call", "readpage_jina"}]
        from typing import Union, List
        namespace = dict(os=os, json=json, time=__import__("time"), Union=Union, List=List,
                         emit_trace=lambda *a, **k: None, normalize_url_input=lambda x: x,
                         guard_capture_item_scope=lambda *a, **k: nullcontext(),
                         guard_capture_url_scope=lambda *a, **k: nullcontext(),
                         is_url_blocked=lambda _: False, provider_strategy=lambda: "original",
                         capture_structured_basis=lambda *a, **k: None, GUARD_FORMAT_VISIT_FINAL_V1="test",
                         shared_sandbox_run=lambda: SimpleNamespace(run_http=lambda payload, **kw: self.request(payload)),
                         current_guard_context=lambda: capture_guard_context({}), guard_fingerprint=guard_fingerprint,
                         _visit_read_max_retries=lambda: 1, tool_http_config=lambda _: {"max_time": 30})
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
        agent = SimpleNamespace(_build_jina_request=lambda url: ("https://reader/test", {}))
        agent.readpage_jina = lambda *args: namespace["readpage_jina"](agent, *args)
        with patch.dict(os.environ, {"SCIENCE_AGENT_READ_PAGE_MODE": "file", "SCIENCE_AGENT_TOOL_HTTP_BACKEND": "gateway_sandbox"}):
            for length in (5999, 6000, 6001):
                text = "  " + "字" * (length - 4) + "\n\n"
                with patch.object(worker, "urlopen") as fetch:
                    fetch.return_value.__enter__.return_value = SimpleNamespace(status=200, read=lambda: text.encode())
                    params = {"url": f"https://source/whitespace/{length}", "goal": "evidence"}
                    first = namespace["call"](agent, params)
                    cached = namespace["call"](agent, params)
                fetch.assert_called_once()
                if length <= 6000:
                    self.assertEqual(first, text)
                    self.assertEqual(cached, text)
                else:
                    self.assertEqual(json.loads(first)["preview"], text[:1000])
                    self.assertTrue(json.loads(cached)["cached"])

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
