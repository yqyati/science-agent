"""Tiny stateful Python worker for a single Agent Sandbox.

Run inside the Sandbox image with ``python3 sandbox_python_worker.py``.  The
gateway endpoint proxies port 18080.  This is intentionally small: execution
policy remains in the harness and the worker only owns one run's globals.
"""

from __future__ import annotations

import io
import ast
import base64
import json
import os
import importlib
import importlib.util
import subprocess
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer


GLOBALS: dict[str, object] = {"__name__": "__agent_sandbox__"}


def prepare_vision(payload):
    packages = {"PIL": "Pillow", "matplotlib": "matplotlib", "IPython": "ipython"}
    missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
    if missing and not payload.get("install", True):
        return {"error": "Missing Vision dependencies: " + ", ".join(missing)}
    if missing:
        command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--timeout", "30", "--retries", "1"]
        if payload.get("index_url"):
            command += ["--index-url", payload["index_url"]]
        command += missing
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"error": "Vision dependency installation failed: " + str(exc)}
        if result.returncode:
            return {"error": "Vision dependency installation failed: " + (result.stderr or result.stdout)[-2500:]}
        importlib.invalidate_caches()
    try:
        for module in packages:
            importlib.import_module(module)
    except Exception as exc:
        return {"error": "Vision dependency import failed: " + str(exc)}
    return {"ready": True, "installed": missing}


def execute_payload(payload):
    code = str(payload.get("code") or "")
    stdout, stderr = io.StringIO(), io.StringIO()
    images, error = [], ""
    patches = []
    pyplot = None
    previous_figures = set()

    def capture(obj):
        if len(images) >= 8:
            return
        buffer = io.BytesIO()
        if hasattr(obj, "savefig"):
            obj.savefig(buffer, format="png")
        elif hasattr(obj, "save") and hasattr(obj, "size"):
            obj.save(buffer, format="PNG")
        elif hasattr(obj, "_repr_png_"):
            data = obj._repr_png_()
            if isinstance(data, tuple):
                data = data[0]
            if isinstance(data, str):
                data = base64.b64decode(data)
            if data:
                buffer.write(data)
        else:
            print(repr(obj))
            return
        raw = buffer.getvalue()
        if raw:
            images.append({"mime_type": "image/png", "data": base64.b64encode(raw).decode("ascii")})

    def display(*objects, **kwargs):
        for obj in objects:
            capture(obj)

    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            if payload.get("visual"):
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as pyplot
                previous_figures = set(pyplot.get_fignums())

                def show(*args, **kwargs):
                    for number in pyplot.get_fignums():
                        capture(pyplot.figure(number))
                    pyplot.close("all")

                patches.append((pyplot, "show", pyplot.show))
                pyplot.show = show
                try:
                    import IPython.display as ipython_display
                except ImportError:
                    ipython_display = None
                if ipython_display is not None:
                    patches.append((ipython_display, "display", ipython_display.display))
                    ipython_display.display = display
                GLOBALS["display"] = display
            tree = ast.parse(code, filename="<agent>", mode="exec")
            tail = None
            if payload.get("visual") and tree.body and isinstance(tree.body[-1], ast.Expr):
                tail = tree.body.pop().value
            exec(compile(tree, "<agent>", "exec"), GLOBALS, GLOBALS)
            if tail is not None:
                value = eval(compile(ast.Expression(tail), "<agent>", "eval"), GLOBALS, GLOBALS)
                if value is not None:
                    capture(value)
            if pyplot is not None:
                for number in set(pyplot.get_fignums()) - previous_figures:
                    capture(pyplot.figure(number))
                    pyplot.close(number)
        except Exception:
            error = traceback.format_exc()
        finally:
            for module, name, original in reversed(patches):
                setattr(module, name, original)
    return {"stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "error": error, "images": images}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_GET(self):  # noqa: N802
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):  # noqa: N802
        if self.path not in {"/execute", "/prepare-vision"}:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            body = json.dumps(
                prepare_vision(payload) if self.path == "/prepare-vision" else execute_payload(payload),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_error(500, str(exc))


if __name__ == "__main__":
    port = int(os.getenv("SANDBOX_PYTHON_WORKER_PORT", "18080"))
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
