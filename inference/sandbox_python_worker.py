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
import threading
import hashlib
import signal
import time
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


GLOBALS: dict[str, object] = {"__name__": "__agent_sandbox__"}
EXECUTION_LOCK = threading.Lock()
OPERATIONS_LOCK = threading.Lock()
OPERATIONS = {}
REPLAY = threading.local()
INTERRUPTS = {}
WORKER_ID = ""


class ExecutionInterrupted(BaseException):
    pass


def replay_audit(event, args):
    """Prevent replay from repeating network/process side effects.

    This is a recovery guard, not a security boundary for hostile Python.
    Reads needed by imports are allowed; writes must stay in /workspace.
    """
    if not getattr(REPLAY, "active", False):
        return
    if event.startswith(("socket.", "subprocess.", "os.exec", "os.spawn")) or event in {
        "os.system", "os.fork", "os.forkpty", "os.putenv", "os.unsetenv", "os.chdir",
        "os.symlink", "os.link", "ctypes.dlopen", "ctypes.dlsym",
    }:
        REPLAY.violation = event
        raise RuntimeError("Recovery replay blocked external operation: " + event)
    paths = []
    if event == "open":
        path, mode, flags = args
        if isinstance(path, int):
            REPLAY.violation = event
            raise RuntimeError("Recovery replay cannot access an existing file descriptor")
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            paths = [path]
    elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate", "os.utime"}:
        paths = [args[0]]
    elif event in {"os.rename", "os.replace"}:
        paths = list(args[:2])
    for path in paths:
        resolved = os.path.realpath(os.fsdecode(path))
        if not resolved.startswith("/workspace/"):
            REPLAY.violation = event
            raise RuntimeError("Recovery replay blocked write outside /workspace")


def execute_operation(payload):
    operation_id = payload.get("operation_id")
    if not operation_id:
        with EXECUTION_LOCK:
            return execute_payload(payload)
    fingerprint = hashlib.sha256(json.dumps(
        {k: payload.get(k) for k in ("code", "visual", "replay")}, sort_keys=True,
    ).encode()).hexdigest()
    with OPERATIONS_LOCK:
        previous = OPERATIONS.get(operation_id)
        if previous:
            if previous["fingerprint"] != fingerprint:
                raise ValueError("operation_id reused for different code")
            return dict(previous)
        OPERATIONS[operation_id] = {"state": "running", "fingerprint": fingerprint}
        interrupted = INTERRUPTS[operation_id] = threading.Event()
    with EXECUTION_LOCK:
        REPLAY.active = bool(payload.get("replay"))
        REPLAY.violation = ""
        try:
            def check_interrupt(frame, event, arg):
                if interrupted.is_set() and frame.f_code.co_filename == "<agent>":
                    raise ExecutionInterrupted("Python execution interrupted after response timeout")
                return check_interrupt if frame.f_code.co_filename == "<agent>" else None

            sys.settrace(check_interrupt)
            try:
                result = execute_payload(payload)
            finally:
                sys.settrace(None)
            if interrupted.is_set():
                result["interrupted"] = True
            if REPLAY.violation:
                result["replay_blocked"] = REPLAY.violation
        finally:
            REPLAY.active = False
    with OPERATIONS_LOCK:
        OPERATIONS[operation_id] = {"state": "completed", "fingerprint": fingerprint, "result": result}
        return dict(OPERATIONS[operation_id])


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
        except (Exception, ExecutionInterrupted, KeyboardInterrupt, SystemExit):
            error = traceback.format_exc()
        finally:
            for module, name, original in reversed(patches):
                setattr(module, name, original)
    return {"stdout": stdout.getvalue(), "stderr": stderr.getvalue(), "error": error, "images": images}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/operations/"):
            operation_id = self.path.removeprefix("/operations/")
            with OPERATIONS_LOCK:
                body = json.dumps(OPERATIONS.get(operation_id, {"state": "unknown"})).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/identity":
            body = WORKER_ID.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/interrupt/"):
            operation_id = self.path.removeprefix("/interrupt/")
            with OPERATIONS_LOCK:
                state = OPERATIONS.get(operation_id, {}).get("state", "unknown")
                if state == "running":
                    INTERRUPTS[operation_id].set()
            body = json.dumps({"state": state}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path not in {"/execute", "/prepare-vision"}:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            body = json.dumps(
                prepare_vision(payload) if self.path == "/prepare-vision" else execute_operation(payload),
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", default="")
    parser.add_argument("--stop", default="")
    args = parser.parse_args()
    if args.stop:
        # Only terminate the recorded process group, after checking PID reuse.
        record = json.loads(Path(f"/tmp/agent-python-{args.stop}.json").read_text())
        pid = record["pid"]
        proc = Path(f"/proc/{pid}/stat")
        if proc.exists():
            stat = proc.read_text().split(") ", 1)[1].split()
            if stat[19] != record["start"] or os.getpgid(pid) != pid:
                raise RuntimeError("Python worker identity mismatch")
            os.killpg(pid, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while proc.exists() and proc.read_text().split(") ", 1)[1].split()[0] != "Z":
                if time.monotonic() >= deadline:
                    raise RuntimeError("Python worker termination unconfirmed")
                time.sleep(0.05)
        print(json.dumps({"stopped": args.stop}))
        raise SystemExit(0)
    WORKER_ID = args.identity
    if WORKER_ID:
        if os.getsid(0) != os.getpid():
            os.setsid()
        record = {"pid": os.getpid(), "start": Path(f"/proc/{os.getpid()}/stat").read_text().split(") ", 1)[1].split()[19]}
        Path(f"/tmp/agent-python-{WORKER_ID}.json").write_text(json.dumps(record))
    sys.addaudithook(replay_audit)
    port = int(os.getenv("SANDBOX_PYTHON_WORKER_PORT", "18080"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
