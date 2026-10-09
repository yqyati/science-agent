"""Small client for the heterogeneous-platform Sandbox gateway.

This module deliberately uses only the documented management and data-plane
interfaces.  It is kept independent from SandboxFusion's ``/v1/sessions``
protocol so the harness can select either backend explicitly.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import ssl
import hashlib
import posixpath
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextvars import ContextVar
from functools import wraps
from typing import Any


class SandboxGatewayError(RuntimeError):
    pass


class SandboxGatewayClient:
    def __init__(self, base_url: str, token: str, resource: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.resource = resource or None

    def _request(self, method: str, path: str, body: Any = None, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        query = urllib.parse.urlencode(params or {}, doseq=True)
        url = self.base_url + path + (("?" + query) if query else "")
        request_headers = {"Authorization": f"Bearer {self.token}"}
        if self.resource:
            request_headers["X-Sandbox-Resource"] = self.resource
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    raw = response.read()
                    return json.loads(raw.decode("utf-8")) if raw else None
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                # GET cannot execute Agent code; retry transient authorization
                # and gateway failures without replaying create/command requests.
                if method == "GET" and exc.code in {403, 502, 503, 504} and attempt < 2:
                    logging.getLogger(__name__).warning("Gateway GET retry path=%s status=%d attempt=%d/3", path, exc.code, attempt + 1)
                    time.sleep(attempt + 1)
                    continue
                raise SandboxGatewayError(f"HTTP {exc.code} {method} {path} (attempt {attempt + 1}/3): {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if method == "GET" and attempt < 2:
                    logging.getLogger(__name__).warning("Gateway GET retry path=%s error=%s attempt=%d/3", path, type(exc).__name__, attempt + 1)
                    time.sleep(attempt + 1)
                    continue
                raise SandboxGatewayError(f"gateway request failed {method} {path}: {exc}") from exc

    def create(self, *, image: str, entrypoint: list[str] | None = None, cpu: str = "2", memory: str = "4Gi", timeout: int = 1800, env: dict[str, str] | None = None, network_policy: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "image": {"uri": image},
            "entrypoint": entrypoint or ["sleep", "inf"],
            "resourceRequests": {"cpu": cpu, "memory": memory},
            "ports": [
                {"containerPort": 44772, "purpose": "system service"},
                {"containerPort": 18080, "purpose": "agent python worker"},
                {"containerPort": 18081, "purpose": "agent outbound http worker"},
            ],
            "timeout": timeout,
        }
        if env:
            payload["env"] = env
        if network_policy is not None:
            payload["networkPolicy"] = network_policy
        data = self._request("POST", "/v1/sandboxes", payload)
        sandbox_id = str((data or {}).get("id") or "")
        if not sandbox_id:
            raise SandboxGatewayError(f"create response has no sandbox id: {data!r}")
        return sandbox_id

    def detail(self, sandbox_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/sandboxes/{urllib.parse.quote(sandbox_id, safe='')}")

    def wait_running(self, sandbox_id: str, timeout: int = 360) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self.detail(sandbox_id)
            status = data.get("status") if isinstance(data, dict) else {}
            state = (status or {}).get("state") or data.get("state")
            if state == "Running":
                return
            if state in {"Failed", "Error", "Terminated"}:
                raise SandboxGatewayError(f"sandbox entered terminal state {state}: {data}")
            time.sleep(5)
        raise SandboxGatewayError(f"sandbox did not become Running within {timeout}s")

    def endpoint(self, sandbox_id: str, port: int = 44772) -> tuple[str, dict[str, str]]:
        data = self._request(
            "GET",
            f"/v1/sandboxes/{urllib.parse.quote(sandbox_id, safe='')}/endpoints/{port}",
            params={"use_server_proxy": "true"},
            headers={"OPEN-SANDBOX-API-KEY": self.token},
        )
        url = str((data or {}).get("endpoint") or "").rstrip("/")
        if not url:
            raise SandboxGatewayError(f"endpoint response has no endpoint: {data!r}")
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        request_headers = dict((data or {}).get("headers") or {})
        request_headers.setdefault("OPEN-SANDBOX-API-KEY", self.token)
        if self.resource:
            request_headers.setdefault("X-Sandbox-Resource", self.resource)
        return url, request_headers

    def command(self, sandbox_id: str, command: str, timeout: int = 180) -> str:
        url, headers = self.endpoint(sandbox_id)
        headers = dict(headers)
        headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url + "/command",
            data=json.dumps({"command": command, "user": "root"}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise SandboxGatewayError(f"command HTTP {exc.code}: {detail}") from exc

    @staticmethod
    def command_output(raw: str) -> str:
        """Extract stdout/stderr/error text from command event streams."""
        lines: list[str] = []
        for line in str(raw or "").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if line.strip():
                    lines.append(line)
                continue
            kind = event.get("type")
            if kind in {"stdout", "stderr"}:
                lines.append(str(event.get("text") or ""))
            elif kind == "error":
                detail = event.get("error") or event.get("text") or "command failed"
                lines.append(str(detail))
        return "\n".join(lines)

    def proxy_json(self, sandbox_id: str, port: int, path: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
        url, headers = self.endpoint(sandbox_id, port)
        request_headers = dict(headers)
        request_headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url + "/" + path.lstrip("/"), data=json.dumps(payload).encode("utf-8"), headers=request_headers, method="POST")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                # Only retry an explicit gateway connection failure, which
                # occurs before the worker accepts code. Timeouts are ambiguous.
                if exc.code == 502 and "Sandbox upstream connection failed" in detail and attempt < 2:
                    logging.getLogger(__name__).warning("Sandbox connection retry id=%s port=%d attempt=%d/3", sandbox_id, port, attempt + 1)
                    time.sleep(attempt + 1)
                    continue
                raise SandboxGatewayError(f"proxy HTTP {exc.code} (attempt {attempt + 1}/3): {detail}") from exc

    def proxy_http(self, sandbox_id: str, port: int, path: str, *, method: str = "GET", headers: dict[str, str] | None = None, body: bytes | None = None, timeout: int = 180) -> tuple[int, str]:
        url, endpoint_headers = self.endpoint(sandbox_id, port)
        request_headers = dict(endpoint_headers)
        request_headers.update(headers or {})
        request = urllib.request.Request(url + "/" + path.lstrip("/"), data=body, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")

    def upload_text(self, sandbox_id: str, remote_path: str, content: str) -> None:
        """Upload small text through the command plane without a second API."""
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        command = f"echo {encoded} | base64 -d > {self._shell_quote(remote_path)}"
        self.command(sandbox_id, command)

    def upload_file_bytes(self, sandbox_id: str, remote_path: str, content: bytes) -> None:
        # Keep each shell argument below Linux's per-argument size limit.
        for offset in range(0, max(1, len(content)), 48 * 1024):
            encoded = base64.b64encode(content[offset:offset + 48 * 1024]).decode("ascii")
            redirect = ">" if offset == 0 else ">>"
            command = f"printf %s '{encoded}' | base64 -d {redirect} {self._shell_quote(remote_path)}"
            self.command(sandbox_id, command)

    @staticmethod
    def _shell_quote(value: str) -> str:
        return "'" + str(value).replace("'", "'\\''") + "'"

    def delete(self, sandbox_id: str) -> None:
        self._request("DELETE", f"/v1/sandboxes/{urllib.parse.quote(sandbox_id, safe='')}")


class AgentSandboxRun:
    """One lazily-created, run-scoped Sandbox.

    The object is intentionally owned by one Agent run.  It must not be shared
    between concurrent runs.  ``close`` is idempotent and belongs in finally.
    """

    def __init__(self):
        self.client: SandboxGatewayClient | None = None
        self.sandbox_id = ""
        self._python_started = False
        self._http_started = False
        self._python_counter = 0
        self._worker_lock = threading.RLock()
        self._uploaded_inputs: dict[str, str] = {}
        self._vision_ready = False

    @property
    def enabled(self) -> bool:
        return bool(self._token())

    @staticmethod
    def _token() -> str:
        # platform_ad_auth.py injects OPEN_SANDBOX_GATEWAY_TOKEN.  The other
        # names are compatibility aliases for callers that already use them.
        for name in (
            "SCIENCE_AGENT_SANDBOX_GATEWAY_TOKEN",
            "OPEN_SANDBOX_GATEWAY_TOKEN",
            "GATEWAY_TOKEN",
            "TOKEN",
        ):
            value = os.getenv(name, "").strip()
            if value:
                return value
        return ""

    def ensure(self) -> str:
        if self.sandbox_id:
            return self.sandbox_id
        token = self._token()
        if not token:
            raise SandboxGatewayError(
                "No gateway token found. Run platform_ad_auth.py -- bash first."
            )
        self.client = SandboxGatewayClient(
            os.getenv("SCIENCE_AGENT_SANDBOX_BASE_URL", "http://10.12.111.133:10018"),
            token,
            os.getenv("SCIENCE_AGENT_SANDBOX_RESOURCE", "10.12.111.143/ailab-ma4agismall"),
        )
        sandbox_env = {
            name: os.environ[name]
            for name in (
                "API_BASE", "API_KEY", "SUMMARY_MODEL_NAME",
                "SERPER_API_KEY", "JINA_API_KEY", "JINA_API_KEYS",
            )
            if os.environ.get(name)
        }
        ca_bundle = self._ca_bundle()
        if ca_bundle:
            sandbox_env.update(SSL_CERT_FILE="/tmp/science-agent-ca.pem", REQUESTS_CA_BUNDLE="/tmp/science-agent-ca.pem")
        network_action = os.getenv("SCIENCE_AGENT_SANDBOX_NETWORK_DEFAULT_ACTION", "").strip().lower()
        if network_action and network_action not in {"allow", "deny"}:
            raise SandboxGatewayError("SCIENCE_AGENT_SANDBOX_NETWORK_DEFAULT_ACTION must be allow or deny")
        network_policy = None
        if network_action:
            # The platform rejects default-allow with an empty egress list.
            # Express allow-all as explicit IPv4/IPv6 rules instead.
            network_policy = {"defaultAction": "deny", "egress": []}
            if network_action == "allow":
                network_policy["egress"] = [
                    {"action": "allow", "target": "0.0.0.0/0"},
                    {"action": "allow", "target": "::/0"},
                ]
        self.sandbox_id = self.client.create(
            image=os.getenv("SCIENCE_AGENT_SANDBOX_IMAGE", "registry2-cloud.d.pjlab.org.cn/ccr-sandbox/claude-code:v1.1.1"),
            cpu=os.getenv("SCIENCE_AGENT_SANDBOX_CPU", "2"),
            memory=os.getenv("SCIENCE_AGENT_SANDBOX_MEMORY", "4Gi"),
            timeout=int(os.getenv("SCIENCE_AGENT_SANDBOX_TIMEOUT", "1800")),
            env=sandbox_env or None,
            network_policy=network_policy,
        )
        try:
            self.client.wait_running(self.sandbox_id, int(os.getenv("SCIENCE_AGENT_SANDBOX_START_TIMEOUT", "360")))
            self.client.command(self.sandbox_id, "mkdir -p /workspace/pages")
            if ca_bundle:
                self.client.upload_file_bytes(self.sandbox_id, "/tmp/science-agent-ca.pem", ca_bundle)
            dns_servers = [value.strip() for value in os.getenv("SCIENCE_AGENT_SANDBOX_DNS_SERVERS", "").split(",") if value.strip()]
            if dns_servers:
                self._configure_dns(dns_servers)
        except Exception:
            self.close()
            raise
        return self.sandbox_id

    @staticmethod
    def _ca_bundle() -> bytes:
        source = os.getenv("SCIENCE_AGENT_SANDBOX_CA_BUNDLE", "").strip()
        if not source:
            return b""
        if source == "auto":
            paths = [ssl.get_default_verify_paths().cafile, "/etc/ssl/certs/ca-certificates.crt"]
            try:
                import certifi
                paths.append(certifi.where())
            except ImportError:
                pass
        else:
            paths = [source]
        certificates = {}
        for path in dict.fromkeys(paths):
            if path and os.path.isfile(path):
                with open(path, "rb") as handle:
                    raw = handle.read()
                marker = b"-----END CERTIFICATE-----"
                for block in raw.split(marker):
                    start = block.find(b"-----BEGIN CERTIFICATE-----")
                    if start >= 0:
                        certificate = block[start:].strip() + b"\n" + marker + b"\n"
                        certificates[certificate] = None
        bundle = b"".join(certificates)
        if not bundle:
            raise SandboxGatewayError("No CA certificates found for SCIENCE_AGENT_SANDBOX_CA_BUNDLE=" + source)
        # Validate the bundle locally before sending it to the Sandbox.
        ssl.create_default_context(cadata=bundle.decode("ascii"))
        return bundle

    def _configure_dns(self, servers: list[str]) -> None:
        assert self.client is not None
        remote = "/tmp/science-agent-dns.py"
        with open(os.path.join(os.path.dirname(__file__), "sandbox_dns.py"), "rb") as handle:
            self.client.upload_file_bytes(self.sandbox_id, remote, handle.read())
        command = "python3 " + self.client._shell_quote(remote) + " " + " ".join(self.client._shell_quote(server) for server in servers)
        raw = self.client.command(self.sandbox_id, command, timeout=45)
        output = self.client.command_output(raw)
        try:
            result = json.loads(output.splitlines()[0])
        except (ValueError, IndexError) as exc:
            raise SandboxGatewayError(f"Sandbox DNS configuration returned an invalid result: {output[:1000]}") from exc
        if result.get("error"):
            raise SandboxGatewayError("Sandbox DNS configuration failed: " + str(result["error"]))

    def _start_worker(self, kind: str, port: int) -> str:
        with self._worker_lock:
            sandbox_id = self.ensure()
            assert self.client is not None
            flag = f"_{kind}_started"
            if getattr(self, flag):
                return sandbox_id
            worker = os.getenv(f"SCIENCE_AGENT_SANDBOX_{kind.upper()}_WORKER", f"/tmp/sandbox_{kind}_worker.py")
            if os.getenv("SCIENCE_AGENT_SANDBOX_UPLOAD_WORKER", "true").lower() in {"1", "true", "yes", "on"}:
                local_worker = os.path.join(os.path.dirname(__file__), f"sandbox_{kind}_worker.py")
                with open(local_worker, "rb") as handle:
                    self.client.upload_file_bytes(sandbox_id, worker, handle.read())
            if kind == "http":
                guard_file = os.path.join(os.path.dirname(__file__), "tool_url_guard.py")
                with open(guard_file, "rb") as handle:
                    self.client.upload_file_bytes(sandbox_id, "/tmp/tool_url_guard.py", handle.read())
            self.client.command(sandbox_id, f"cd /workspace && nohup python3 {self.client._shell_quote(worker)} >/tmp/agent-{kind}-worker.log 2>&1 &", timeout=30)
            deadline = time.monotonic() + float(os.getenv("SCIENCE_AGENT_SANDBOX_WORKER_START_TIMEOUT", "30"))
            last_error = "not ready"
            while time.monotonic() < deadline:
                try:
                    status, body = self.client.proxy_http(sandbox_id, port, "/health", timeout=5)
                    if status == 200 and body == "ok":
                        setattr(self, flag, True)
                        return sandbox_id
                    last_error = f"HTTP {status}"
                except (SandboxGatewayError, urllib.error.URLError, TimeoutError) as exc:
                    last_error = str(exc)
                time.sleep(0.5)
            raise SandboxGatewayError(f"{kind} worker on port {port} failed health check: {last_error}; inspect /tmp/agent-{kind}-worker.log")

    def run_http(self, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
        sandbox_id = self._start_worker("http", 18081)
        assert self.client is not None
        payload = dict(payload, timeout=timeout)
        return self.client.proxy_json(sandbox_id, 18081, "/request", payload, timeout=timeout + 10)

    def run_python(self, code: str, timeout: int = 180) -> str:
        sandbox_id = self._start_worker("python", 18080)
        assert self.client is not None
        data = self.client.proxy_json(sandbox_id, 18080, "/execute", {"code": code}, timeout=timeout)
        output = str(data.get("stdout") or "") + str(data.get("stderr") or "")
        if data.get("error"):
            output += str(data["error"])
        return output

    def run_visual(self, code: str, files: dict[str, str], timeout: int = 180) -> dict[str, Any]:
        sandbox_id = self._start_worker("python", 18080)
        assert self.client is not None
        with self._worker_lock:
            if not self._vision_ready:
                data = self.client.proxy_json(sandbox_id, 18080, "/prepare-vision", {
                    "install": os.getenv("SCIENCE_AGENT_SANDBOX_INSTALL_VISION_DEPS", "true").lower() in {"1", "true", "yes", "on"},
                    "index_url": os.getenv("SCIENCE_AGENT_SANDBOX_PIP_INDEX_URL", ""),
                }, timeout=330)
                if not data.get("ready"):
                    raise SandboxGatewayError(str(data.get("error") or "Vision dependencies not ready"))
                self._vision_ready = True
        for path, encoded in files.items():
            normalized = posixpath.normpath(path)
            if normalized.startswith(("/", "../")) or normalized in {".", ".."}:
                raise SandboxGatewayError("Visual input must use a relative Sandbox path")
            content = base64.b64decode(encoded, validate=True)
            digest = hashlib.sha256(content).hexdigest()
            if self._uploaded_inputs.get(normalized) == digest:
                continue
            directory = posixpath.dirname(normalized) or "."
            self.client.command(sandbox_id, "mkdir -p " + self.client._shell_quote("/workspace/" + directory))
            self.client.upload_file_bytes(sandbox_id, "/workspace/" + normalized, content)
            self._uploaded_inputs[normalized] = digest
        return self.client.proxy_json(sandbox_id, 18080, "/execute", {"code": code, "visual": True}, timeout=timeout)

    def close(self) -> None:
        if self.client is not None and self.sandbox_id:
            for attempt in range(3):
                try:
                    self.client.delete(self.sandbox_id)
                    break
                except Exception:
                    logging.getLogger(__name__).warning(
                        "Sandbox deletion failed id=%s attempt=%d/3", self.sandbox_id, attempt + 1,
                    )
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            self.sandbox_id = ""
            self._python_started = False
            self._http_started = False
            self._uploaded_inputs.clear()
            self._vision_ready = False


_SHARED_RUN: ContextVar[AgentSandboxRun | None] = ContextVar("science_agent_sandbox_run", default=None)


def shared_sandbox_run() -> AgentSandboxRun:
    run = _SHARED_RUN.get()
    if run is None:
        run = AgentSandboxRun()
        _SHARED_RUN.set(run)
    return run


def close_shared_sandbox_run() -> None:
    run = _SHARED_RUN.get()
    if run is not None:
        run.close()
        _SHARED_RUN.set(None)


def sandbox_run_scope(function):
    """Give each gateway-enabled Agent invocation a fresh run and cleanup."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        enabled = any(os.getenv(name, "").strip().lower() == "gateway_sandbox" for name in (
            "PYTHON_SESSION_BACKEND", "SCIENCE_AGENT_TOOL_HTTP_BACKEND",
        ))
        if not enabled:
            return function(*args, **kwargs)
        run = AgentSandboxRun()
        token = _SHARED_RUN.set(run)
        try:
            return function(*args, **kwargs)
        finally:
            try:
                run.close()
            finally:
                _SHARED_RUN.reset(token)
    return wrapped
