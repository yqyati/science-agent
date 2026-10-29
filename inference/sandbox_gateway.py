"""Small client for the heterogeneous-platform Sandbox gateway.

This module deliberately uses only the documented management and data-plane
interfaces.  It is kept independent from SandboxFusion's ``/v1/sessions``
protocol so the harness can select either backend explicitly.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
import shlex
import ssl
import hashlib
import posixpath
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any
from sandbox_recovery import RecoveryJournal, result_digest


class SandboxGatewayError(RuntimeError):
    pass


class SandboxQuotaExceededError(SandboxGatewayError):
    """The gateway explicitly rejected creation because the resource pool is full."""


class SandboxInactiveError(SandboxGatewayError):
    """Management API confirmed that the addressed Sandbox is unavailable."""


class SandboxStateError(SandboxGatewayError):
    """Execution state prevents dispatch; transport retries cannot resolve it."""


class SandboxGatewayClient:
    def __init__(self, base_url: str, token: str, resource: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.resource = resource or None
        self._token_lock = threading.Lock()
        self._token_file = Path(os.getenv(
            "SANDBOX_LOCAL_ENV_FILE", str(Path(__file__).resolve().parents[1] / "sandbox.local.env"),
        )).expanduser().resolve()

    @staticmethod
    def _token_expiry(token: str) -> float | None:
        # Unverified JWT claims only schedule credential reloads; the gateway verifies auth.
        try:
            part = token.split(".")[1]
            value = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))).get("exp")
            return float(value) if isinstance(value, (int, float)) else None
        except (IndexError, ValueError, TypeError, AttributeError):
            return None

    def _refresh_token(self, *, force: bool = False) -> bool:
        with self._token_lock:
            expiry = self._token_expiry(self.token)
            now = time.time()
            if not force and (expiry is None or expiry > now + 30):
                return False
            candidate = None
            try:
                for line in self._token_file.read_text().splitlines():
                    line = line.strip().removeprefix("export ").strip()
                    key, sep, value = line.partition("=")
                    if sep and key.strip() == "OPEN_SANDBOX_GATEWAY_TOKEN":
                        parts = shlex.split(value, comments=True)
                        candidate = parts[0] if len(parts) == 1 else None
            except (OSError, ValueError, UnicodeError):
                # Never source shell code or log credentials from the environment file.
                logging.getLogger(__name__).warning("Cannot read Sandbox credential file: %s", self._token_file)
            candidate_expiry = self._token_expiry(candidate) if candidate else None
            usable = candidate_expiry > now if candidate_expiry is not None else expiry is None
            if candidate and candidate != self.token and usable:
                self.token = candidate
                logging.getLogger(__name__).info("Reloaded Sandbox credential from %s", self._token_file)
                return True
            if expiry is not None and expiry <= now:
                raise SandboxGatewayError(
                    f"Sandbox token expired; update OPEN_SANDBOX_GATEWAY_TOKEN in {self._token_file}. "
                    "No valid replacement credential was found."
                )
            return False

    def _request(self, method: str, path: str, body: Any = None, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        query = urllib.parse.urlencode(params or {}, doseq=True)
        url = self.base_url + path + (("?" + query) if query else "")
        request_headers = {}
        if self.resource:
            request_headers["X-Sandbox-Resource"] = self.resource
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        for attempt in range(3):
            self._refresh_token()
            request_headers["Authorization"] = f"Bearer {self.token}"
            if "OPEN-SANDBOX-API-KEY" in request_headers:
                request_headers["OPEN-SANDBOX-API-KEY"] = self.token
            request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    raw = response.read()
                    return json.loads(raw.decode("utf-8")) if raw else None
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                # An already-removed sandbox needs no further cleanup.
                if method == "DELETE" and exc.code == 404 and "sandbox not found" in detail.lower():
                    return None
                # Read-only lookups can safely repeat after an auth rejection. Tool/command
                # POSTs are never replayed here, since their execution outcome may be unknown.
                if exc.code == 401 and method == "GET" and attempt == 0 and self._refresh_token(force=True):
                    continue
                # GET cannot execute Agent code; retry transient authorization
                # and gateway failures without replaying create/command requests.
                if method == "GET" and exc.code in {403, 502, 503, 504} and attempt < 2:
                    logging.getLogger(__name__).warning("Gateway GET retry path=%s status=%d attempt=%d/3", path, exc.code, attempt + 1)
                    time.sleep(attempt + 1)
                    continue
                error = SandboxGatewayError
                if method == "POST" and path == "/v1/sandboxes" and exc.code == 429:
                    try:
                        info = json.loads(detail).get("detail", {})
                    except (ValueError, AttributeError):
                        info = {}
                    if isinstance(info, dict) and info.get("code") == "QUOTA_EXCEEDED":
                        error = SandboxQuotaExceededError
                if method == "GET" and path.startswith("/v1/sandboxes/"):
                    try:
                        info = json.loads(detail).get("detail", {})
                    except (ValueError, AttributeError):
                        info = {}
                    if (exc.code == 409 and isinstance(info, dict) and info.get("code") == "SANDBOX_NOT_ACTIVE") or (
                        exc.code == 404 and "sandbox not found" in detail.lower()
                    ):
                        error = SandboxInactiveError
                raise error(f"HTTP {exc.code} {method} {path} (attempt {attempt + 1}/3): {detail}") from exc
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
        retry_delays = (1, 5, 10)
        quota_groups = {"ailab-ai4sgi": "ailab-ma4agismall", "ailab-ma4agismall": "ailab-ai4sgi"}
        quota_switched = False
        for attempt in range(len(retry_delays) + 1):
            try:
                data = self._request("POST", "/v1/sandboxes", payload)
                break
            except SandboxGatewayError as exc:
                cause = exc.__cause__
                transient = (cause.code in {403, 429, 500, 502, 503, 504}
                             if isinstance(cause, urllib.error.HTTPError)
                             else isinstance(cause, (urllib.error.URLError, TimeoutError)))
                if not transient:
                    raise
                if attempt == len(retry_delays):
                    raise SandboxGatewayError(f"Sandbox create failed after {attempt + 1} attempts: {exc}") from exc
                cluster, separator, group = (self.resource or "").rpartition("/")
                if isinstance(exc, SandboxQuotaExceededError) and separator and group in quota_groups:
                    previous = self.resource
                    self.resource = cluster + "/" + quota_groups[group]
                    delay = retry_delays[attempt] if quota_switched else 0
                    quota_switched = True
                    logging.getLogger(__name__).warning(
                        "Sandbox create quota exhausted resource=%s; retry resource=%s attempt=%d/4 delay=%ss",
                        previous, self.resource, attempt + 2, delay,
                    )
                    if delay:
                        time.sleep(delay)
                    continue
                # Creation retries must not replay command or tool POSTs. Without
                # server idempotency, a lost response can leave an unknown instance.
                logging.getLogger(__name__).warning(
                    "Sandbox create retry attempt=%d/4 status=%s error=%s delay=%ss; "
                    "a lost response may leave an instance without a known ID",
                    attempt + 1, getattr(cause, "code", None), type(cause).__name__, retry_delays[attempt],
                )
                time.sleep(retry_delays[attempt])
        sandbox_id = str((data or {}).get("id") or "")
        if not sandbox_id:
            raise SandboxGatewayError(f"create response has no sandbox id: {data!r}")
        return sandbox_id

    def detail(self, sandbox_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/sandboxes/{urllib.parse.quote(sandbox_id, safe='')}")

    def wait_running(self, sandbox_id: str, timeout: int = 360) -> None:
        deadline = time.monotonic() + timeout
        last_status = "not Running"
        while time.monotonic() < deadline:
            data = self.detail(sandbox_id)
            status = data.get("status") if isinstance(data, dict) else {}
            state = (status or {}).get("state") or data.get("state")
            if state == "Running":
                # Running can precede proxy authorization and command readiness.
                # Probe with GET before sending bootstrap commands or Agent code.
                try:
                    code, _ = self.proxy_http(sandbox_id, 44772, "/ping", timeout=10)
                except SandboxInactiveError:
                    raise
                except (SandboxGatewayError, urllib.error.URLError, TimeoutError) as exc:
                    last_status = "command ping " + type(exc).__name__
                else:
                    if code == 200:
                        return
                    if code not in {403, 429, 500, 502, 503, 504}:
                        raise SandboxGatewayError(f"command service ping HTTP {code}")
                    last_status = f"command ping HTTP {code}"
                logging.getLogger(__name__).warning(
                    "Sandbox waiting for command service id=%s status=%s", sandbox_id, last_status,
                )
            else:
                last_status = str(state)
            if state in {"Failed", "Error", "Terminated"}:
                raise SandboxGatewayError(f"sandbox entered terminal state {state}: {data}")
            time.sleep(5)
        raise SandboxGatewayError(f"sandbox did not become Running with ready command service within {timeout}s: {last_status}")

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
        retry_delays = (1, 5, 10)
        for attempt in range(len(retry_delays) + 1):
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
                # This explicit connection failure precedes command dispatch.
                # Other errors/timeouts can hide execution; never replay them.
                if exc.code == 502 and "Sandbox upstream connection failed" in detail and attempt < len(retry_delays):
                    logging.getLogger(__name__).warning(
                        "Sandbox command connection retry id=%s attempt=%d/4 delay=%ss",
                        sandbox_id, attempt + 1, retry_delays[attempt],
                    )
                    time.sleep(retry_delays[attempt])
                    continue
                raise SandboxGatewayError(f"command HTTP {exc.code} (attempt {attempt + 1}/4): {detail}") from exc

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
        self._recovery_enabled = os.getenv("SCIENCE_AGENT_SANDBOX_RECOVERY", "1").lower() in {"1", "true", "yes"}
        self._journal = None
        self._recovery_count = 0
        self._recovery_notice = []
        self._pending_python = None
        self._python_worker_id = ""
        self._python_restart_needed = False
        self._recovery_broken = False
        self._create_options = None

    def _get_journal(self):
        if self._journal is None:
            root = os.getenv("SCIENCE_AGENT_SANDBOX_RECOVERY_DIR")
            if not root:
                root = os.path.join(os.getenv("OUTPUT_ROOT", "/tmp"), "sandbox_recovery")
            self._journal = RecoveryJournal(root)
        return self._journal

    def take_recovery_notice(self):
        with self._worker_lock:
            notice = "\n".join(self._recovery_notice)
            self._recovery_notice.clear()
            return notice

    def _restore(self):
        """Called only after an explicit management-plane inactive response."""
        maximum = int(os.getenv("SCIENCE_AGENT_SANDBOX_RECOVERY_MAX", "2"))
        if self._recovery_count >= maximum:
            raise SandboxStateError("Sandbox recovery limit reached; environment remains unavailable")
        self._recovery_count += 1
        journal = self._get_journal()
        old_id = self.sandbox_id
        events = list(journal.events)
        self.sandbox_id = ""
        self._python_started = self._http_started = self._vision_ready = False
        self._uploaded_inputs.clear()
        self._pending_python = None
        self._python_restart_needed = False
        self._recovery_broken = True
        restored = 0
        failure_reason = "replacement_creation_failed"
        journal.record({"kind": "recovery_start", "old_id": old_id})
        try:
            self.ensure()
            deadline = time.monotonic() + float(os.getenv("SCIENCE_AGENT_SANDBOX_REPLAY_TIMEOUT", "120"))
            completed = {e["operation_id"]: e for e in events if e["kind"] == "python_result"}
            cancelled = {e["operation_id"] for e in events if e["kind"] in {"python_cancelled", "python_interrupted"}}
            partial_history = any(e["kind"] in {"python_interrupted", "python_session_reset"} for e in events)
            for event in events:
                failure_reason = "replay_time_budget_exhausted"
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SandboxGatewayError("Replay time budget exhausted")
                if event["kind"] == "barrier":
                    failure_reason = "earlier_recovery_incomplete"
                    raise SandboxGatewayError("Earlier environment state was incomplete")
                if event["kind"] == "file":
                    failure_reason = "file_restore_failed"
                    path = "/workspace/" + event["path"]
                    self.client.command(self.sandbox_id, "mkdir -p " + self.client._shell_quote(posixpath.dirname(path)), timeout=min(30, remaining))
                    self.client.upload_file_bytes(self.sandbox_id, path, journal.read(event))
                elif event["kind"] == "python_start":
                    if event["operation_id"] in cancelled:
                        continue
                    result = completed.get(event["operation_id"])
                    if result is None:
                        failure_reason = "historical_execution_outcome_unknown"
                        raise SandboxGatewayError("Historical Python execution outcome is unknown")
                    if event["payload"].get("visual"):
                        failure_reason = "visual_replay_unsupported"
                        raise SandboxGatewayError("Visual Python replay is not supported in this first version")
                    failure_reason = "python_replay_request_failed"
                    sid = self._start_worker("python", 18080)
                    replay = dict(event["payload"], replay=True, operation_id=uuid.uuid4().hex)
                    self._pending_python = replay["operation_id"]
                    data = self.client.proxy_json(sid, 18080, "/execute", replay, timeout=max(1, min(event["timeout"], deadline - time.monotonic())))
                    if data.get("state") == "completed":
                        self._pending_python = None
                    if data.get("state") != "completed" or result_digest(data["result"]) != result["sha256"]:
                        failure_reason = "python_replay_output_mismatch"
                        if data.get("result", {}).get("replay_blocked"):
                            failure_reason = "python_replay_operation_blocked"
                        raise SandboxGatewayError("Python replay output differs from the saved result")
                    restored += 1
            self._recovery_broken = partial_history
            notice = (f"[Sandbox recovery] Environment replaced. Restored files and replayed {restored} Python calls "
                      "with matching recorded outputs. Hidden-state equivalence is unverified. "
                      "The current request was not completed.")
            if partial_history:
                notice += (" Recovery is partial: interrupted code and earlier-session code were not replayed. "
                           "Their unbacked file changes and interrupted memory changes are not restored.")
        except Exception as exc:
            journal.record({"kind": "barrier", "reason": type(exc).__name__})
            notice = (f"[Sandbox recovery] Recovery incomplete; replacement_available={bool(self.sandbox_id)}; "
                      f"replayed_python_calls={restored}; reason={failure_reason}; error_type={type(exc).__name__}. "
                      "Variables/files may be missing or partially modified. "
                      "The current request was not completed. Earlier unconfirmed execution outcomes remain unknown.")
        journal.record({"kind": "recovery_end", "new_id": self.sandbox_id, "restored_calls": restored,
                        "complete": not self._recovery_broken})
        self._recovery_notice.append(notice)
        # The old instance may merely be stopped; deletion is best-effort.
        try:
            if old_id:
                self.client.delete(old_id)
        except Exception:
            logging.getLogger(__name__).warning("Could not clean replaced Sandbox id=%s", old_id)
        raise SandboxStateError(notice)

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
            os.getenv("SCIENCE_AGENT_SANDBOX_RESOURCE", "10.12.111.143/ailab-ai4sgi"),
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
        options = dict(
            image=os.getenv("SCIENCE_AGENT_SANDBOX_IMAGE", "registry.h.pjlab.org.cn/ailab-ai4sgi-ailab_ai4sgi_pool/yqy:science-agent-cpu-20261010"),
            cpu=os.getenv("SCIENCE_AGENT_SANDBOX_CPU", "2"),
            memory=os.getenv("SCIENCE_AGENT_SANDBOX_MEMORY", "4Gi"),
            timeout=int(os.getenv("SCIENCE_AGENT_SANDBOX_TIMEOUT", "1800")),
            env=sandbox_env or None,
            network_policy=network_policy,
        )
        if self._create_options is None:
            self._create_options = options
        self.sandbox_id = self.client.create(**self._create_options)
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
            arguments = ""
            if kind == "python":
                self._python_worker_id = uuid.uuid4().hex
                arguments = " --identity " + self._python_worker_id
            self.client.command(sandbox_id, f"cd /workspace && nohup python3 {self.client._shell_quote(worker)}{arguments} >/tmp/agent-{kind}-worker.log 2>&1 &", timeout=30)
            deadline = time.monotonic() + float(os.getenv("SCIENCE_AGENT_SANDBOX_WORKER_START_TIMEOUT", "30"))
            last_error = "not ready"
            while time.monotonic() < deadline:
                try:
                    status, body = self.client.proxy_http(sandbox_id, port, "/health", timeout=5)
                    if status == 200 and body == "ok":
                        if kind == "python":
                            identity_status, identity = self.client.proxy_http(sandbox_id, port, "/identity", timeout=5)
                            if identity_status != 200 or identity != self._python_worker_id:
                                time.sleep(0.5)
                                continue
                        setattr(self, flag, True)
                        return sandbox_id
                    last_error = f"HTTP {status}"
                except SandboxInactiveError:
                    raise
                except (SandboxGatewayError, urllib.error.URLError, TimeoutError) as exc:
                    last_error = str(exc)
                time.sleep(0.5)
            raise SandboxGatewayError(f"{kind} worker on port {port} failed health check: {last_error}; worker_log=/tmp/agent-{kind}-worker.log")

    def run_http(self, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
        if self._recovery_enabled:
            with self._worker_lock:
                try:
                    # HTTP-only requests do not consume Python state. Page writes
                    # retain execution ordering; recovery notices gate all tools.
                    if self._recovery_notice or payload.get("save_page"):
                        self._resolve_pending()
                    sid = self._start_worker("http", 18081)
                    request = dict(payload, timeout=timeout, recovery_backup=bool(payload.get("save_page")))
                    data = self.client.proxy_json(sid, 18081, "/request", request, timeout=timeout + 10)
                    files = data.pop("recovery_files", {})
                    if payload.get("save_page") and data.get("page", {}).get("path"):
                        try:
                            if not files:
                                raise ValueError("Worker did not provide page backup")
                            for path, content in files.items():
                                self._get_journal().file(path, content.encode("utf-8"))
                        except Exception:
                            self._get_journal().record({"kind": "barrier", "reason": "page backup failed"})
                            raise
                    return data
                except SandboxInactiveError:
                    self._restore()
        sandbox_id = self._start_worker("http", 18081)
        assert self.client is not None
        payload = dict(payload, timeout=timeout)
        return self.client.proxy_json(sandbox_id, 18081, "/request", payload, timeout=timeout + 10)

    def run_python(self, code: str, timeout: int = 180) -> str:
        if self._recovery_enabled:
            with self._worker_lock:
                try:
                    data = self._execute_recorded({"code": code}, timeout)
                except SandboxInactiveError:
                    self._restore()
        else:
            sandbox_id = self._start_worker("python", 18080)
            assert self.client is not None
            data = self.client.proxy_json(sandbox_id, 18080, "/execute", {"code": code}, timeout=timeout)
        output = str(data.get("stdout") or "") + str(data.get("stderr") or "")
        if data.get("error"):
            output += str(data["error"])
        return output

    def _resolve_pending(self):
        if self._recovery_notice:
            # Tool wrappers may retry internally. The agent must see the reset
            # notice before any wrapper can dispatch more stateful work.
            raise SandboxStateError("Sandbox status changed; pending recovery notice. Current request was not dispatched")
        if self._pending_python is None:
            return
        operation_id = self._pending_python
        if self._python_restart_needed:
            self._restart_python()
        status, body = self.client.proxy_http(self.sandbox_id, 18080, "/operations/" + operation_id, timeout=10)
        result = json.loads(body) if status == 200 else {}
        if result.get("state") != "completed":
            raise SandboxStateError("Previous Python call is running or its outcome is unknown; no new operation was sent")
        self._get_journal().record({"kind": "python_result", "operation_id": operation_id,
                                   "sha256": result_digest(result["result"])})
        self._pending_python = None
        self._recovery_notice.append("[Sandbox execution] The previously timed-out Python call completed. Saved output:\n" +
                                     str(result["result"].get("stdout", ""))[:3000] + str(result["result"].get("error", ""))[:1000])
        raise SandboxStateError("Previous Python result recovered; current request was not executed.")

    def _restart_python(self):
        self._python_restart_needed = True
        worker = os.getenv("SCIENCE_AGENT_SANDBOX_PYTHON_WORKER", "/tmp/sandbox_python_worker.py")
        identity = self._python_worker_id
        try:
            if not identity:
                raise SandboxGatewayError("Python worker identity unavailable")
            command = "python3 " + self.client._shell_quote(worker) + " --stop " + self.client._shell_quote(identity)
            raw = self.client.command(self.sandbox_id, command, timeout=15)
            confirmation = json.loads(self.client.command_output(raw))
            if confirmation.get("stopped") != identity:
                raise SandboxGatewayError("Python worker termination unconfirmed")
            self._python_started = self._vision_ready = False
            self._start_worker("python", 18080)
            self._get_journal().record({"kind": "python_session_reset"})
        except SandboxInactiveError:
            raise
        except Exception as exc:
            raise SandboxStateError("Python interruption did not complete; session restart unconfirmed. "
                                    "Execution and memory state are unknown. error_type=" + type(exc).__name__) from exc
        self._pending_python = None
        self._python_restart_needed = False
        notice = ("[Sandbox execution] Python session restarted after interruption did not complete. "
                  "The previous worker was terminated. Memory state was cleared; sandbox files were retained "
                  "and may contain partial writes. Historical code was not replayed. External operation outcomes are unverified.")
        self._recovery_notice.append(notice)
        raise SandboxStateError(notice)

    def _interrupt_python(self):
        operation_id = self._pending_python
        try:
            status, body = self.client.proxy_http(self.sandbox_id, 18080, "/operations/" + operation_id, timeout=5)
            data = json.loads(body) if status == 200 else {}
            if data.get("state") == "completed":
                return data
            self.client.proxy_json(self.sandbox_id, 18080, "/interrupt/" + operation_id, {}, timeout=5)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status, body = self.client.proxy_http(self.sandbox_id, 18080, "/operations/" + operation_id, timeout=2)
                data = json.loads(body) if status == 200 else {}
                if data.get("state") == "completed":
                    return data
                time.sleep(0.1)
        except SandboxInactiveError:
            raise
        except (SandboxGatewayError, urllib.error.URLError, TimeoutError, ValueError):
            pass
        self._restart_python()

    def _execute_recorded(self, payload, timeout):
        self._resolve_pending()
        sid = self._start_worker("python", 18080)
        operation_id = uuid.uuid4().hex
        payload = dict(payload, operation_id=operation_id)
        journal = self._get_journal()
        journal.record({"kind": "python_start", "operation_id": operation_id, "payload": payload, "timeout": timeout})
        self._pending_python = operation_id
        try:
            data = self.client.proxy_json(sid, 18080, "/execute", payload, timeout=timeout)
        except SandboxInactiveError:
            # Endpoint lookup failed before dispatch: this call did not run.
            journal.record({"kind": "python_cancelled", "operation_id": operation_id})
            self._pending_python = None
            raise
        except (SandboxGatewayError, urllib.error.URLError, TimeoutError):
            data = self._interrupt_python()
        if data.get("state") != "completed":
            data = self._interrupt_python()
        result = data["result"]
        if result.get("interrupted"):
            journal.record({"kind": "python_interrupted", "operation_id": operation_id})
            self._recovery_notice.append(
                "[Sandbox execution] Python execution was interrupted after response timeout. "
                "The session was retained. Variables and files may contain changes made before interruption. "
                "External operation outcomes are unverified.")
        journal.record({"kind": "python_result", "operation_id": operation_id, "sha256": result_digest(result)})
        self._pending_python = None
        return result

    def run_visual(self, code: str, files: dict[str, str], timeout: int = 180) -> dict[str, Any]:
        if self._recovery_enabled:
            with self._worker_lock:
                try:
                    self._resolve_pending()
                    return self._run_visual(code, files, timeout)
                except SandboxInactiveError:
                    self._restore()
        return self._run_visual(code, files, timeout)

    def _run_visual(self, code, files, timeout):
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
            if self._recovery_enabled:
                try:
                    self._get_journal().file(normalized, content)
                except Exception:
                    self._get_journal().record({"kind": "barrier", "reason": "input backup failed"})
                    raise
            self._uploaded_inputs[normalized] = digest
        if self._recovery_enabled:
            return self._execute_recorded({"code": code, "visual": True}, timeout)
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


def take_recovery_notice():
    run = _SHARED_RUN.get()
    return run.take_recovery_notice() if run is not None else ""


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
    if inspect.iscoroutinefunction(function):
        @wraps(function)
        async def async_wrapped(*args, **kwargs):
            run = AgentSandboxRun()
            token = _SHARED_RUN.set(run)
            try:
                return await function(*args, **kwargs)
            finally:
                try:
                    await asyncio.to_thread(run.close)
                except Exception:
                    logging.getLogger(__name__).exception("Sandbox cleanup failed id=%s", run.sandbox_id)
                finally:
                    _SHARED_RUN.reset(token)
        return async_wrapped

    @wraps(function)
    def wrapped(*args, **kwargs):
        enabled = any(os.getenv(name, "").strip().lower() == "gateway_sandbox" for name in (
            "PYTHON_SESSION_BACKEND", "SCIENCE_AGENT_TOOL_HTTP_BACKEND",
        ))
        # The outer task may own cleanup until post-generation judging ends.
        if not enabled or _SHARED_RUN.get() is not None:
            return function(*args, **kwargs)
        run = AgentSandboxRun()
        token = _SHARED_RUN.set(run)
        try:
            return function(*args, **kwargs)
        finally:
            try:
                run.close()
            except Exception:
                logging.getLogger(__name__).exception("Sandbox cleanup failed id=%s", run.sandbox_id)
            finally:
                _SHARED_RUN.reset(token)
    return wrapped
