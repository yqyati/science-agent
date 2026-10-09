from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests
from sandbox_gateway import SandboxGatewayError, shared_sandbox_run, close_shared_sandbox_run

from runtime_env import get_env_bool, get_env_float, get_env_value

try:
    from science_agent.tracing import emit_trace
except Exception:  # pragma: no cover
    def emit_trace(event: str, *, layer: str, **fields):
        return None


NETWORK_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.ProxyError,
    requests.exceptions.SSLError,
    requests.exceptions.Timeout,
)
_BACKEND_COOLDOWN_LOCK = threading.Lock()
_REQUESTS_COOLDOWN_UNTIL: dict[str, float] = {}


@dataclass
class ToolHttpResponse:
    status_code: int
    text: str
    backend: str
    duration_s: float
    headers: dict[str, str] | None = None

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if 400 <= int(self.status_code):
            raise requests.HTTPError(
                f"{self.status_code} Error",
                response=_ResponseShim(self.status_code, self.text),
            )


class _ResponseShim:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text


def _split_env_pool(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _env_name(prefix: str, suffix: str) -> str:
    return f"{prefix.upper()}_{suffix}"


def _setting(prefix: str, suffix: str, default: Any = None) -> Any:
    return get_env_value(_env_name(prefix, suffix), f"TOOL_HTTP_{suffix}", default=default)


def _setting_float(prefix: str, suffix: str, default: float) -> float:
    value = _setting(prefix, suffix, default=None)
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def _setting_bool(prefix: str, suffix: str, default: bool) -> bool:
    env_name = _env_name(prefix, suffix)
    if os.environ.get(env_name) is not None:
        return get_env_bool(env_name, default=default)
    return get_env_bool(f"TOOL_HTTP_{suffix}", default=default)


def _proxy_url(prefix: str) -> str:
    explicit = os.environ.get(_env_name(prefix, "PROXY_URL"), "").strip()
    if explicit:
        return explicit
    values = _split_env_pool(os.environ.get(_env_name(prefix, "PROXY_URLS")))
    return values[0] if values else ""


def _proxy_host(proxy_url: str) -> str:
    if not proxy_url:
        return ""
    parsed = urlparse(proxy_url)
    if not parsed.netloc:
        return proxy_url
    host = parsed.hostname or parsed.netloc
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return host


def _drop_conflicting_trace_fields(fields: dict[str, Any], *reserved: str) -> dict[str, Any]:
    reserved_keys = {str(key) for key in reserved}
    return {key: value for key, value in fields.items() if str(key) not in reserved_keys}


def tool_http_config(prefix: str) -> dict[str, Any]:
    prefix = prefix.upper()
    backend = str(_setting(prefix, "HTTP_BACKEND", default="auto")).strip().lower() or "auto"
    if backend not in {"requests", "curl", "auto"}:
        backend = "auto"
    legacy_names = [_env_name(prefix, "TIMEOUT")]
    if prefix == "JINA":
        legacy_names.append("VISIT_SERVER_TIMEOUT")
    legacy_timeout = get_env_value(*legacy_names, default=None)
    legacy_timeout_f = float(legacy_timeout) if legacy_timeout not in (None, "") else 180.0
    connect_timeout = _setting_float(prefix, "CONNECT_TIMEOUT_SEC", default=5.0)
    read_timeout = _setting_float(prefix, "READ_TIMEOUT_SEC", default=legacy_timeout_f)
    max_time = _setting_float(prefix, "MAX_TIME_SEC", default=max(read_timeout, connect_timeout))
    requests_failure_cooldown = _setting_float(prefix, "REQUESTS_FAILURE_COOLDOWN_SEC", default=60.0)
    return {
        "prefix": prefix,
        "backend": backend,
        "proxy_url": _proxy_url(prefix),
        "use_env_proxy": _setting_bool(prefix, "USE_ENV_PROXY", default=True),
        "connect_timeout": max(0.1, connect_timeout),
        "read_timeout": max(0.1, read_timeout),
        "max_time": max(0.1, max_time),
        "requests_failure_cooldown": max(0.0, requests_failure_cooldown),
    }


def _apply_config_overrides(
    config: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    if not overrides:
        return config
    merged = dict(config)
    for key, value in overrides.items():
        if value is None or str(value).strip() == "":
            continue
        if key in {"connect_timeout", "read_timeout", "max_time"}:
            merged[key] = max(0.1, float(value))
        elif key == "requests_failure_cooldown":
            merged[key] = max(0.0, float(value))
        elif key == "backend":
            backend = str(value).strip().lower() or "auto"
            merged[key] = backend if backend in {"requests", "curl", "auto"} else "auto"
        elif key == "use_env_proxy":
            merged[key] = bool(value)
        elif key == "proxy_url":
            merged[key] = str(value).strip()
        else:
            merged[key] = value
    return merged


def _requests_cooldown_remaining(prefix: str) -> float:
    now = time.time()
    with _BACKEND_COOLDOWN_LOCK:
        until = _REQUESTS_COOLDOWN_UNTIL.get(prefix.upper(), 0.0)
        if until <= now:
            _REQUESTS_COOLDOWN_UNTIL.pop(prefix.upper(), None)
            return 0.0
        return max(0.0, until - now)


def _mark_requests_unhealthy(prefix: str, cooldown_s: float) -> None:
    if cooldown_s <= 0:
        return
    with _BACKEND_COOLDOWN_LOCK:
        _REQUESTS_COOLDOWN_UNTIL[prefix.upper()] = max(
            _REQUESTS_COOLDOWN_UNTIL.get(prefix.upper(), 0.0),
            time.time() + cooldown_s,
        )


def _requests_call(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None,
    json_payload: Any,
    config: dict[str, Any],
) -> ToolHttpResponse:
    started_at = time.time()
    session = requests.Session()
    session.trust_env = bool(config["use_env_proxy"])
    proxies = None
    if config["proxy_url"]:
        proxies = {"http": config["proxy_url"], "https": config["proxy_url"]}
        session.trust_env = False
    response = session.request(
        method,
        url,
        headers=headers,
        json=json_payload,
        timeout=(float(config["connect_timeout"]), float(config["read_timeout"])),
        proxies=proxies,
    )
    return ToolHttpResponse(
        status_code=response.status_code,
        text=response.text,
        backend="requests",
        duration_s=time.time() - started_at,
        headers=dict(response.headers),
    )


def _curl_call(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None,
    json_payload: Any,
    config: dict[str, Any],
) -> ToolHttpResponse:
    started_at = time.time()
    marker = "\n__SCIENCE_AGENT_HTTP_STATUS__:"
    cmd = [
        "curl",
        "-sS",
        "-L",
        "--compressed",
        "--connect-timeout",
        str(float(config["connect_timeout"])),
        "--max-time",
        str(float(config["max_time"])),
        "-X",
        method.upper(),
    ]
    if config["proxy_url"]:
        cmd.extend(["--proxy", str(config["proxy_url"])])
    elif not config["use_env_proxy"]:
        cmd.extend(["--noproxy", "*"])
    for key, value in (headers or {}).items():
        cmd.extend(["-H", f"{key}: {value}"])
    if json_payload is not None:
        cmd.extend(["--data-binary", json.dumps(json_payload, ensure_ascii=False)])
    cmd.extend(["-w", marker + "%{http_code}", url])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=float(config["max_time"]) + 5.0,
        check=False,
    )
    stdout = proc.stdout or ""
    if marker in stdout:
        body, status = stdout.rsplit(marker, 1)
        status_code = int((status.strip() or "0")[:3] or "0")
    else:
        body = stdout
        status_code = 0
    if proc.returncode != 0:
        error = (proc.stderr or "").strip() or f"curl exited with code {proc.returncode}"
        raise RuntimeError(f"{error}; http_status={status_code}")
    return ToolHttpResponse(
        status_code=status_code,
        text=body,
        backend="curl",
        duration_s=time.time() - started_at,
        headers={},
    )


def tool_http_request(
    *,
    prefix: str,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    json_payload: Any = None,
    trace_layer: str,
    trace_event: str,
    trace_fields: dict[str, Any] | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> ToolHttpResponse:
    if os.getenv("SCIENCE_AGENT_TOOL_HTTP_BACKEND", "").strip().lower() == "gateway_sandbox":
        try:
            return _sandbox_http_request(method=method, url=url, headers=headers, json_payload=json_payload)
        except SandboxGatewayError:
            if os.getenv("SCIENCE_AGENT_SANDBOX_HTTP_STRICT", "true").strip().lower() in {"1", "true", "yes", "on"}:
                raise
    config = _apply_config_overrides(tool_http_config(prefix), config_overrides)
    trace_fields = dict(trace_fields or {})
    backend_order = [config["backend"]]
    if config["backend"] == "auto":
        cooldown_remaining = _requests_cooldown_remaining(config["prefix"])
        backend_order = ["curl", "requests"] if cooldown_remaining > 0 else ["requests", "curl"]
        trace_fields.setdefault("requests_cooldown_remaining_s", round(cooldown_remaining, 3))
    last_exc: Exception | None = None

    for backend in backend_order:
        started_at = time.time()
        start_trace_fields = _drop_conflicting_trace_fields(
            trace_fields,
            "layer",
            "backend",
            "proxy_host",
            "use_env_proxy",
            "connect_timeout_s",
            "read_timeout_s",
            "max_time_s",
        )
        emit_trace(
            f"{trace_event}_http_attempt_start",
            layer=trace_layer,
            backend=backend,
            proxy_host=_proxy_host(str(config["proxy_url"])),
            use_env_proxy=bool(config["use_env_proxy"]),
            connect_timeout_s=config["connect_timeout"],
            read_timeout_s=config["read_timeout"],
            max_time_s=config["max_time"],
            **start_trace_fields,
        )
        try:
            if backend == "curl":
                response = _curl_call(
                    method=method,
                    url=url,
                    headers=headers,
                    json_payload=json_payload,
                    config=config,
                )
            else:
                response = _requests_call(
                    method=method,
                    url=url,
                    headers=headers,
                    json_payload=json_payload,
                    config=config,
                )
            end_trace_fields = _drop_conflicting_trace_fields(
                trace_fields,
                "layer",
                "backend",
                "status_code",
                "response_chars",
                "duration_s",
            )
            emit_trace(
                f"{trace_event}_http_attempt_end",
                layer=trace_layer,
                backend=response.backend,
                status_code=response.status_code,
                response_chars=len(response.text or ""),
                duration_s=round(time.time() - started_at, 3),
                **end_trace_fields,
            )
            return response
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if backend == "requests" and isinstance(exc, NETWORK_EXCEPTIONS):
                _mark_requests_unhealthy(config["prefix"], float(config["requests_failure_cooldown"]))
            error_trace_fields = _drop_conflicting_trace_fields(
                trace_fields,
                "layer",
                "backend",
                "duration_s",
                "error_type",
                "error",
                "fallback_remaining",
                "requests_failure_cooldown_s",
            )
            emit_trace(
                f"{trace_event}_http_attempt_error",
                layer=trace_layer,
                backend=backend,
                duration_s=round(time.time() - started_at, 3),
                error_type=type(exc).__name__,
                error=str(exc),
                fallback_remaining=backend != backend_order[-1],
                requests_failure_cooldown_s=config["requests_failure_cooldown"] if backend == "requests" and isinstance(exc, NETWORK_EXCEPTIONS) else 0,
                **error_trace_fields,
            )
            if config["backend"] != "auto":
                break
            if backend == "requests" and not isinstance(exc, NETWORK_EXCEPTIONS):
                break

    assert last_exc is not None
    raise last_exc


def close_sandbox_http_run() -> None:
    close_shared_sandbox_run()


def _sandbox_http_request(*, method: str, url: str, headers: dict[str, str] | None, json_payload: Any) -> ToolHttpResponse:
    run = shared_sandbox_run()
    payload = {"method": method, "url": url, "headers": headers or {}, "json": json_payload}
    result = run.run_http(payload, timeout=int(os.getenv("TOOL_HTTP_SANDBOX_TIMEOUT", "180")))
    return ToolHttpResponse(
        status_code=int(result.get("status_code", 0)),
        text=str(result.get("text", "")),
        backend="gateway_sandbox",
        duration_s=float(result.get("duration_s", 0)),
        headers=result.get("headers") or {},
    )
