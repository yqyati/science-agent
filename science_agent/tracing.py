from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator


_TRACE_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "science_agent_trace_context",
    default={},
)
_WRITE_LOCK = threading.Lock()
_TRACE_WRITE_WARNED: set[str] = set()


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def trace_file() -> str:
    explicit = str(os.getenv("SCIENCE_AGENT_TRACE_FILE") or "").strip()
    if explicit:
        return explicit
    for env_name in ("RUN_DIR", "SCIENCE_AGENT_RUN_DIR", "SCIENCE_AGENT_GATEWAY_RUN_DIR"):
        run_dir = str(os.getenv(env_name) or "").strip()
        if run_dir:
            return str(Path(run_dir) / "trace.jsonl")
    return str(Path.cwd() / "science_agent_trace.jsonl")


def trace_enabled() -> bool:
    raw_value = os.getenv("SCIENCE_AGENT_TRACE")
    if raw_value is not None:
        return _truthy(raw_value)
    return True


def current_trace_context() -> dict[str, Any]:
    return dict(_TRACE_CONTEXT.get() or {})


def merge_trace_values(*contexts: dict[str, Any] | None, **values: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    def put(key: Any, value: Any) -> None:
        if value is not None and value != "":
            merged[str(key)] = value

    for context in contexts:
        if not context:
            continue
        for key, value in context.items():
            put(key, value)
    for key, value in values.items():
        put(key, value)
    return merged


@contextlib.contextmanager
def trace_context(**values: Any) -> Iterator[None]:
    context = current_trace_context()
    for key, value in values.items():
        if value is not None and value != "":
            context[key] = value
    token = _TRACE_CONTEXT.set(context)
    try:
        yield
    finally:
        _TRACE_CONTEXT.reset(token)


def trace_context_from_headers(headers: Any) -> dict[str, Any]:
    context: dict[str, Any] = {}
    trace_id = headers.get("X-Science-Agent-Trace-Id") if headers is not None else None
    if trace_id:
        context["trace_id"] = str(trace_id)
    raw_context = headers.get("X-Science-Agent-Trace-Context") if headers is not None else None
    if raw_context:
        try:
            decoded = json.loads(str(raw_context))
            if isinstance(decoded, dict):
                for key, value in decoded.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        context[str(key)] = value
        except Exception:
            context["trace_context_parse_error"] = True
    return context


def merge_trace_context(headers: Any = None, **values: Any) -> dict[str, Any]:
    return merge_trace_values(trace_context_from_headers(headers), **values)


def trace_headers(extra: dict[str, Any] | None = None) -> dict[str, str]:
    if not trace_enabled():
        return {}
    context = current_trace_context()
    env_defaults = {
        "run_id": os.getenv("SCIENCE_AGENT_RUN_ID"),
        "run_name": os.getenv("RUN_NAME"),
        "served_model": os.getenv("SERVED_MODEL_NAME") or os.getenv("MODEL_NAME"),
    }
    for key, value in env_defaults.items():
        if value is not None and value != "" and not context.get(key):
            context[key] = value
    if extra:
        for key, value in extra.items():
            if value is not None and value != "":
                context[key] = value
    trace_id = str(context.get("trace_id") or os.getenv("SCIENCE_AGENT_TRACE_ID") or context.get("run_id") or "")
    if not trace_id:
        return {}
    safe_context = {
        str(key): value
        for key, value in context.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    return {
        "X-Science-Agent-Trace-Id": trace_id,
        "X-Science-Agent-Trace-Context": json.dumps(safe_context, ensure_ascii=False, sort_keys=True),
    }


def _max_field_chars() -> int:
    raw_value = os.getenv("SCIENCE_AGENT_TRACE_MAX_FIELD_CHARS", "512")
    try:
        return max(32, int(float(raw_value)))
    except ValueError:
        return 512


def _safe_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > max_chars:
            return value[:max_chars] + f"...<truncated {len(value) - max_chars} chars>"
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_value(val, max_chars) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        limit = 20
        output = [_safe_value(item, max_chars) for item in list(value)[:limit]]
        if len(value) > limit:
            output.append(f"...<truncated {len(value) - limit} items>")
        return output
    return str(value)[:max_chars]


def emit_trace(event: str, *, layer: str, **fields: Any) -> None:
    if not trace_enabled():
        return
    path = trace_file()
    if not path:
        return

    max_chars = _max_field_chars()
    context = current_trace_context()
    record = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "time": time.time(),
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
        "layer": layer,
        "event": event,
    }
    record.update({key: _safe_value(value, max_chars) for key, value in context.items()})
    record.update({key: _safe_value(value, max_chars) for key, value in fields.items()})

    try:
        trace_path = Path(path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with _WRITE_LOCK:
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
    except Exception as exc:  # noqa: BLE001
        with _WRITE_LOCK:
            if path not in _TRACE_WRITE_WARNED:
                _TRACE_WRITE_WARNED.add(path)
                print(
                    f"[science-agent trace] warning: failed to write trace file {path}: {exc}",
                    file=sys.stderr,
                )
