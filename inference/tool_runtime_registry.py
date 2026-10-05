import copy
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import tiktoken

from science_agent.tracing import emit_trace, trace_headers
from tool_advanced_web import AdvancedScholar, AdvancedSearch, AdvancedVisit, DeepSearch
from tool_python import PythonInterpreter
from tool_scholar import Scholar
from tool_search_mcp_r1 import Search
from tool_visit_mcp_r1 import Visit
from tool_visual_python import VisualPythonInterpreter


def _postprocess_remote_search(tool_key: str, content: Any) -> Any:
    """Bound remote search output without changing read_page or calling an LLM."""
    if tool_key not in {"search", "google_scholar"} or not isinstance(content, str):
        return content
    max_results_name = "SCIENCE_AGENT_SCHOLAR_MAX_RESULTS" if tool_key == "google_scholar" else "SCIENCE_AGENT_SEARCH_MAX_RESULTS"
    snippet_name = "SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS" if tool_key == "google_scholar" else "SCIENCE_AGENT_SEARCH_SNIPPET_CHARS"
    try:
        max_results = max(1, int(os.environ.get(max_results_name, "6")))
        snippet_chars = max(100, int(os.environ.get(snippet_name, "700")))
    except (TypeError, ValueError):
        return content

    entry_re = re.compile(r"^\s*\d+\.\s+(?:\*\*)?\[(?P<title>[^\]]+)\]\((?P<link>[^)]+)\)(?:\*\*)?\s*$")
    entries: list[tuple[str, str, list[str]]] = []
    current: list[Any] | None = None
    for line in content.splitlines():
        match = entry_re.match(line)
        if match:
            if current is not None:
                entries.append((current[0], current[1], current[2]))
            current = [match.group("title").strip(), match.group("link").strip(), []]
        elif current is not None:
            current[2].append(line)
    if current is not None:
        entries.append((current[0], current[1], current[2]))
    if not entries:
        return content

    seen_links: set[str] = set()
    selected: list[tuple[str, str, str]] = []
    for title, link, body in entries:
        if link in seen_links:
            continue
        seen_links.add(link)
        clean_body = re.sub(r"\s+", " ", " ".join(part.strip() for part in body if part.strip())).strip()
        if len(clean_body) > snippet_chars:
            clean_body = clean_body[:snippet_chars].rsplit(" ", 1)[0].rstrip() + "..."
        selected.append((title, link, clean_body))
        if len(selected) >= max_results:
            break

    lines = content.splitlines()
    if tool_key == "search":
        header = next((line for line in lines if line.startswith("# Search Results")), "# Search Results")
        answer_lines: list[str] = []
        in_answer = False
        for line in lines:
            if line.strip().startswith("## Answer"):
                in_answer = True
                answer_lines.append(line)
            elif in_answer and line.strip().startswith("## Web Results"):
                break
            elif in_answer:
                answer_lines.append(line)
        result = [header, ""]
        if answer_lines:
            result.extend(answer_lines)
            result.append("")
        result.extend(["## Web Results", ""])
        for index, (title, link, snippet) in enumerate(selected, 1):
            result.append(f"{index}. **[{title}]({link})**")
            if snippet:
                result.append(f"   - {snippet}")
        return "\n".join(result)

    prefix = next((line for line in lines if line.startswith("A Google scholar")), "A Google scholar")
    result = [prefix, "", "## Scholar Results"]
    for index, (title, link, snippet) in enumerate(selected, 1):
        result.append(f"{index}. [{title}]({link})")
        if snippet:
            result.append(snippet)
    return "\n".join(result)


def _postprocess_remote_read_page(content: Any) -> Any:
    """Keep remote read_page results within a token budget, favoring Summary."""
    if not isinstance(content, str):
        return content
    try:
        max_tokens = max(1, int(os.environ.get("SCIENCE_AGENT_READ_PAGE_MAX_TOKENS", "3000")))
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return content

    raw_tokens = encoding.encode(content)
    if len(raw_tokens) <= max_tokens:
        return content

    summary_match = re.search(r"(?im)^\s*Summary:\s*\n?", content)
    if summary_match is None:
        # No structured summary marker: retain the tail, which is the safest
        # fallback for the usual gateway response ordering.
        trimmed = encoding.decode(raw_tokens[-max_tokens:])
    else:
        prefix = content[:summary_match.start()].rstrip()
        summary = content[summary_match.end():].lstrip()
        summary_tokens = encoding.encode(summary)
        if len(summary_tokens) >= max_tokens:
            trimmed = encoding.decode(summary_tokens[:max_tokens])
        else:
            summary_header = "Summary:\n"
            evidence_separator = "\n\n"
            wrapper_tokens = len(encoding.encode(summary_header + evidence_separator))
            remaining = max_tokens - len(summary_tokens) - wrapper_tokens
            prefix_tokens = encoding.encode(prefix)
            # Keep the end of evidence immediately before Summary. This
            # preserves the most recent evidence while guaranteeing Summary.
            evidence_tail = encoding.decode(prefix_tokens[-remaining:]) if remaining > 0 else ""
            trimmed = f"{evidence_tail}{evidence_separator}{summary_header}{summary}" if evidence_tail else f"{summary_header}{summary}"

    emit_trace(
        "remote_read_page_postprocess",
        layer="remote_tool_client",
        raw_chars=len(content),
        raw_tokens=len(raw_tokens),
        final_chars=len(trimmed),
        final_tokens=len(encoding.encode(trimmed)),
        max_tokens=max_tokens,
        summary_preferred=True,
    )
    return trimmed


DEFAULT_TOOL_KEYS = ["search", "visit"]


def _search_factory():
    return Search()


def _visit_factory():
    return Visit()


def _python_factory():
    return PythonInterpreter()


def _vision_factory():
    return VisualPythonInterpreter()


def _scholar_factory():
    return Scholar()


def _advanced_search_factory():
    return AdvancedSearch()


def _advanced_visit_factory():
    return AdvancedVisit()


def _advanced_scholar_factory():
    return AdvancedScholar()


def _deepsearch_factory():
    return DeepSearch()


TOOL_BUILDERS: dict[str, Callable[[], Any]] = {
    "search": _search_factory,
    "visit": _visit_factory,
    "python": _python_factory,
    "vision": _vision_factory,
    "google_scholar": _scholar_factory,
    "advanced_search": _advanced_search_factory,
    "advanced_visit": _advanced_visit_factory,
    "advanced_scholar": _advanced_scholar_factory,
    "deepsearch": _deepsearch_factory,
}

TOOL_SCHEMA_NAMES = {
    "search": "google_search-google_search",
    "visit": "read-page-server-read_page",
    "python": "PythonInterpreter",
    "vision": "VisualPythonInterpreter",
    "google_scholar": "google_scholar",
    "advanced_search": "advanced_search",
    "advanced_visit": "advanced_visit",
    "advanced_scholar": "advanced_scholar",
    "deepsearch": "deepsearch",
}


TOOL_ALIASES = {
    "search": "search",
    "google_search": "search",
    "google_search-google_search": "search",
    "visit": "visit",
    "read-page-server-read_page": "visit",
    "python": "python",
    "code_executor": "python",
    "code_interpreter": "python",
    "pythoninterpreter": "python",
    "vision": "vision",
    "visualpythoninterpreter": "vision",
    "google_scholar": "google_scholar",
    "advanced_search": "advanced_search",
    "advanced-search": "advanced_search",
    "advanced-web-search": "advanced_search",
    "advanced_web_search": "advanced_search",
    "advanced_visit": "advanced_visit",
    "advanced-visit": "advanced_visit",
    "advanced-read-page": "advanced_visit",
    "advanced_read_page": "advanced_visit",
    "advanced_scholar": "advanced_scholar",
    "advanced-scholar": "advanced_scholar",
    "deepsearch": "deepsearch",
    "deep_search": "deepsearch",
    "deep-search": "deepsearch",
}


GEMINI_COMPAT_BASE_URLS = {
    "http://35.220.164.252:3888/v1",
}


def schema_tool_name(tool_name: str) -> str:
    return TOOL_SCHEMA_NAMES[canonicalize_tool_name(tool_name)]


def _normalized_base_url(base_url: str | None) -> str:
    return str(base_url or "").strip().rstrip("/").lower()


def _schema_for_gemini_compat(value: Any) -> Any:
    if isinstance(value, list):
        return [_schema_for_gemini_compat(item) for item in value]
    if not isinstance(value, dict):
        return value

    result = {key: _schema_for_gemini_compat(item) for key, item in value.items()}
    raw_type = result.get("type")
    if isinstance(raw_type, list) and "array" in raw_type and "items" in result:
        result["type"] = "array"
    return result


def compat_tool_schemas_for_base_url(tool_schemas: Any, base_url: str | None) -> Any:
    if _normalized_base_url(base_url) not in GEMINI_COMPAT_BASE_URLS:
        return tool_schemas
    return _schema_for_gemini_compat(copy.deepcopy(tool_schemas))


def _remote_tool_guards_from_env() -> dict[str, Any]:
    guards: dict[str, Any] = {}
    if os.getenv("AGENT_EVAL_BENCHMARK") is not None:
        guards["benchmark"] = os.getenv("AGENT_EVAL_BENCHMARK", "")
    raw_guard = os.getenv("AGENT_EVAL_URL_GUARD")
    if raw_guard is not None and raw_guard.strip() != "":
        guards["url_guard"] = raw_guard.strip().lower() in {"1", "true", "yes", "y", "on"}
    if os.getenv("AGENT_EVAL_URL_GUARD_BENCHMARKS") is not None:
        guards["url_guard_benchmarks"] = os.getenv("AGENT_EVAL_URL_GUARD_BENCHMARKS", "")
    if os.getenv("AGENT_EVAL_BLOCKED_URLS_FILE") is not None:
        guards["blocked_urls_file"] = os.getenv("AGENT_EVAL_BLOCKED_URLS_FILE", "")
    return guards


def _cache_control_bool_from_env(name: str) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return False
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value (0/1, true/false, yes/no, on/off)")


def _remote_tool_cache_control_from_env() -> dict[str, bool]:
    cache_control = {
        "read": _cache_control_bool_from_env("SCIENCE_AGENT_GATEWAY_CACHE_READ"),
        "write": _cache_control_bool_from_env("SCIENCE_AGENT_GATEWAY_CACHE_WRITE"),
        "refresh": _cache_control_bool_from_env("SCIENCE_AGENT_GATEWAY_CACHE_REFRESH"),
    }
    if cache_control["refresh"] and not cache_control["write"]:
        raise ValueError(
            "SCIENCE_AGENT_GATEWAY_CACHE_REFRESH=1 requires "
            "SCIENCE_AGENT_GATEWAY_CACHE_WRITE=1"
        )
    return cache_control


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _remote_tool_provider_strategy_from_env() -> dict[str, str]:
    # Only forward enabled switches. Default 0 values in an eval script should
    # not override the gateway machine's deployment-level strategy.
    provider_env: dict[str, str] = {}
    for name in (
        "TOOL_USE_ADVANCED_BACKEND",
        "TOOL_PROVIDER_JINA",
        "TOOL_PROVIDER_EXA",
        "TOOL_PROVIDER_TAVILY",
        "TOOL_PROVIDER_SHUYAN",
    ):
        if _env_truthy(name):
            provider_env[name] = "1"
    return provider_env


def _remote_tool_summary_kwargs_from_env() -> dict[str, str]:
    slot = str(os.getenv("SUMMARY_MODEL_SLOT") or os.getenv("SCIENCE_AGENT_SUMMARY_MODEL_SLOT") or "").strip()
    return {"summary_model_slot": slot} if slot else {}


def _env_int(name: str) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None or str(raw_value).strip() == "":
        return None
    try:
        return int(float(str(raw_value).strip()))
    except ValueError:
        return None


def _remote_gateway_timeout() -> int:
    explicit = _env_int("SCIENCE_AGENT_GATEWAY_TIMEOUT")
    if explicit is not None and explicit > 0:
        return explicit

    candidates = [
        _env_int("PYTHON_TOOL_TIMEOUT_SEC"),
        _env_int("VISIT_SERVER_TIMEOUT"),
        _env_int("SERPER_TIMEOUT"),
        _env_int("SUMMARY_REQUEST_TIMEOUT_SEC"),
        _env_int("LLM_REQUEST_TIMEOUT_SEC"),
    ]
    base_timeout = max([value for value in candidates if value and value > 0] or [300])
    buffer_sec = _env_int("SCIENCE_AGENT_GATEWAY_TIMEOUT_BUFFER_SEC")
    if buffer_sec is None:
        buffer_sec = 300
    return max(1, base_timeout + max(0, buffer_sec))


class RemoteToolProxy:
    def __init__(self, tool_key: str):
        self.tool_key = tool_key
        local_tool = TOOL_BUILDERS[tool_key]()
        self.name = TOOL_SCHEMA_NAMES[tool_key]
        self.description = str(getattr(local_tool, "description", ""))
        self.parameters = copy.deepcopy(getattr(local_tool, "parameters", {}))
        self._session_id: str | None = None
        self._guards: dict[str, Any] | None = None
        self.is_remote_tool_proxy = True

    def reset_jupyter_history(self) -> None:
        self.close_session()

    def set_guards(self, guards: dict[str, Any] | None) -> None:
        self._guards = dict(guards or {}) or None

    def bind_session(self, session_id: str) -> None:
        self._session_id = str(session_id or "")

    @property
    def session_id(self) -> str:
        return self._session_id or ""

    def close_session(self) -> None:
        session_id = self._session_id
        self._session_id = None
        if not session_id:
            return
        try:
            self._request_json("DELETE", f"/v1/sessions/{session_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"[RemoteToolProxy] Warning: failed to close gateway session {session_id}: {exc}")

    def set_text_response_fallback(self, enabled: bool) -> None:
        del enabled

    def set_task_context(
        self,
        item: dict[str, Any],
        *,
        base_dir: Optional[str] = None,
        artifact_dir: Optional[str] = None,
    ) -> str:
        data = self.create_shared_session(
            item,
            tools=[self.tool_key],
            base_dir=base_dir,
            artifact_dir=artifact_dir,
        )
        hints = data.get("hints") or {}
        if isinstance(hints, dict):
            return str(hints.get(self.name) or hints.get(self.tool_key) or "")
        return ""

    def create_shared_session(
        self,
        item: dict[str, Any],
        *,
        tools: Iterable[str],
        base_dir: Optional[str] = None,
        artifact_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        payload = {
            "sample": item,
            "tools": list(tools),
            "run_id": os.getenv("SCIENCE_AGENT_RUN_ID", ""),
            "tools_kwargs": {
                "dataset_base_dir": base_dir,
                "artifact_dir": artifact_dir,
            },
        }
        payload["tools_kwargs"].update(_remote_tool_summary_kwargs_from_env())
        provider_strategy = _remote_tool_provider_strategy_from_env()
        if provider_strategy:
            payload["tools_kwargs"]["provider_strategy"] = provider_strategy
        if self._guards:
            payload["tools_kwargs"]["guards"] = self._guards
        data = self._post("/v1/sessions", payload)
        self._session_id = str(data.get("session_id") or "")
        return data

    def append_vision_inputs(self, images: list[dict[str, Any]]) -> dict[str, Any]:
        if not self._session_id:
            raise RuntimeError("gateway session is required before appending vision inputs")
        payload = {
            "tool": "VisualPythonInterpreter",
            "images": images,
            "run_id": os.getenv("SCIENCE_AGENT_RUN_ID", ""),
        }
        return self._post(f"/v1/sessions/{self._session_id}/vision_inputs", payload)

    def call(self, params, files=None, timeout=None, **kwargs):
        del files, kwargs
        payload = {
            "tool": self.name,
            "arguments": params if isinstance(params, dict) else {"code": params},
            "run_id": os.getenv("SCIENCE_AGENT_RUN_ID", ""),
            "cache_control": _remote_tool_cache_control_from_env(),
        }
        guards = self._guards if self._guards is not None else _remote_tool_guards_from_env()
        if guards:
            payload["guards"] = guards
        provider_strategy = _remote_tool_provider_strategy_from_env()
        tools_kwargs = _remote_tool_summary_kwargs_from_env()
        if provider_strategy:
            tools_kwargs["provider_strategy"] = provider_strategy
        if tools_kwargs:
            payload["tools_kwargs"] = tools_kwargs
        if timeout is not None:
            payload["timeout"] = timeout
        if self._session_id:
            data = self._post(f"/v1/sessions/{self._session_id}/tool_call", payload)
        else:
            data = self._post("/v1/tool_call", payload)
        content = data.get("content", data.get("result", ""))
        # The remote gateway does not receive local output caps. Apply them
        # client-side so remote search and read_page share bounded results.
        if self.tool_key == "visit":
            return _postprocess_remote_read_page(content)
        return _postprocess_remote_search(self.tool_key, content)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request_json("POST", path, payload)

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        base_url = (os.getenv("SCIENCE_AGENT_GATEWAY_URL") or os.getenv("TOOL_GATEWAY_URL") or "").rstrip("/")
        if not base_url:
            raise RuntimeError("SCIENCE_AGENT_GATEWAY_URL or TOOL_GATEWAY_URL is required for remote tools.")
        max_attempts = 1
        if method == "POST" and path == "/v1/sessions":
            max_attempts = max(1, _env_int("SCIENCE_AGENT_GATEWAY_SESSION_CREATE_RETRIES") or 3)
        last_exc: Exception | None = None
        data: dict[str, Any] | None = None
        timeout = _remote_gateway_timeout()
        tool_name = ""
        if isinstance(payload, dict):
            tool_name = str(payload.get("tool") or payload.get("name") or "")
        session_id = ""
        path_parts = [part for part in path.split("/") if part]
        if len(path_parts) >= 3 and path_parts[0] == "v1" and path_parts[1] == "sessions":
            session_id = path_parts[2]
        for attempt in range(1, max_attempts + 1):
            started_at = time.time()
            emit_trace(
                "gateway_request_start",
                layer="remote_tool_client",
                method=method,
                path=path,
                tool=tool_name,
                session_id=session_id,
                attempt=attempt,
                max_attempts=max_attempts,
                timeout_s=timeout,
                base_url=base_url,
            )
            headers = {"Content-Type": "application/json"}
            headers.update(
                trace_headers(
                    {
                        "gateway_path": path,
                        "tool": tool_name,
                        "session_id": session_id,
                    }
                )
            )
            request = Request(
                base_url + path,
                data=json.dumps(payload or {}).encode("utf-8") if method != "GET" else None,
                headers=headers,
                method=method,
            )
            try:
                with urlopen(request, timeout=timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                emit_trace(
                    "gateway_request_end",
                    layer="remote_tool_client",
                    method=method,
                    path=path,
                    tool=tool_name,
                    session_id=session_id,
                    attempt=attempt,
                    duration_s=round(time.time() - started_at, 3),
                    ok=bool(data.get("ok")) if isinstance(data, dict) else None,
                    response_tool=str(data.get("tool") or "") if isinstance(data, dict) else "",
                    route=data.get("route") if isinstance(data, dict) else None,
                )
                break
            except HTTPError as exc:
                try:
                    body = exc.read().decode("utf-8")
                    data = json.loads(body)
                except Exception:
                    emit_trace(
                        "gateway_request_error",
                        layer="remote_tool_client",
                        method=method,
                        path=path,
                        tool=tool_name,
                        session_id=session_id,
                        attempt=attempt,
                        duration_s=round(time.time() - started_at, 3),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise
                emit_trace(
                    "gateway_request_end",
                    layer="remote_tool_client",
                    method=method,
                    path=path,
                    tool=tool_name,
                    session_id=session_id,
                    attempt=attempt,
                    duration_s=round(time.time() - started_at, 3),
                    http_status=exc.code,
                    ok=bool(data.get("ok")) if isinstance(data, dict) else None,
                    error=str(data.get("error") or "") if isinstance(data, dict) else "",
                )
                break
            except (URLError, OSError, TimeoutError, ConnectionError) as exc:
                last_exc = exc
                emit_trace(
                    "gateway_request_error",
                    layer="remote_tool_client",
                    method=method,
                    path=path,
                    tool=tool_name,
                    session_id=session_id,
                    attempt=attempt,
                    duration_s=round(time.time() - started_at, 3),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                if attempt >= max_attempts:
                    raise
                sleep_sec = min(5.0, 0.2 * attempt)
                time.sleep(sleep_sec)
        if data is None:
            raise RuntimeError(f"remote gateway request failed: {last_exc}")
        if not data.get("ok", False):
            raise RuntimeError(str(data.get("error") or "remote tool call failed"))
        return data


def _build_tool(tool_key: str) -> Any:
    if str(os.getenv("AGENT_TOOL_BACKEND") or "").strip().lower() == "remote":
        return RemoteToolProxy(tool_key)
    return TOOL_BUILDERS[tool_key]()


TOOL_POLICY = {
    "search": "Use search when external facts are uncertain or you need to find candidate sources and URLs.",
    "visit": "Use read-page after you already have a relevant URL or when the question directly references a webpage.",
    "python": "Use PythonInterpreter when a Jupyter notebook cell materially helps scientific reasoning, exact computation, data analysis, plotting, or intermediate checking.",
    "vision": "Use VisualPythonInterpreter when image inspection, cropping, plotting, OCR-like measurement, or visual computation materially helps solve a multimodal problem.",
    "google_scholar": "Use Google Scholar when academic papers or scholarly evidence are more appropriate than general web results.",
    "advanced_search": "Use advanced_search only when normal search is insufficient and you need concise, high-relevance evidence or candidate URLs.",
    "advanced_visit": "Use advanced_visit after you have URL(s) and need targeted extraction from webpages, PDFs, dynamic pages, or structured metadata.",
    "advanced_scholar": "Use advanced_scholar when academic papers, citations, PDFs, or research metadata are central.",
    "deepsearch": "Use deepsearch only for hard research questions after normal search, visit, and scholar tools are insufficient.",
}

@dataclass
class ToolRuntimeSelection:
    requested_tool_keys: list[str]
    active_tool_keys: list[str]
    active_tool_names: list[str]
    active_tools: list[Any]
    active_tool_map: dict[str, Any]
    active_tool_schemas: list[dict[str, Any]]
    prompt_append: str


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def split_requested_tools(raw_tools: Optional[str]) -> list[str]:
    if raw_tools is None:
        return []
    return [part.strip() for part in str(raw_tools).split(",") if part.strip()]


def canonicalize_tool_name(tool_name: str) -> str:
    normalized = str(tool_name).strip()
    if not normalized:
        raise ValueError("Tool name cannot be empty.")
    canonical = TOOL_ALIASES.get(normalized.lower())
    if canonical is None:
        valid_names = ", ".join(sorted(TOOL_BUILDERS))
        raise ValueError(
            f"Unknown tool '{tool_name}'. Supported tool keys: {valid_names}"
        )
    return canonical


def _coerce_requested_keys(
    function_list: Optional[Iterable[Any]],
    default_tools: Optional[Iterable[str]] = None,
) -> list[str]:
    requested_items = list(function_list or [])
    if not requested_items:
        requested_items = list(default_tools or DEFAULT_TOOL_KEYS)

    resolved: list[str] = []
    for item in requested_items:
        if isinstance(item, str):
            resolved.append(canonicalize_tool_name(item))
            continue

        tool_name = getattr(item, "name", None)
        if isinstance(tool_name, str) and tool_name.strip():
            resolved.append(canonicalize_tool_name(tool_name))
            continue

        raise ValueError(f"Unsupported tool declaration: {item!r}")

    return _dedupe_keep_order(resolved)


def build_openai_tool_schema(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": str(getattr(tool, "name", "")),
            "description": str(getattr(tool, "description", "")).strip(),
            "parameters": copy.deepcopy(getattr(tool, "parameters", {})),
        },
    }


def _compress_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def _format_required_args(tool: Any) -> str:
    parameters = getattr(tool, "parameters", {}) or {}
    required = parameters.get("required", [])
    if isinstance(required, list) and required:
        return ", ".join(str(item) for item in required)
    return "none"


def build_dynamic_tool_prompt(
    active_tool_keys: list[str],
    active_tools: list[Any],
) -> str:
    lines = [
        "# Available tools",
        "Only the following tools are available in this run. Call only these tools.",
    ]
    for tool in active_tools:
        description = _compress_whitespace(getattr(tool, "description", ""))
        required_args = _format_required_args(tool)
        lines.append(
            f"- `{tool.name}`: {description} Required arguments: {required_args}."
        )

    lines.append("")
    lines.append("Tool policy:")
    for tool_key in active_tool_keys:
        policy = TOOL_POLICY.get(tool_key)
        if policy:
            lines.append(f"- {policy}")

    return "\n".join(lines).strip()


def build_agent_eval_tool_guidance(
    function_list: Optional[Iterable[Any]] = None,
    *,
    default_tools: Optional[Iterable[str]] = None,
) -> str:
    runtime = build_tool_runtime(
        function_list,
        default_tools=default_tools,
    )
    lines = [
        "Tool Use In This Run:",
        "Use tools only when they materially help solve the problem. If you can solve it directly, do not call tools just to look busy.",
        "",
        runtime.prompt_append,
        "",
        "Final response format for this benchmark:",
        "- Put your reasoning inside <think></think>.",
        "- Put your final answer inside <answer></answer>.",
        "- Inside <answer>, preserve the benchmark-specific answer format requested above.",
        "- Do not end with apologies or tool-limit notes.",
    ]
    return "\n".join(lines).strip()


def build_tool_runtime(
    function_list: Optional[Iterable[Any]],
    *,
    supported_tool_keys: Optional[Iterable[str]] = None,
    default_tools: Optional[Iterable[str]] = None,
) -> ToolRuntimeSelection:
    requested_tool_keys = _coerce_requested_keys(function_list, default_tools)

    supported_keys = (
        set(supported_tool_keys) if supported_tool_keys is not None else set(TOOL_BUILDERS)
    )
    unsupported = [tool_key for tool_key in requested_tool_keys if tool_key not in supported_keys]
    if unsupported:
        unsupported_text = ", ".join(unsupported)
        supported_text = ", ".join(sorted(supported_keys))
        raise ValueError(
            f"Unsupported tools for this agent: {unsupported_text}. "
            f"Supported tool keys: {supported_text}"
        )

    active_tools = [_build_tool(tool_key) for tool_key in requested_tool_keys]
    active_tool_map: dict[str, Any] = {}
    for tool_key, tool in zip(requested_tool_keys, active_tools):
        active_tool_map[getattr(tool, "name", TOOL_SCHEMA_NAMES[tool_key])] = tool
        for alias, canonical in TOOL_ALIASES.items():
            if canonical == tool_key:
                active_tool_map[alias] = tool
    active_tool_schemas = [build_openai_tool_schema(tool) for tool in active_tools]
    active_tool_names = [tool.name for tool in active_tools]
    prompt_append = build_dynamic_tool_prompt(requested_tool_keys, active_tools)

    return ToolRuntimeSelection(
        requested_tool_keys=requested_tool_keys,
        active_tool_keys=requested_tool_keys,
        active_tool_names=active_tool_names,
        active_tools=active_tools,
        active_tool_map=active_tool_map,
        active_tool_schemas=active_tool_schemas,
        prompt_append=prompt_append,
    )


def call_runtime_tool(
    active_tool_map: dict[str, Any],
    tool_name: str,
    tool_args: Any,
    **kwargs,
) -> Union[str, list]:
    lookup_names = [tool_name, str(tool_name).lower()]
    try:
        lookup_names.append(schema_tool_name(tool_name))
    except Exception:
        pass
    tool = None
    for lookup_name in lookup_names:
        if lookup_name in active_tool_map:
            tool = active_tool_map[lookup_name]
            break
    if tool is None:
        return f"Error: Tool {tool_name} not enabled for this run"

    schema_name = str(getattr(tool, "name", tool_name))
    if "python" in schema_name.lower() or "python" in str(tool_name).lower() or str(tool_name).lower() in {"code_executor", "code_interpreter"}:
        code = tool_args
        if isinstance(tool_args, dict):
            code = tool_args.get("code") or tool_args.get("raw") or ""
        raw_result = tool.call(code)
        if isinstance(raw_result, (str, list)):
            return raw_result
        return str(raw_result)

    normalized_args = copy.deepcopy(tool_args)
    if isinstance(normalized_args, dict):
        normalized_args["params"] = copy.deepcopy(normalized_args)

    raw_result = tool.call(normalized_args, **kwargs)
    return raw_result if isinstance(raw_result, (str, list)) else str(raw_result)
