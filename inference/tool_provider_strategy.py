from __future__ import annotations

import json
import os
import random
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import quote

from runtime_env import get_env_bool, get_env_float, get_env_int
from tool_http_client import tool_http_request
from tool_url_guard import (
    GUARD_FORMAT_NORMALIZED_SCHOLAR_V1,
    GUARD_FORMAT_NORMALIZED_SEARCH_V1,
    blocked_search_message,
    blocked_visit_message,
    capture_structured_basis,
    is_search_result_blocked,
    is_text_blocked,
    is_url_blocked,
    partial_search_note,
)

try:
    from science_agent.tracing import emit_trace
except Exception:  # pragma: no cover
    def emit_trace(event: str, *, layer: str, **fields):
        return None


TRUE_VALUES = {"1", "true", "yes", "on", "y"}
PASSTHROUGH_ENV = "_SCIENCE_AGENT_TOOL_PROVIDER_PASSTHROUGH"
SEARCH_STRATEGIES = {
    "original",
    "advanced",
    "jina",
    "exa",
    "tavily",
    "shuyan",
}


@dataclass
class SearchProviderResult:
    title: str = "Untitled"
    link: str = "#"
    snippet: str = ""
    date: str = ""
    source: str = ""
    image: str = ""
    raw: dict[str, Any] | None = None


def _split_pool(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        values = _split_pool(os.environ.get(name))
        if values:
            return values[0]
    return default


def _random_env(*names: str, default: str = "") -> str:
    values: list[str] = []
    for name in names:
        values.extend(_split_pool(os.environ.get(name)))
    return random.choice(values) if values else default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    return get_env_int(name, default=default)


def _env_float(name: str, default: float) -> float:
    return get_env_float(name, default=default)


def provider_strategy() -> str:
    if _env_bool(PASSTHROUGH_ENV, False):
        return "original"
    flags = {
        "advanced": _env_bool("TOOL_USE_ADVANCED_BACKEND", False),
        "jina": _env_bool("TOOL_PROVIDER_JINA", False),
        "exa": _env_bool("TOOL_PROVIDER_EXA", False),
        "tavily": _env_bool("TOOL_PROVIDER_TAVILY", False),
        "shuyan": _env_bool("TOOL_PROVIDER_SHUYAN", False),
    }
    enabled = [name for name, is_enabled in flags.items() if is_enabled]
    if len(enabled) > 1:
        raise RuntimeError(
            "Tool provider switches are mutually exclusive; enabled="
            + ",".join(enabled)
        )
    return enabled[0] if enabled else "original"


@contextmanager
def provider_passthrough():
    previous = os.environ.get(PASSTHROUGH_ENV)
    os.environ[PASSTHROUGH_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PASSTHROUGH_ENV, None)
        else:
            os.environ[PASSTHROUGH_ENV] = previous


def _env_int_first(default: int, *names: str) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip() != "":
            return int(value)
    return int(default)


def _env_float_first(default: float, *names: str) -> float:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip() != "":
            return float(value)
    return float(default)


def _provider_retry_settings(tool: str, *, strategy: str = "") -> tuple[int, float]:
    if strategy == "shuyan":
        if tool in {"search", "google_scholar"}:
            return (
                max(1, _env_int_first(2, "SHUYAN_SEARCH_MAX_RETRIES", "SHUYAN_MAX_RETRIES")),
                max(0.0, _env_float_first(1.0, "SHUYAN_SEARCH_RETRY_SLEEP_SEC", "SHUYAN_RETRY_SLEEP_SEC")),
            )
        if tool == "visit":
            return (
                max(1, _env_int_first(2, "SHUYAN_READER_MAX_RETRIES", "SHUYAN_MAX_RETRIES")),
                max(0.0, _env_float_first(1.0, "SHUYAN_READER_RETRY_SLEEP_SEC", "SHUYAN_RETRY_SLEEP_SEC")),
            )
    if tool in {"search", "google_scholar"}:
        return (
            max(1, _env_int("SERPER_MAX_RETRIES", 10)),
            max(0.0, _env_float("SERPER_RETRY_SLEEP_SEC", 1.0)),
        )
    if tool == "visit":
        return (
            max(1, _env_int("VISIT_READ_MAX_RETRIES", 5)),
            max(0.0, _env_float("VISIT_RETRY_SLEEP_SEC", 2.0)),
        )
    return 1, 0.0


def _provider_error_retryable(exc: Exception, *, strategy: str = "") -> bool:
    # Configuration errors will not recover by retrying and should fail fast.
    text = str(exc)
    if "provider is enabled but no API key is configured" in text:
        return False
    if strategy == "shuyan":
        lowered = text.lower()
        non_retryable_markers = (
            "400 error",
            "401 error",
            "403 error",
            "404 error",
            "422 error",
            "bad request",
            "unauthorized",
            "forbidden",
            "not found",
            "invalid api key",
            "insufficient balance",
            "quota",
        )
        if any(marker in lowered for marker in non_retryable_markers):
            return False
        if re.search(r"(?:code|http_status|status)[=:]\s*(?:400|401|403|404|422)\b", lowered):
            return False
    return True


def _provider_retry_sleep_seconds(
    *,
    tool: str,
    strategy: str,
    base_sleep_s: float,
    attempt: int,
    exc: Exception,
) -> float:
    if strategy != "shuyan":
        return base_sleep_s
    kind = "SEARCH" if tool in {"search", "google_scholar"} else "READER"
    text = str(exc).lower()
    is_rate_limited = (
        "429 error" in text
        or "http_429" in text
        or "too many requests" in text
        or re.search(r"(?:http_status|status)[=:]\s*429\b", text) is not None
    )
    sleep_s = base_sleep_s
    if is_rate_limited:
        sleep_s = _env_float_first(
            max(base_sleep_s, 3.0),
            f"SHUYAN_{kind}_429_RETRY_SLEEP_SEC",
            "SHUYAN_429_RETRY_SLEEP_SEC",
            f"SHUYAN_{kind}_RETRY_SLEEP_SEC",
            "SHUYAN_RETRY_SLEEP_SEC",
        )
    multiplier = _env_float_first(
        1.5,
        f"SHUYAN_{kind}_RETRY_BACKOFF_MULTIPLIER",
        "SHUYAN_RETRY_BACKOFF_MULTIPLIER",
    )
    max_sleep_s = _env_float_first(
        15.0,
        f"SHUYAN_{kind}_RETRY_MAX_SLEEP_SEC",
        "SHUYAN_RETRY_MAX_SLEEP_SEC",
    )
    sleep_s = max(0.0, sleep_s) * max(1.0, multiplier) ** max(0, attempt - 1)
    return min(max(0.0, sleep_s), max(0.0, max_sleep_s))


def _run_provider_with_retries(
    *,
    tool: str,
    strategy: str,
    fn: Callable[[], Any],
    query: str = "",
    url: str = "",
) -> Any:
    max_attempts, retry_sleep_s = _provider_retry_settings(tool, strategy=strategy)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        attempt_started_at = time.time()
        trace_fields = {
            "tool": tool,
            "strategy": strategy,
            "attempt": attempt,
            "max_attempts": max_attempts,
        }
        if query:
            trace_fields["query"] = query
        if url:
            trace_fields["url"] = url
        emit_trace("tool_provider_attempt_start", layer="tool_provider", **trace_fields)
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            retryable = _provider_error_retryable(exc, strategy=strategy)
            will_retry = retryable and attempt < max_attempts
            emit_trace(
                "tool_provider_attempt_error",
                layer="tool_provider",
                **trace_fields,
                error_type=type(exc).__name__,
                error=str(exc),
                retryable=retryable,
                will_retry=will_retry,
                duration_s=round(time.time() - attempt_started_at, 3),
            )
            if not will_retry:
                break
            sleep_s = _provider_retry_sleep_seconds(
                tool=tool,
                strategy=strategy,
                base_sleep_s=retry_sleep_s,
                attempt=attempt,
                exc=exc,
            )
            if sleep_s > 0:
                time.sleep(sleep_s)
            continue
        emit_trace(
            "tool_provider_attempt_end",
            layer="tool_provider",
            **trace_fields,
            status="ok",
            duration_s=round(time.time() - attempt_started_at, 3),
        )
        return result
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{tool} provider retry loop ended without a result")


def _clean_one_line(value: Any, *, max_chars: int = 1200) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _trim_text(value: Any, *, max_chars: int = 4000) -> str:
    text = str(value or "").strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n...[truncated]"
    return text


def _result_blocked(result: SearchProviderResult) -> bool:
    raw = result.raw if isinstance(result.raw, dict) else {}
    if raw and is_search_result_blocked(raw):
        return True
    if result.link and result.link != "#" and is_url_blocked(result.link):
        return True
    return False


def format_search_results(
    query: str,
    results: list[SearchProviderResult],
    *,
    answer_box: str = "",
    blocked_count: int = 0,
) -> str:
    capture_structured_basis(
        "search",
        {"results": results, "answer_box": answer_box},
        request={"query": query},
        formatter=GUARD_FORMAT_NORMALIZED_SEARCH_V1,
    )
    if answer_box and is_text_blocked(answer_box):
        blocked_count += 1
        answer_box = ""
    visible: list[SearchProviderResult] = []
    for result in results:
        if _result_blocked(result):
            blocked_count += 1
            continue
        visible.append(result)
    if blocked_count and not answer_box and not visible:
        return blocked_search_message(query=query)

    answer_box_info = f"## Answer\n{_clean_one_line(answer_box, max_chars=1500)}\n\n" if answer_box else ""
    organic_info = ""
    for index, item in enumerate(visible[:10], start=1):
        title = _clean_one_line(item.title or "N/A", max_chars=300)
        link = str(item.link or "#").strip() or "#"
        snippet = _clean_one_line(item.snippet or "N/A", max_chars=1400)
        organic_info += f"{index}. **[{title}]({link})**\n   - {snippet}\n"
    if not organic_info and not answer_box_info:
        return f"No results found for '{query}'. Try with a more general query."
    content = f"# Search Results for: '{query}'\n\n {answer_box_info} ## Web Results\n\n" + organic_info
    if blocked_count:
        content += "\n" + partial_search_note()
    return content


def format_scholar_results(
    query: str,
    results: list[SearchProviderResult],
    *,
    blocked_count: int = 0,
) -> str:
    capture_structured_basis(
        "google_scholar",
        {"results": results},
        request={"query": query},
        formatter=GUARD_FORMAT_NORMALIZED_SCHOLAR_V1,
    )
    snippets: list[str] = []
    for item in results:
        if _result_blocked(item):
            blocked_count += 1
            continue
        idx = len(snippets) + 1
        title = _clean_one_line(item.title or "Untitled", max_chars=400)
        link = str(item.link or "no available link").strip() or "no available link"
        fields: list[str] = []
        if item.source:
            fields.append("\npublicationInfo: " + _clean_one_line(item.source, max_chars=500))
        if item.date:
            fields.append("\nDate published: " + _clean_one_line(item.date, max_chars=100))
        snippet = ("\n" + _clean_one_line(item.snippet, max_chars=1500)) if item.snippet else ""
        snippets.append(f"{idx}. [{title}]({link}){''.join(fields)}\n{snippet}")
    if blocked_count and not snippets:
        return blocked_search_message(query=query)
    content = (
        f"A Google scholar for '{query}' found {len(snippets)} results:\n\n"
        "## Scholar Results\n"
        + "\n\n".join(snippets)
    )
    if blocked_count:
        content += "\n" + partial_search_note()
    return content


def format_visit_result(url: str, goal: str, evidence: str, summary: str) -> str:
    if is_text_blocked(evidence) or is_text_blocked(summary):
        return blocked_visit_message()
    return (
        f"The useful information in {url} for user goal {goal} as follows: \n\n"
        "Evidence in page: \n"
        + str(evidence)
        + "\n\n"
        "Summary: \n"
        + str(summary)
        + "\n\n"
    )


def format_visit_unavailable(url: str, goal: str) -> str:
    return format_visit_result(
        url,
        goal,
        "The provided webpage content could not be accessed. Please check the URL or file format.",
        "The webpage content could not be processed, and therefore, no information is available.",
    )


def _headers_with_key(provider: str, key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if not key:
        return headers
    if provider == "exa":
        headers["x-api-key"] = key
        headers["Authorization"] = f"Bearer {key}"
    elif provider == "tavily":
        headers["Authorization"] = f"Bearer {key}"
    elif provider == "jina":
        headers["Authorization"] = f"Bearer {key}"
    elif provider == "shuyan":
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _provider_missing_key(provider: str) -> str:
    return (
        f"{provider} provider is enabled but no API key is configured. "
        f"Set {provider.upper()}_API_KEY or {provider.upper()}_API_KEYS on the gateway."
    )


def _normalize_result(item: dict[str, Any], *, source: str = "") -> SearchProviderResult:
    title = item.get("title") or item.get("name") or item.get("query") or item.get("url") or item.get("link") or "Untitled"
    link = item.get("link") or item.get("url") or item.get("id") or item.get("sourceUrl") or "#"
    snippet_parts: list[str] = []
    for key in ("snippet", "content", "text", "summary", "description", "raw_content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            snippet_parts.append(value.strip())
    highlights = item.get("highlights")
    if isinstance(highlights, list):
        snippet_parts.extend(str(value).strip() for value in highlights if str(value).strip())
    snippet = " ".join(snippet_parts)
    date = item.get("date") or item.get("year") or item.get("publishedDate") or item.get("published_date") or item.get("datePublished") or ""
    origin = item.get("publicationInfo") or item.get("author") or item.get("source") or item.get("siteName") or source
    image = item.get("image") or item.get("imageUrl") or item.get("thumbnailUrl") or item.get("siteIcon") or item.get("favicon") or ""
    return SearchProviderResult(
        title=str(title),
        link=str(link or "#"),
        snippet=_clean_one_line(snippet, max_chars=_env_int("TOOL_PROVIDER_SNIPPET_MAX_CHARS", 1800)),
        date=str(date or ""),
        source=str(origin or ""),
        image=str(image or ""),
        raw=item,
    )


def _parse_jina_search_text(text: str, *, max_results: int) -> list[SearchProviderResult]:
    markdown_results: list[SearchProviderResult] = []
    seen: set[str] = set()
    pattern = re.compile(r"\[([^\]]{1,300})\]\((https?://[^)\s]+)\)")
    for match in pattern.finditer(text or ""):
        title = match.group(1).strip()
        link = match.group(2).strip()
        if link in seen:
            continue
        seen.add(link)
        start = match.end()
        snippet = _clean_one_line((text[start : start + 900] or "").split("\n\n", 1)[0], max_chars=900)
        markdown_results.append(SearchProviderResult(title=title, link=link, snippet=snippet, source="jina"))
        if len(markdown_results) >= max_results:
            break
    if markdown_results:
        return markdown_results
    trimmed = _trim_text(text, max_chars=_env_int("JINA_SEARCH_FALLBACK_MAX_CHARS", 3500))
    return [SearchProviderResult(title="Search results", link="#", snippet=trimmed, source="jina")] if trimmed else []


def _jina_search_results(query: str) -> list[SearchProviderResult]:
    template = _first_env("JINA_SEARCH_URL_TEMPLATE", "JINA_SEARCH_URL_TEMPLATES", default="https://s.jina.ai/{query}")
    url = template.format(query=quote(query), raw_query=query)
    response = tool_http_request(
        prefix="JINA",
        method="GET",
        url=url,
        headers=_headers_with_key("jina", _random_env("JINA_API_KEY", "JINA_API_KEYS")),
        trace_layer="tool_provider",
        trace_event="jina_search_provider",
        trace_fields={"query": query, "api_url": url},
    )
    response.raise_for_status()
    return _parse_jina_search_text(response.text, max_results=_env_int("JINA_SEARCH_RESULTS", 8))


def _shuyan_response_data(response) -> dict[str, Any]:
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Shuyan provider returned a non-object JSON response")
    code = data.get("code")
    if code not in (None, 0, "0"):
        message = data.get("message") or data.get("msg") or data.get("error") or "unknown error"
        raise RuntimeError(f"Shuyan provider returned code={code}: {message}")
    payload = data.get("data", data)
    if isinstance(payload, dict):
        return payload
    return {"value": payload}


def _shuyan_search_payload(query: str, *, scholar: bool = False) -> dict[str, Any]:
    if scholar:
        query = query + " scholarly paper research article"
    payload: dict[str, Any] = {
        "q": query,
        "num": max(1, min(50, _env_int("SHUYAN_SEARCH_RESULTS", 10))),
    }
    freshness = _first_env("SHUYAN_SEARCH_RANGE", default="")
    if freshness:
        payload["range"] = freshness
    file_type = _first_env("SHUYAN_SEARCH_FILE_TYPE", default="")
    if file_type:
        payload["fileType"] = file_type
    include_sites = _split_pool(os.environ.get("SHUYAN_SEARCH_INCLUDE_SITES"))
    if include_sites:
        payload["includeSite"] = include_sites
    exclude_sites = _split_pool(os.environ.get("SHUYAN_SEARCH_EXCLUDE_SITES"))
    if exclude_sites:
        payload["excludeSite"] = exclude_sites
    return payload


def _shuyan_http_overrides(kind: str) -> dict[str, Any]:
    kind = kind.upper()
    if kind == "SEARCH":
        default_connect = 5.0
        default_read = 45.0
        default_max_time = 60.0
    elif kind == "READER":
        default_connect = 10.0
        default_read = 120.0
        default_max_time = 180.0
    else:
        default_connect = 10.0
        default_read = 120.0
        default_max_time = 180.0
    return {
        "connect_timeout": _env_float_first(
            default_connect,
            f"SHUYAN_{kind}_CONNECT_TIMEOUT_SEC",
            "SHUYAN_CONNECT_TIMEOUT_SEC",
        ),
        "read_timeout": _env_float_first(
            default_read,
            f"SHUYAN_{kind}_READ_TIMEOUT_SEC",
            "SHUYAN_READ_TIMEOUT_SEC",
        ),
        "max_time": _env_float_first(
            default_max_time,
            f"SHUYAN_{kind}_MAX_TIME_SEC",
            "SHUYAN_MAX_TIME_SEC",
        ),
        "requests_failure_cooldown": _env_float_first(
            5.0,
            f"SHUYAN_{kind}_REQUESTS_FAILURE_COOLDOWN_SEC",
            "SHUYAN_REQUESTS_FAILURE_COOLDOWN_SEC",
        ),
    }


def _shuyan_search_results(query: str, *, scholar: bool = False) -> list[SearchProviderResult]:
    key = _random_env("SHUYAN_API_KEY", "SHUYAN_API_KEYS")
    if not key:
        raise RuntimeError(_provider_missing_key("shuyan"))
    api_url = _first_env("SHUYAN_SEARCH_API_URL", "SHUYAN_SEARCH_API_URLS", default="https://api.shuyanai.com/v1/search")
    response = tool_http_request(
        prefix="SHUYAN",
        method="POST",
        url=api_url,
        headers=_headers_with_key("shuyan", key),
        json_payload=_shuyan_search_payload(query, scholar=scholar),
        trace_layer="tool_provider",
        trace_event="shuyan_search_provider",
        trace_fields={"query": query, "api_url": api_url, "scholar": scholar},
        config_overrides=_shuyan_http_overrides("SEARCH"),
    )
    data = _shuyan_response_data(response)
    raw_results = data.get("webPages") or data.get("results") or data.get("items") or []
    return [
        _normalize_result(item, source="shuyan")
        for item in raw_results
        if isinstance(item, dict)
    ]


def _shuyan_visit(url: str, goal: str) -> str:
    key = _random_env("SHUYAN_API_KEY", "SHUYAN_API_KEYS")
    if not key:
        raise RuntimeError(_provider_missing_key("shuyan"))
    api_url = _first_env("SHUYAN_READER_API_URL", "SHUYAN_READER_API_URLS", default="https://api.shuyanai.com/v1/reader")
    payload = {
        "url": url,
        "format": _first_env("SHUYAN_READER_FORMAT", default="markdown") or "markdown",
    }
    response = tool_http_request(
        prefix="SHUYAN",
        method="POST",
        url=api_url,
        headers=_headers_with_key("shuyan", key),
        json_payload=payload,
        trace_layer="tool_provider",
        trace_event="shuyan_reader_provider",
        trace_fields={"url": url, "api_url": api_url},
        config_overrides=_shuyan_http_overrides("READER"),
    )
    data = _shuyan_response_data(response)
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    content = ""
    for key_name in ("markdown", "text", "content", "html", "value"):
        value = data.get(key_name)
        if isinstance(value, str) and value.strip():
            content = value.strip()
            break
    if not content or is_text_blocked(content):
        return format_visit_unavailable(url, goal)
    evidence = _trim_text(content, max_chars=_env_int("SHUYAN_VISIT_EVIDENCE_MAX_CHARS", 12000))
    summary_parts: list[str] = []
    for key_name in ("title", "description"):
        value = metadata.get(key_name)
        if isinstance(value, str) and value.strip():
            summary_parts.append(value.strip())
    summary_parts.append(evidence)
    summary = _trim_text("\n\n".join(summary_parts), max_chars=_env_int("SHUYAN_VISIT_SUMMARY_MAX_CHARS", 3000))
    return format_visit_result(url, goal, evidence, summary)


def _exa_search_payload(query: str, *, scholar: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "type": _first_env("EXA_SEARCH_TYPE", default="auto"),
        "numResults": max(1, _env_int("EXA_SEARCH_RESULTS", 10)),
        "contents": {
            "text": {"maxCharacters": max(1, _env_int("EXA_CONTENT_TEXT_MAX_CHARS", 1200))},
            "highlights": {"numSentences": max(1, _env_int("EXA_HIGHLIGHTS_SENTENCES", 2))},
        },
    }
    if _env_bool("EXA_INCLUDE_SUMMARY", True):
        payload["contents"]["summary"] = {"query": query}
    if scholar:
        category = _first_env("EXA_SCHOLAR_CATEGORY", default="research paper")
        if category:
            payload["category"] = category
    else:
        category = _first_env("EXA_SEARCH_CATEGORY", default="")
        if category:
            payload["category"] = category
    return payload


def _exa_search_results(query: str, *, scholar: bool = False) -> list[SearchProviderResult]:
    key = _random_env("EXA_API_KEY", "EXA_API_KEYS")
    if not key:
        raise RuntimeError(_provider_missing_key("exa"))
    url = _first_env("EXA_SEARCH_API_URL", "EXA_SEARCH_API_URLS", default="https://api.exa.ai/search")
    response = tool_http_request(
        prefix="EXA",
        method="POST",
        url=url,
        headers=_headers_with_key("exa", key),
        json_payload=_exa_search_payload(query, scholar=scholar),
        trace_layer="tool_provider",
        trace_event="exa_search_provider",
        trace_fields={"query": query, "api_url": url, "scholar": scholar},
    )
    response.raise_for_status()
    data = response.json()
    raw_results = data.get("results") if isinstance(data, dict) else []
    return [
        _normalize_result(item, source="exa")
        for item in raw_results
        if isinstance(item, dict)
    ]


def _exa_visit(url: str, goal: str) -> str:
    key = _random_env("EXA_API_KEY", "EXA_API_KEYS")
    if not key:
        raise RuntimeError(_provider_missing_key("exa"))
    api_url = _first_env("EXA_CONTENTS_API_URL", "EXA_CONTENTS_API_URLS", default="https://api.exa.ai/contents")
    payload: dict[str, Any] = {
        "urls": [url],
        "text": {"maxCharacters": max(1, _env_int("EXA_VISIT_TEXT_MAX_CHARS", 12000))},
        "highlights": {"query": goal, "numSentences": max(1, _env_int("EXA_VISIT_HIGHLIGHTS_SENTENCES", 5))},
    }
    if _env_bool("EXA_VISIT_INCLUDE_SUMMARY", True):
        payload["summary"] = {"query": goal}
    response = tool_http_request(
        prefix="EXA",
        method="POST",
        url=api_url,
        headers=_headers_with_key("exa", key),
        json_payload=payload,
        trace_layer="tool_provider",
        trace_event="exa_contents_provider",
        trace_fields={"url": url, "api_url": api_url},
    )
    if response.status_code == 400:
        fallback_payload = dict(payload)
        fallback_payload.pop("urls", None)
        fallback_payload["ids"] = [url]
        response = tool_http_request(
            prefix="EXA",
            method="POST",
            url=api_url,
            headers=_headers_with_key("exa", key),
            json_payload=fallback_payload,
            trace_layer="tool_provider",
            trace_event="exa_contents_provider_retry_ids",
            trace_fields={"url": url, "api_url": api_url},
        )
    response.raise_for_status()
    data = response.json()
    results = data.get("results") if isinstance(data, dict) else []
    if not results:
        return format_visit_unavailable(url, goal)
    item = results[0] if isinstance(results[0], dict) else {}
    evidence_parts: list[str] = []
    highlights = item.get("highlights")
    if isinstance(highlights, list):
        evidence_parts.extend(str(value) for value in highlights if str(value).strip())
    text = item.get("text")
    if isinstance(text, str) and text.strip():
        evidence_parts.append(text)
    evidence = _trim_text("\n\n".join(evidence_parts), max_chars=_env_int("EXA_VISIT_EVIDENCE_MAX_CHARS", 12000))
    summary = item.get("summary") or evidence[: _env_int("EXA_VISIT_SUMMARY_MAX_CHARS", 3000)]
    return format_visit_result(url, goal, evidence or "No directly usable page text returned.", _trim_text(summary, max_chars=3000))


def _tavily_search_payload(query: str, *, scholar: bool = False) -> dict[str, Any]:
    if scholar:
        query = query + " scholarly article research paper"
    payload: dict[str, Any] = {
        "query": query,
        "search_depth": _first_env("TAVILY_SEARCH_DEPTH", default="advanced"),
        "max_results": max(1, _env_int("TAVILY_SEARCH_RESULTS", 10)),
        "include_answer": _env_bool("TAVILY_INCLUDE_ANSWER", True),
        "include_raw_content": _env_bool("TAVILY_INCLUDE_RAW_CONTENT", False),
        "chunks_per_source": max(1, min(3, _env_int("TAVILY_CHUNKS_PER_SOURCE", 3))),
    }
    topic = _first_env("TAVILY_SEARCH_TOPIC", default="")
    if topic:
        payload["topic"] = topic
    return payload


def _tavily_search(query: str, *, scholar: bool = False) -> tuple[list[SearchProviderResult], str]:
    key = _random_env("TAVILY_API_KEY", "TAVILY_API_KEYS")
    if not key:
        raise RuntimeError(_provider_missing_key("tavily"))
    url = _first_env("TAVILY_SEARCH_API_URL", "TAVILY_SEARCH_API_URLS", default="https://api.tavily.com/search")
    response = tool_http_request(
        prefix="TAVILY",
        method="POST",
        url=url,
        headers=_headers_with_key("tavily", key),
        json_payload=_tavily_search_payload(query, scholar=scholar),
        trace_layer="tool_provider",
        trace_event="tavily_search_provider",
        trace_fields={"query": query, "api_url": url, "scholar": scholar},
    )
    response.raise_for_status()
    data = response.json()
    raw_results = data.get("results") if isinstance(data, dict) else []
    results = [
        _normalize_result(item, source="tavily")
        for item in raw_results
        if isinstance(item, dict)
    ]
    answer = str(data.get("answer") or "") if isinstance(data, dict) else ""
    return results, answer


def _tavily_visit(url: str, goal: str) -> str:
    key = _random_env("TAVILY_API_KEY", "TAVILY_API_KEYS")
    if not key:
        raise RuntimeError(_provider_missing_key("tavily"))
    api_url = _first_env("TAVILY_EXTRACT_API_URL", "TAVILY_EXTRACT_API_URLS", default="https://api.tavily.com/extract")
    payload: dict[str, Any] = {
        "urls": [url],
        "query": goal,
        "extract_depth": _first_env("TAVILY_EXTRACT_DEPTH", default="advanced"),
        "chunks_per_source": max(1, min(5, _env_int("TAVILY_EXTRACT_CHUNKS_PER_SOURCE", 5))),
        "include_images": _env_bool("TAVILY_EXTRACT_INCLUDE_IMAGES", False),
        "include_favicon": _env_bool("TAVILY_EXTRACT_INCLUDE_FAVICON", False),
        "timeout": max(1.0, min(60.0, _env_float("TAVILY_EXTRACT_TIMEOUT_SEC", 60.0))),
    }
    response = tool_http_request(
        prefix="TAVILY",
        method="POST",
        url=api_url,
        headers=_headers_with_key("tavily", key),
        json_payload=payload,
        trace_layer="tool_provider",
        trace_event="tavily_extract_provider",
        trace_fields={"url": url, "api_url": api_url},
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("results") if isinstance(data, dict) else []
    if not results:
        return format_visit_unavailable(url, goal)
    item = results[0] if isinstance(results[0], dict) else {}
    raw_content = item.get("raw_content") or item.get("content") or item.get("text") or ""
    evidence = _trim_text(raw_content, max_chars=_env_int("TAVILY_VISIT_EVIDENCE_MAX_CHARS", 12000))
    return format_visit_result(url, goal, evidence or "No directly usable page text returned.", evidence[:3000])


def _advanced_search_results(query: str, *, scholar: bool = False) -> list[SearchProviderResult]:
    from tool_advanced_web import _call_serper, _extract_candidates, _rerank_candidates

    endpoint = "scholar" if scholar else "search"
    limit_name = "ADVANCED_SCHOLAR_RESULTS" if scholar else "ADVANCED_SEARCH_WEB_RESULTS"
    final_name = "ADVANCED_SCHOLAR_FINAL_RESULTS" if scholar else "ADVANCED_SEARCH_FINAL_RESULTS"
    data = _call_serper(endpoint, query)
    candidates, _blocked = _extract_candidates(data, endpoint, limit=max(1, _env_int(limit_name, 8 if not scholar else 5)))
    ranked = _rerank_candidates(query, candidates, top_n=max(1, _env_int(final_name, 6 if not scholar else 5)))
    return [
        SearchProviderResult(
            title=str(item.get("title") or "Untitled"),
            link=str(item.get("link") or "#"),
            snippet=str(item.get("snippet") or ""),
            date=str(item.get("date") or ""),
            source=str(item.get("origin") or item.get("source") or ""),
            image=str(item.get("image") or ""),
            raw=item.get("raw") if isinstance(item.get("raw"), dict) else item,
        )
        for item in ranked
    ]


def strategy_search(query: str) -> str:
    strategy = provider_strategy()
    started_at = time.time()
    try:
        try:
            if strategy == "advanced":
                results = _run_provider_with_retries(
                    tool="search",
                    strategy=strategy,
                    query=query,
                    fn=lambda: _advanced_search_results(query),
                )
                return format_search_results(query, results)
            if strategy == "jina":
                results = _run_provider_with_retries(
                    tool="search",
                    strategy=strategy,
                    query=query,
                    fn=lambda: _jina_search_results(query),
                )
                return format_search_results(query, results)
            if strategy == "exa":
                results = _run_provider_with_retries(
                    tool="search",
                    strategy=strategy,
                    query=query,
                    fn=lambda: _exa_search_results(query),
                )
                return format_search_results(query, results)
            if strategy == "tavily":
                results, answer = _run_provider_with_retries(
                    tool="search",
                    strategy=strategy,
                    query=query,
                    fn=lambda: _tavily_search(query),
                )
                return format_search_results(query, results, answer_box=answer)
            if strategy == "shuyan":
                results = _run_provider_with_retries(
                    tool="search",
                    strategy=strategy,
                    query=query,
                    fn=lambda: _shuyan_search_results(query),
                )
                return format_search_results(query, results)
        except Exception as exc:  # noqa: BLE001
            emit_trace(
                "tool_provider_strategy_error",
                layer="tool_provider",
                tool="search",
                strategy=strategy,
                query=query,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return (
                "Google search could not complete after configured provider retries. "
                "Please try a more specific query or retry later."
            )
    finally:
        emit_trace(
            "tool_provider_strategy_end",
            layer="tool_provider",
            tool="search",
            strategy=strategy,
            query=query,
            duration_s=round(time.time() - started_at, 3),
        )
    raise RuntimeError(f"Unsupported tool provider strategy for search: {strategy}")


def strategy_scholar(query: str) -> str:
    strategy = provider_strategy()
    started_at = time.time()
    try:
        try:
            if strategy == "advanced":
                results = _run_provider_with_retries(
                    tool="google_scholar",
                    strategy=strategy,
                    query=query,
                    fn=lambda: _advanced_search_results(query, scholar=True),
                )
                return format_scholar_results(query, results)
            if strategy == "jina":
                results = _run_provider_with_retries(
                    tool="google_scholar",
                    strategy=strategy,
                    query=query,
                    fn=lambda: _jina_search_results(query + " scholarly paper"),
                )
                return format_scholar_results(query, results)
            if strategy == "exa":
                results = _run_provider_with_retries(
                    tool="google_scholar",
                    strategy=strategy,
                    query=query,
                    fn=lambda: _exa_search_results(query, scholar=True),
                )
                return format_scholar_results(query, results)
            if strategy == "tavily":
                results, _answer = _run_provider_with_retries(
                    tool="google_scholar",
                    strategy=strategy,
                    query=query,
                    fn=lambda: _tavily_search(query, scholar=True),
                )
                return format_scholar_results(query, results)
            if strategy == "shuyan":
                results = _run_provider_with_retries(
                    tool="google_scholar",
                    strategy=strategy,
                    query=query,
                    fn=lambda: _shuyan_search_results(query, scholar=True),
                )
                return format_scholar_results(query, results)
        except Exception as exc:  # noqa: BLE001
            emit_trace(
                "tool_provider_strategy_error",
                layer="tool_provider",
                tool="google_scholar",
                strategy=strategy,
                query=query,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return (
                "Google Scholar could not complete after configured provider retries. "
                f"Please try again later. Last error: {type(exc).__name__}: {exc}"
            )
    finally:
        emit_trace(
            "tool_provider_strategy_end",
            layer="tool_provider",
            tool="google_scholar",
            strategy=strategy,
            query=query,
            duration_s=round(time.time() - started_at, 3),
        )
    raise RuntimeError(f"Unsupported tool provider strategy for google_scholar: {strategy}")


def strategy_visit(url: str, goal: str, *, original_visit: Any) -> str:
    strategy = provider_strategy()
    started_at = time.time()
    try:
        try:
            if strategy == "advanced":
                from tool_advanced_web import AdvancedVisit

                tool = AdvancedVisit()
                tool._reader_options = {"url": url, "goal": goal}
                try:
                    return tool.readpage_jina(url, goal)
                finally:
                    tool._reader_options = {}
            if strategy == "jina":
                with provider_passthrough():
                    return original_visit.readpage_jina(url, goal)
            if strategy == "exa":
                return _run_provider_with_retries(
                    tool="visit",
                    strategy=strategy,
                    url=url,
                    fn=lambda: _exa_visit(url, goal),
                )
            if strategy == "tavily":
                return _run_provider_with_retries(
                    tool="visit",
                    strategy=strategy,
                    url=url,
                    fn=lambda: _tavily_visit(url, goal),
                )
            if strategy == "shuyan":
                return _run_provider_with_retries(
                    tool="visit",
                    strategy=strategy,
                    url=url,
                    fn=lambda: _shuyan_visit(url, goal),
                )
        except Exception as exc:  # noqa: BLE001
            emit_trace(
                "tool_provider_strategy_error",
                layer="tool_provider",
                tool="visit",
                strategy=strategy,
                url=url,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return format_visit_unavailable(url, goal)
    finally:
        emit_trace(
            "tool_provider_strategy_end",
            layer="tool_provider",
            tool="visit",
            strategy=strategy,
            url=url,
            duration_s=round(time.time() - started_at, 3),
        )
    raise RuntimeError(f"Unsupported tool provider strategy for visit: {strategy}")
