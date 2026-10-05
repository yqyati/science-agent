import base64
import hashlib
import json
import mimetypes
import os
import random
import re
import subprocess
import tempfile
import time
from typing import Any, Iterable, List, Union
from urllib.parse import quote, urljoin, urlparse

import requests
from qwen_agent.tools.base import BaseTool, register_tool

from runtime_env import get_env_bool, get_env_float, get_env_int
from tool_http_client import tool_http_config, tool_http_request
from tool_url_guard import (
    blocked_search_message,
    blocked_visit_message,
    is_search_result_blocked,
    is_text_blocked,
    is_url_blocked,
    partial_search_note,
)
from tool_visit_mcp_r1 import Visit, normalize_url_input

try:
    from science_agent.tracing import emit_trace
except Exception:  # pragma: no cover
    def emit_trace(event: str, *, layer: str, **fields):
        return None


TRUE_VALUES = {"1", "true", "yes", "on", "y"}


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
        values = _split_pool(os.getenv(name))
        if values:
            return values[0]
    return default


def _random_env(*names: str, default: str = "") -> str:
    values: list[str] = []
    for name in names:
        values.extend(_split_pool(os.getenv(name)))
    return random.choice(values) if values else default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    return get_env_int(name, default=default)


def _env_float(name: str, default: float) -> float:
    return get_env_float(name, default=default)


def _derive_endpoint(base_url: str, endpoint: str) -> str:
    base_url = (base_url or "https://google.serper.dev/search").strip().rstrip("/")
    parts = base_url.split("/")
    if parts:
        parts[-1] = endpoint.strip("/")
        return "/".join(parts)
    return f"https://google.serper.dev/{endpoint.strip('/')}"


def _serper_url(endpoint: str) -> str:
    upper = endpoint.upper()
    explicit = _first_env(
        f"SERPER_{upper}_API_URL",
        f"SERPER_{upper}_API_URLS",
        default="",
    )
    if explicit:
        return explicit
    if endpoint == "search":
        return _first_env("SERPER_API_URL", "SERPER_SEARCH_API_URL", "SERPER_SEARCH_API_URLS", default="https://google.serper.dev/search")
    base = _first_env("SERPER_API_URL", "SERPER_SEARCH_API_URL", "SERPER_SEARCH_API_URLS", default="https://google.serper.dev/search")
    return _derive_endpoint(base, endpoint)


def _jina_key() -> str:
    return _random_env("JINA_API_KEY", "JINA_API_KEYS")


def _serper_key() -> str:
    return _random_env("SERPER_API_KEY", "SERPER_API_KEYS", "SERPER_KEY_ID")


def _serper_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = _serper_key()
    if key:
        headers["X-API-KEY"] = key
    return headers


def _jina_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = _jina_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _query_payload(query: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"q": query}
    if any("\u4e00" <= char <= "\u9fff" for char in query):
        payload.update({"location": "China", "gl": "cn", "hl": "zh-cn"})
    else:
        payload.update({"location": "United States", "gl": "us", "hl": "en"})
    num = _env_int("ADVANCED_SEARCH_REQUEST_NUM", 8)
    if num > 0:
        payload["num"] = num
    return payload


def _call_serper(endpoint: str, query: str) -> dict[str, Any]:
    url = _serper_url(endpoint)
    payload = _query_payload(query)
    started_at = time.time()
    config = tool_http_config("SERPER")
    emit_trace(
        "advanced_serper_request_start",
        layer="tool_advanced_search",
        endpoint=endpoint,
        query=query,
        api_url=url,
        backend=config["backend"],
        connect_timeout_s=config["connect_timeout"],
        read_timeout_s=config["read_timeout"],
        max_time_s=config["max_time"],
    )
    response = tool_http_request(
        prefix="SERPER",
        method="POST",
        url=url,
        headers=_serper_headers(),
        json_payload=payload,
        trace_layer="tool_advanced_search",
        trace_event=f"advanced_serper_{endpoint}",
        trace_fields={"endpoint": endpoint, "query": query, "api_url": url},
    )
    emit_trace(
        "advanced_serper_request_end",
        layer="tool_advanced_search",
        endpoint=endpoint,
        query=query,
        api_url=url,
        status_code=response.status_code,
        backend=response.backend,
        response_chars=len(response.text or ""),
        duration_s=round(time.time() - started_at, 3),
    )
    response.raise_for_status()
    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"{endpoint} returned invalid JSON: {response.text[:300]}") from exc
    return data if isinstance(data, dict) else {"raw": data}


def _candidate_text(candidate: dict[str, Any]) -> str:
    fields = [
        candidate.get("title", ""),
        candidate.get("snippet", ""),
        candidate.get("source", ""),
        candidate.get("date", ""),
        candidate.get("link", ""),
    ]
    return "\n".join(str(field) for field in fields if str(field).strip())


def _candidate_from_item(item: dict[str, Any], source: str) -> dict[str, Any]:
    link = (
        item.get("link")
        or item.get("url")
        or item.get("imageUrl")
        or item.get("thumbnailUrl")
        or item.get("patentLink")
        or ""
    )
    title = item.get("title") or item.get("name") or item.get("query") or link or "Untitled"
    snippet = (
        item.get("snippet")
        or item.get("description")
        or item.get("summary")
        or item.get("source")
        or ""
    )
    return {
        "source": source,
        "title": str(title),
        "link": str(link),
        "snippet": str(snippet),
        "date": str(item.get("date") or item.get("year") or ""),
        "origin": str(item.get("source") or item.get("publicationInfo") or ""),
        "image": str(item.get("imageUrl") or item.get("thumbnailUrl") or ""),
        "raw": item,
    }


def _items_for_endpoint(data: dict[str, Any], endpoint: str) -> list[dict[str, Any]]:
    if endpoint == "news":
        keys = ("news", "organic", "topStories")
    elif endpoint == "images":
        keys = ("images", "organic")
    elif endpoint == "patents":
        keys = ("organic", "patents")
    elif endpoint == "scholar":
        keys = ("organic",)
    else:
        keys = ("organic",)
    out: list[dict[str, Any]] = []
    for key in keys:
        values = data.get(key)
        if isinstance(values, list):
            out.extend(item for item in values if isinstance(item, dict))
    return out


def _extract_candidates(data: dict[str, Any], endpoint: str, *, limit: int) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    blocked = 0
    answer = data.get("answerBox")
    if isinstance(answer, dict) and endpoint == "search":
        if is_search_result_blocked(answer):
            blocked += 1
        else:
            candidates.append(_candidate_from_item(answer, "answer"))
    for item in _items_for_endpoint(data, endpoint):
        if is_search_result_blocked(item):
            blocked += 1
            continue
        candidate = _candidate_from_item(item, endpoint)
        if candidate.get("link") and is_url_blocked(str(candidate["link"])):
            blocked += 1
            continue
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates, blocked


def _autocomplete_suggestions(query: str) -> list[str]:
    if not _env_bool("ADVANCED_SEARCH_AUTOCOMPLETE", False):
        return []
    try:
        data = _call_serper("autocomplete", query)
    except Exception as exc:
        emit_trace(
            "advanced_autocomplete_error",
            layer="tool_advanced_search",
            query=query,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return []
    raw = data.get("suggestions") or data.get("autocomplete") or data.get("organic") or []
    suggestions: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                value = item
            elif isinstance(item, dict):
                value = str(item.get("value") or item.get("query") or item.get("title") or "")
            else:
                value = ""
            value = value.strip()
            if value and value.lower() != query.lower() and value not in suggestions:
                suggestions.append(value)
    return suggestions[: max(0, _env_int("ADVANCED_SEARCH_MAX_AUTOCOMPLETE", 3))]


def _jina_search(query: str) -> str:
    template = _first_env("JINA_SEARCH_URL_TEMPLATE", "JINA_SEARCH_URL_TEMPLATES", default="https://s.jina.ai/{query}")
    url = template.format(query=quote(query), raw_query=query)
    response = tool_http_request(
        prefix="JINA",
        method="GET",
        url=url,
        headers=_jina_headers(),
        trace_layer="tool_advanced_search",
        trace_event="advanced_fallback_search",
        trace_fields={"query": query, "api_url": url},
    )
    response.raise_for_status()
    return response.text


def _rerank_candidates(query: str, candidates: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
    if not candidates or not _env_bool("ADVANCED_SEARCH_RERANK", True):
        return candidates[:top_n]
    if not _jina_key():
        return candidates[:top_n]
    documents = [_candidate_text(candidate) for candidate in candidates]
    max_docs = max(1, _env_int("ADVANCED_SEARCH_RERANK_MAX_DOCS", 12))
    documents = documents[:max_docs]
    candidates = candidates[:max_docs]
    payload = {
        "model": _first_env("JINA_RERANK_MODEL", default="jina-reranker-v2-base-multilingual"),
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }
    url = _first_env("JINA_RERANK_API_URL", "JINA_RERANK_API_URLS", default="https://api.jina.ai/v1/rerank")
    try:
        response = tool_http_request(
            prefix="JINA",
            method="POST",
            url=url,
            headers=_jina_headers(),
            json_payload=payload,
            trace_layer="tool_advanced_search",
            trace_event="advanced_rerank",
            trace_fields={"query": query, "api_url": url, "documents": len(documents)},
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        emit_trace(
            "advanced_rerank_error",
            layer="tool_advanced_search",
            query=query,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return candidates[:top_n]
    ranked: list[dict[str, Any]] = []
    for result in data.get("results") or []:
        if not isinstance(result, dict):
            continue
        try:
            index = int(result.get("index"))
        except Exception:
            continue
        if index < 0 or index >= len(candidates):
            continue
        candidate = dict(candidates[index])
        score = result.get("relevance_score")
        if score is not None:
            candidate["score"] = score
        ranked.append(candidate)
    seen = {id(item.get("raw")) for item in ranked}
    for candidate in candidates:
        if id(candidate.get("raw")) not in seen and len(ranked) < top_n:
            ranked.append(candidate)
    return ranked[:top_n]


def _format_candidates(title: str, candidates: list[dict[str, Any]], *, limit: int) -> str:
    if not candidates:
        return f"## {title}\nNo usable results.\n"
    lines = [f"## {title}"]
    for index, item in enumerate(candidates[:limit], start=1):
        title_text = str(item.get("title") or "Untitled").replace("\n", " ").strip()
        link = str(item.get("link") or "").strip()
        snippet = str(item.get("snippet") or "").replace("\n", " ").strip()
        prefix = f"{index}. "
        if link:
            lines.append(f"{prefix}**[{title_text}]({link})**")
        else:
            lines.append(f"{prefix}**{title_text}**")
        details: list[str] = []
        if item.get("origin"):
            details.append(str(item["origin"]))
        if item.get("date"):
            details.append(str(item["date"]))
        if item.get("score") is not None:
            details.append(f"relevance={float(item['score']):.4f}")
        if details:
            lines.append("   - " + " | ".join(details))
        if snippet:
            lines.append("   - " + snippet)
        if item.get("image") and item.get("image") != link:
            lines.append("   - image: " + str(item["image"]))
    return "\n".join(lines) + "\n"


IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)", re.IGNORECASE)
IMAGE_URL_RE = re.compile(r"https?://[^\s<>)\"']+", re.IGNORECASE)
IMAGE_EXTENSION_RE = re.compile(r"\.(?:png|jpe?g|webp|gif|bmp|tiff?)(?:$|[?#])", re.IGNORECASE)
DEFAULT_VISUAL_MIME_TYPES = "image/png,image/jpeg,image/webp,image/gif"
VISUAL_INTENT_RE = re.compile(
    r"\b("
    r"figure|fig\.?|image|photo|picture|diagram|plot|chart|graph|visual|"
    r"ocr|spectrum|spectra|microscopy|microscope|x[- ]?ray|scan|"
    r"table|caption|drawing|map|geometry|measure|crop"
    r")\b",
    re.IGNORECASE,
)


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _visual_injection_enabled() -> bool:
    return _env_bool("ADVANCED_VISUAL_INJECTION", True)


def _visual_context_allows_injection(context: str = "") -> bool:
    if not _visual_injection_enabled():
        return False
    mode = _first_env("ADVANCED_VISUAL_INJECTION_MODE", default="visual_goal").strip().lower()
    if mode in {"always", "all", "on", "1", "true", "yes"}:
        return True
    if mode in {"never", "none", "off", "0", "false", "no"}:
        return False
    return bool(VISUAL_INTENT_RE.search(str(context or "")))


def _visual_max_images() -> int:
    return max(0, _env_int("ADVANCED_VISUAL_MAX_IMAGES", 1))


def _visual_download_retries() -> int:
    return max(1, _env_int("ADVANCED_VISUAL_DOWNLOAD_RETRIES", 3))


def _visual_allowed_mime_types() -> set[str]:
    raw = _first_env("ADVANCED_VISUAL_ALLOWED_MIME_TYPES", default=DEFAULT_VISUAL_MIME_TYPES)
    values = {part.strip().lower() for part in raw.split(",") if part.strip()}
    return values or {part.strip() for part in DEFAULT_VISUAL_MIME_TYPES.split(",")}


def _visual_http_config() -> dict[str, Any]:
    config = dict(tool_http_config("ADVANCED_VISUAL"))
    if not config.get("proxy_url"):
        proxy = _first_env(
            "ADVANCED_VISUAL_PROXY_URL",
            "ADVANCED_VISUAL_PROXY_URLS",
            "JINA_PROXY_URL",
            "JINA_PROXY_URLS",
            "SERPER_PROXY_URL",
            "SERPER_PROXY_URLS",
            default="",
        )
        if proxy:
            config["proxy_url"] = proxy
    timeout = os.getenv("ADVANCED_VISUAL_DOWNLOAD_TIMEOUT_SEC")
    if timeout is not None and str(timeout).strip():
        try:
            timeout_f = max(0.1, float(str(timeout).strip()))
            config["connect_timeout"] = min(float(config["connect_timeout"]), timeout_f)
            config["read_timeout"] = timeout_f
            config["max_time"] = max(timeout_f, float(config["connect_timeout"]))
        except ValueError:
            pass
    return config


def _visual_request_headers() -> dict[str, str]:
    return {
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "User-Agent": "science-agent-advanced-visual/1.0",
    }


def _content_type(headers: dict[str, str] | None) -> str:
    headers = headers or {}
    for key, value in headers.items():
        if key.lower() == "content-type":
            return str(value).split(";", 1)[0].strip().lower()
    return ""


def _sniff_image_mime(image_bytes: bytes, headers: dict[str, str] | None, url: str) -> str:
    content_type = _content_type(headers)
    if content_type.startswith("image/"):
        return content_type
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    guessed, _ = mimetypes.guess_type(url)
    return str(guessed or "").lower()


def _requests_image_get(url: str, config: dict[str, Any]) -> tuple[int, dict[str, str], bytes, str]:
    session = requests.Session()
    session.trust_env = bool(config["use_env_proxy"])
    proxies = None
    if config.get("proxy_url"):
        proxies = {"http": str(config["proxy_url"]), "https": str(config["proxy_url"])}
        session.trust_env = False
    response = session.get(
        url,
        headers=_visual_request_headers(),
        timeout=(float(config["connect_timeout"]), float(config["read_timeout"])),
        proxies=proxies,
    )
    return response.status_code, dict(response.headers), response.content, "requests"


def _curl_image_get(url: str, config: dict[str, Any]) -> tuple[int, dict[str, str], bytes, str]:
    with tempfile.NamedTemporaryFile() as body_file, tempfile.NamedTemporaryFile() as header_file:
        cmd = [
            "curl",
            "-sS",
            "-L",
            "--compressed",
            "--connect-timeout",
            str(float(config["connect_timeout"])),
            "--max-time",
            str(float(config["max_time"])),
            "-D",
            header_file.name,
            "-o",
            body_file.name,
        ]
        if config.get("proxy_url"):
            cmd.extend(["--proxy", str(config["proxy_url"])])
        elif not config.get("use_env_proxy"):
            cmd.extend(["--noproxy", "*"])
        for key, value in _visual_request_headers().items():
            cmd.extend(["-H", f"{key}: {value}"])
        cmd.extend(["-w", "%{http_code}", url])
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(config["max_time"]) + 5.0,
            check=False,
        )
        status_text = (proc.stdout or "").strip()
        status_code = int(status_text[-3:] or "0") if status_text[-3:].isdigit() else 0
        if proc.returncode != 0:
            error = (proc.stderr or "").strip() or f"curl exited with code {proc.returncode}"
            raise RuntimeError(f"{error}; http_status={status_code}")
        header_file.seek(0)
        headers: dict[str, str] = {}
        for raw_line in header_file.read().decode("utf-8", errors="replace").splitlines():
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            headers[key.strip()] = value.strip()
        body_file.seek(0)
        return status_code, headers, body_file.read(), "curl"


def _download_visual_image(url: str) -> tuple[str, bytes, str] | None:
    if not _visual_injection_enabled():
        return None
    if not url or is_url_blocked(url):
        emit_trace("advanced_visual_skip", layer="tool_advanced_visual", url=url, reason="blocked_or_empty")
        return None

    config = _visual_http_config()
    backend_order = [str(config.get("backend") or "auto").lower()]
    if backend_order[0] == "auto":
        backend_order = ["requests", "curl"]
    allowed_mime_types = _visual_allowed_mime_types()
    last_error = ""
    for attempt in range(1, _visual_download_retries() + 1):
        for backend in backend_order:
            started_at = time.time()
            emit_trace(
                "advanced_visual_download_start",
                layer="tool_advanced_visual",
                url=url,
                attempt=attempt,
                max_attempts=_visual_download_retries(),
                backend=backend,
                connect_timeout_s=config["connect_timeout"],
                read_timeout_s=config["read_timeout"],
                max_time_s=config["max_time"],
            )
            try:
                if backend == "curl":
                    status_code, headers, image_bytes, used_backend = _curl_image_get(url, config)
                else:
                    status_code, headers, image_bytes, used_backend = _requests_image_get(url, config)
                mime_type = _sniff_image_mime(image_bytes, headers, url)
                if status_code >= 400:
                    raise RuntimeError(f"http_status={status_code}")
                if mime_type not in allowed_mime_types:
                    raise RuntimeError(f"unsupported_mime={mime_type or 'unknown'}")
                if not image_bytes:
                    raise RuntimeError("empty_image_response")
                emit_trace(
                    "advanced_visual_download_end",
                    layer="tool_advanced_visual",
                    url=url,
                    attempt=attempt,
                    backend=used_backend,
                    status_code=status_code,
                    mime_type=mime_type,
                    image_bytes=len(image_bytes),
                    duration_s=round(time.time() - started_at, 3),
                )
                return mime_type, image_bytes, used_backend
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                emit_trace(
                    "advanced_visual_download_error",
                    layer="tool_advanced_visual",
                    url=url,
                    attempt=attempt,
                    backend=backend,
                    duration_s=round(time.time() - started_at, 3),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                if str(config.get("backend") or "auto").lower() != "auto":
                    break
        if attempt < _visual_download_retries():
            time.sleep(max(0.0, _env_float("ADVANCED_VISUAL_RETRY_SLEEP_SEC", 0.5)))
    emit_trace("advanced_visual_download_failed", layer="tool_advanced_visual", url=url, error=last_error)
    return None


def _image_content_part(mime_type: str, image_bytes: bytes) -> dict[str, Any]:
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}}


def _extract_visual_urls_from_candidate(candidate: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ("image", "imageUrl", "thumbnailUrl", "thumbnail", "image_url"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            urls.append(value.strip())
    raw = candidate.get("raw")
    if isinstance(raw, dict):
        for key in ("imageUrl", "thumbnailUrl", "thumbnail", "image", "original", "sourceUrl"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                urls.append(value.strip())
    link = str(candidate.get("link") or "").strip()
    if link and IMAGE_EXTENSION_RE.search(link):
        urls.append(link)
    return _dedupe_keep_order(urls)


def _visual_candidates_from_search(candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for candidate in candidates:
        for url in _extract_visual_urls_from_candidate(candidate):
            out.append(
                {
                    "url": url,
                    "title": str(candidate.get("title") or "visual evidence"),
                    "source": str(candidate.get("source") or ""),
                    "page_url": str(candidate.get("link") or ""),
                }
            )
    return out


def _extract_visual_urls_from_text(text: str, base_urls: Iterable[str] = ()) -> list[str]:
    urls: list[str] = []
    bases = [str(base).strip() for base in base_urls if str(base).strip()]
    default_base = bases[0] if bases else ""
    for match in IMAGE_MARKDOWN_RE.finditer(text or ""):
        url = match.group(1).strip().strip("<>")
        if default_base:
            url = urljoin(default_base, url)
        urls.append(url)
    for match in IMAGE_URL_RE.finditer(text or ""):
        url = match.group(0).strip().rstrip(".,;]")
        if IMAGE_EXTENSION_RE.search(url):
            urls.append(url)
    return _dedupe_keep_order(urls)


def _visual_candidates_from_visit(text: str, visited_urls: Iterable[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for url in _extract_visual_urls_from_text(text, visited_urls):
        out.append({"url": url, "title": "visual evidence from visited page", "source": "visit", "page_url": ""})
    return out


def _format_visual_tool_response(
    tool_name: str,
    text: str,
    candidates: list[dict[str, str]],
    *,
    visual_context: str = "",
):
    if not _visual_context_allows_injection(visual_context) or not candidates:
        return text
    max_images = _visual_max_images()
    content: list[dict[str, Any]] = []
    visual_notes: list[str] = []
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    for candidate in candidates:
        if max_images and len(content) >= max_images:
            break
        url = str(candidate.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        downloaded = _download_visual_image(url)
        if downloaded is None:
            continue
        mime_type, image_bytes, backend = downloaded
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        content.append(_image_content_part(mime_type, image_bytes))
        if _env_bool("ADVANCED_VISUAL_INCLUDE_SOURCE_URLS", True):
            title = str(candidate.get("title") or "visual evidence").replace("\n", " ").strip()
            visual_notes.append(f"- image {len(content)}: {title} ({url})")
        emit_trace(
            "advanced_visual_injected",
            layer="tool_advanced_visual",
            tool=tool_name,
            url=url,
            mime_type=mime_type,
            image_bytes=len(image_bytes),
            backend=backend,
        )
    if not content:
        return text
    note = "## Visual Evidence Injected\nThe following image evidence is attached directly in this tool response."
    if visual_notes:
        note += "\n" + "\n".join(visual_notes)
    return [{"type": "text", "text": f"{text.rstrip()}\n\n{note}"}] + content


def _normalize_queries(value: Any) -> list[str]:
    if isinstance(value, str):
        queries = [value]
    elif isinstance(value, list):
        queries = [str(item) for item in value if str(item).strip()]
    else:
        queries = []
    return [query.strip() for query in queries if query.strip()]


def _arxiv_fallback_urls(url: str) -> list[str]:
    parsed = urlparse(str(url or "").strip())
    if not parsed.netloc.endswith("arxiv.org"):
        return []
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2 or parts[0] not in {"html", "abs", "pdf"}:
        return []
    arxiv_id = "/".join(parts[1:]).removesuffix(".pdf")
    if not arxiv_id:
        return []
    candidates: list[str] = []
    if parts[0] == "html":
        candidates.extend([f"https://arxiv.org/abs/{arxiv_id}", f"https://arxiv.org/pdf/{arxiv_id}"])
    elif parts[0] == "abs":
        candidates.append(f"https://arxiv.org/pdf/{arxiv_id}")
    elif parts[0] == "pdf":
        candidates.append(f"https://arxiv.org/abs/{arxiv_id}")
    return [candidate for candidate in _dedupe_keep_order(candidates) if candidate != url]


def _advanced_visit_raw_unusable(content: str) -> bool:
    text = str(content or "")
    lowered = text.lower()
    markers = (
        "[read_page] failed",
        "[document_parser]",
        "no html for",
        "html is not available for the source",
        "conversion failure",
        "could not be accessed",
        "could not be processed",
    )
    return not text.strip() or any(marker in lowered for marker in markers)


def _advanced_visit_summary_unusable(text: str) -> bool:
    lowered = str(text or "").lower()
    markers = (
        "the provided webpage content could not be accessed",
        "the webpage content could not be processed",
        "no meaningful analysis",
        "html is unavailable",
        "html is not available for the source",
    )
    useful_markers = ("## title", "## abstract", "evidence in page", "summary:")
    return any(marker in lowered for marker in markers) and not (
        "## abstract" in lowered or "doi:" in lowered or "arxiv:" in lowered
    ) and any(marker in lowered for marker in useful_markers)


@register_tool("advanced_search", allow_overwrite=True)
class AdvancedSearch(BaseTool):
    name = "advanced_search"
    description = "Search a small set of high-relevance web sources for science/research evidence, with optional reranking and fallback search. Use when normal search is insufficient and you need precise candidate URLs or concise external evidence."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "One or more search queries. Include precise scientific terms, entities, or alternative query forms.",
            },
            "goal": {
                "type": "string",
                "description": "Optional goal describing what evidence is needed.",
            },
        },
        "required": ["query"],
    }

    def call(self, params: Union[str, dict], **kwargs):
        del kwargs
        if not isinstance(params, dict):
            return "[advanced_search] Invalid request format: Input must be a JSON object containing 'query'."
        queries = _normalize_queries(params.get("query"))
        if not queries:
            return "[advanced_search] Invalid request format: 'query' cannot be empty."
        goal = str(params.get("goal") or "").strip()
        outputs: list[str] = []
        visual_candidates: list[dict[str, str]] = []
        for query in queries:
            text, candidates = self._search_one(query, goal=goal)
            outputs.append(text)
            visual_candidates.extend(candidates)
        text = "\n=======\n".join(outputs).strip()
        return _format_visual_tool_response(
            "advanced_search",
            text,
            visual_candidates,
            visual_context="\n".join([*queries, goal]),
        )

    def _search_one(self, query: str, *, goal: str = "") -> tuple[str, list[dict[str, str]]]:
        output_parts = [f"# Advanced Search Results for: '{query}'"]
        if goal:
            output_parts.append(f"Goal: {goal}")
        blocked_count = 0
        all_rankable: list[dict[str, Any]] = []
        notes: list[str] = []
        visual_candidates: list[dict[str, str]] = []
        web_queries = [query]
        suggestions = _autocomplete_suggestions(query)
        expanded_count = max(0, _env_int("ADVANCED_SEARCH_EXPANDED_QUERIES_PER_QUERY", 0))
        web_queries.extend(suggestions[:expanded_count])
        if suggestions and _env_bool("ADVANCED_SEARCH_SHOW_SUGGESTIONS", False):
            output_parts.append("Suggested query variants: " + "; ".join(suggestions))

        for web_query in web_queries:
            try:
                data = _call_serper("search", web_query)
                candidates, blocked = _extract_candidates(
                    data,
                    "search",
                    limit=max(1, _env_int("ADVANCED_SEARCH_WEB_RESULTS", 8)),
                )
                blocked_count += blocked
                for item in candidates:
                    item["source"] = "web"
                all_rankable.extend(candidates)
            except Exception as exc:
                notes.append(f"web search failed for '{web_query}': {type(exc).__name__}: {exc}")
                emit_trace(
                    "advanced_search_backend_error",
                    layer="tool_advanced_search",
                    backend="serper",
                    query=web_query,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

        fallback_text = ""
        mode = _first_env("ADVANCED_SEARCH_FALLBACK_MODE", "JINA_SEARCH_MODE", default="fallback").lower()
        if (not all_rankable and mode in {"fallback", "auto"}) or mode == "augment":
            try:
                fallback_text = _jina_search(query)
            except Exception as exc:
                notes.append(f"fallback search failed: {type(exc).__name__}: {exc}")
                emit_trace(
                    "advanced_search_backend_error",
                    layer="tool_advanced_search",
                    backend="fallback",
                    query=query,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

        ranked = _rerank_candidates(
            goal or query,
            all_rankable,
            top_n=max(1, _env_int("ADVANCED_SEARCH_FINAL_RESULTS", 6)),
        )
        output_parts.append(_format_candidates("Web Evidence", ranked, limit=len(ranked)))
        visual_candidates.extend(_visual_candidates_from_search(ranked))

        for endpoint, env_name, title, default_limit in (
            ("news", "ADVANCED_SEARCH_NEWS", "Current News", 3),
            ("patents", "ADVANCED_SEARCH_PATENTS", "Patent Evidence", 3),
            ("images", "ADVANCED_SEARCH_IMAGES", "Image Evidence", 3),
        ):
            if not _env_bool(env_name, False):
                continue
            try:
                data = _call_serper(endpoint, query)
                candidates, blocked = _extract_candidates(
                    data,
                    endpoint,
                    limit=max(1, _env_int(f"{env_name}_RESULTS", default_limit)),
                )
                blocked_count += blocked
                output_parts.append(_format_candidates(title, candidates, limit=len(candidates)))
                if endpoint == "images":
                    visual_candidates.extend(_visual_candidates_from_search(candidates))
            except Exception as exc:
                notes.append(f"{title.lower()} search failed: {type(exc).__name__}: {exc}")
                emit_trace(
                    "advanced_search_backend_error",
                    layer="tool_advanced_search",
                    backend=endpoint,
                    query=query,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

        if fallback_text:
            output_parts.append("## Fallback Search Text\n" + fallback_text[: max(1000, _env_int("ADVANCED_SEARCH_FALLBACK_MAX_CHARS", 4000))])
        if blocked_count:
            output_parts.append(partial_search_note())
        if notes and _env_bool("ADVANCED_SEARCH_SHOW_NOTES", False):
            output_parts.append("## Search Notes\n" + "\n".join(f"- {note}" for note in notes))
        if not all_rankable and not fallback_text and blocked_count:
            return blocked_search_message(query=query), []
        if not all_rankable and not fallback_text:
            output_parts.append("No usable evidence found for this query.")
        text = "\n\n".join(part for part in output_parts if part).strip()
        return text, visual_candidates


@register_tool("advanced_scholar", allow_overwrite=True)
class AdvancedScholar(BaseTool):
    name = "advanced_scholar"
    description = "Search a concise set of scholarly literature and research metadata. Use when peer-reviewed papers, academic sources, citations, or PDF/source links are central."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "One or more scholarly queries.",
            },
            "goal": {
                "type": "string",
                "description": "Optional goal describing the evidence or claim to verify.",
            },
        },
        "required": ["query"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        del kwargs
        if not isinstance(params, dict):
            return "[advanced_scholar] Invalid request format: Input must be a JSON object containing 'query'."
        queries = _normalize_queries(params.get("query"))
        if not queries:
            return "[advanced_scholar] Invalid request format: 'query' cannot be empty."
        goal = str(params.get("goal") or "").strip()
        return "\n=======\n".join(self._scholar_one(query, goal=goal) for query in queries).strip()

    def _scholar_one(self, query: str, *, goal: str = "") -> str:
        parts = [f"# Advanced Scholar Results for: '{query}'"]
        notes: list[str] = []
        blocked_count = 0
        try:
            data = _call_serper("scholar", query)
            candidates, blocked = _extract_candidates(
                data,
                "scholar",
                limit=max(1, _env_int("ADVANCED_SCHOLAR_RESULTS", 5)),
            )
            blocked_count += blocked
        except Exception as exc:
            candidates = []
            notes.append(f"scholarly search failed: {type(exc).__name__}: {exc}")

        ranked = _rerank_candidates(
            goal or query,
            candidates,
            top_n=max(1, _env_int("ADVANCED_SCHOLAR_FINAL_RESULTS", 5)),
        )
        parts.append(_format_candidates("Scholarly Evidence", ranked, limit=len(ranked)))

        if _env_bool("ADVANCED_SCHOLAR_PATENTS", False):
            try:
                data = _call_serper("patents", query)
                patent_candidates, blocked = _extract_candidates(
                    data,
                    "patents",
                    limit=max(1, _env_int("ADVANCED_SCHOLAR_PATENT_RESULTS", 5)),
                )
                blocked_count += blocked
                parts.append(_format_candidates("Related Patent Evidence", patent_candidates, limit=len(patent_candidates)))
            except Exception as exc:
                notes.append(f"patent search failed: {type(exc).__name__}: {exc}")
        if blocked_count:
            parts.append(partial_search_note())
        if notes and _env_bool("ADVANCED_SCHOLAR_SHOW_NOTES", False):
            parts.append("## Scholar Notes\n" + "\n".join(f"- {note}" for note in notes))
        if not candidates and blocked_count:
            return blocked_search_message(query=query)
        return "\n\n".join(parts).strip()


@register_tool("advanced_visit", allow_overwrite=True)
class AdvancedVisit(Visit):
    name = "advanced_visit"
    description = "Read known URLs for targeted scientific evidence extraction, including webpages, PDFs, dynamic pages, and structured metadata. Use after search or when a URL is given; avoid repeatedly reading URLs that return no usable evidence."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) to read. Can be a single URL or an array of URLs.",
            },
            "goal": {
                "type": "string",
                "description": "The exact evidence to extract from the URL(s).",
            },
            "extract_only": {
                "type": "string",
                "description": "Optional CSS selector for focused extraction, such as article or .main-content.",
            },
            "wait_for": {
                "type": "string",
                "description": "Optional CSS selector to wait for on dynamic pages.",
            },
            "exclude": {
                "type": "string",
                "description": "Optional CSS selector to remove before extraction, such as nav or footer.",
            },
            "include_image_captions": {
                "type": "boolean",
                "description": "Whether image captions may help understand figures, diagrams, or visual evidence.",
            },
            "token_budget": {
                "type": "integer",
                "description": "Optional extraction token budget for very long pages.",
            },
        },
        "required": ["url", "goal"],
    }

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self._reader_options: dict[str, Any] = {}

    def call(self, params: Union[str, dict], **kwargs):
        if not isinstance(params, dict):
            return "[advanced_visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal'."
        try:
            url = normalize_url_input(params["url"])
            goal = str(params["goal"])
        except Exception:
            return "[advanced_visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal'."
        self._reader_options = dict(params)
        try:
            text = super().call({"url": url, "goal": goal}, **kwargs)
            visited_urls = [url] if isinstance(url, str) else [str(item) for item in url]
            if _advanced_visit_summary_unusable(text):
                return "[advanced_visit] No usable evidence could be extracted from: " + ", ".join(visited_urls)
            visual_candidates = _visual_candidates_from_visit(text, visited_urls)
            return _format_visual_tool_response(
                "advanced_visit",
                text,
                visual_candidates,
                visual_context="\n".join([goal, *visited_urls]),
            )
        finally:
            self._reader_options = {}

    def html_readpage_jina(self, url: str) -> str:
        content = super().html_readpage_jina(url)
        if not _advanced_visit_raw_unusable(content):
            return content
        for fallback_url in _arxiv_fallback_urls(url):
            emit_trace(
                "advanced_visit_arxiv_fallback",
                layer="tool_advanced_visit",
                original_url=url,
                fallback_url=fallback_url,
            )
            fallback_content = super().html_readpage_jina(fallback_url)
            if not _advanced_visit_raw_unusable(fallback_content):
                return fallback_content
        return content

    def _build_jina_request(self, url: str):
        request_url, headers = super()._build_jina_request(url)
        options = self._reader_options or {}
        if _env_bool("ADVANCED_VISIT_JSON_RESPONSE", False):
            headers["Accept"] = "application/json"
        if _env_bool("ADVANCED_VISIT_READERLM", True):
            headers["x-respond-with"] = _first_env("ADVANCED_VISIT_RESPOND_WITH", default="readerlm-v2")
        if _env_bool("ADVANCED_VISIT_USE_GOAL_INSTRUCTION", True):
            goal = str(options.get("goal") or "").strip()
            if goal:
                headers["x-instruction"] = "Extract evidence relevant to this goal: " + goal
        if options.get("extract_only"):
            headers["x-target-selector"] = str(options["extract_only"])
        if options.get("wait_for"):
            headers["x-wait-for-selector"] = str(options["wait_for"])
        if options.get("exclude"):
            headers["x-remove-selector"] = str(options["exclude"])
        include_captions = options.get("include_image_captions")
        if include_captions is None:
            include_captions = _env_bool("ADVANCED_VISIT_IMAGE_CAPTIONS", False)
        if bool(include_captions):
            headers["x-with-generated-alt"] = "true"
        if _env_bool("ADVANCED_VISIT_LINKS_SUMMARY", False):
            headers["x-with-links-summary"] = "true"
        if _env_bool("ADVANCED_VISIT_IMAGES_SUMMARY", False):
            headers["x-with-images-summary"] = "true"
        if _env_bool("ADVANCED_VISIT_NO_CACHE", False):
            headers["x-no-cache"] = "true"
        token_budget = options.get("token_budget") or _first_env("ADVANCED_VISIT_TOKEN_BUDGET", default="")
        if token_budget:
            headers["x-token-budget"] = str(token_budget)
        timeout_sec = options.get("timeout_sec") or _first_env("ADVANCED_VISIT_READER_TIMEOUT_SEC", default="")
        if timeout_sec:
            headers["x-timeout"] = str(timeout_sec)
        return request_url, headers


@register_tool("deepsearch", allow_overwrite=True)
class DeepSearch(BaseTool):
    name = "deepsearch"
    description = "Run a high-depth research process that searches, reads, and reasons iteratively. Use only when normal search, visit, and scholar tools are insufficient or when a difficult question needs deeper evidence gathering."
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The complete research question.",
            },
            "goal": {
                "type": "string",
                "description": "Optional focus for the research process.",
            },
        },
        "required": ["question"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        del kwargs
        if not isinstance(params, dict):
            return "[deepsearch] Invalid request format: Input must be a JSON object containing 'question'."
        question = str(params.get("question") or "").strip()
        if not question:
            return "[deepsearch] Invalid request format: 'question' cannot be empty."
        goal = str(params.get("goal") or "").strip()
        user_text = question if not goal else f"{question}\n\nResearch goal: {goal}"
        api_url = _first_env("DEEPSEARCH_API_URL", "DEEPSEARCH_API_URLS", default="https://deepsearch.jina.ai/v1/chat/completions")
        payload = {
            "model": _first_env("DEEPSEARCH_MODEL", default="jina-deepsearch-v1"),
            "messages": [{"role": "user", "content": user_text}],
            "stream": False,
            "temperature": _env_float("DEEPSEARCH_TEMPERATURE", 0.2),
        }
        max_tokens = _env_int("DEEPSEARCH_MAX_TOKENS", 8192)
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        max_retries = max(1, _env_int("DEEPSEARCH_MAX_RETRIES", 2))
        retry_sleep = max(0.0, _env_float("DEEPSEARCH_RETRY_SLEEP_SEC", 2.0))
        last_error = ""
        for attempt in range(max_retries):
            started_at = time.time()
            try:
                response = tool_http_request(
                    prefix="DEEPSEARCH",
                    method="POST",
                    url=api_url,
                    headers=_jina_headers(),
                    json_payload=payload,
                    trace_layer="tool_deepsearch",
                    trace_event="deepsearch",
                    trace_fields={"api_url": api_url, "attempt": attempt + 1, "max_attempts": max_retries},
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    message = choices[0].get("message") or {}
                    content = str(message.get("content") or "").strip()
                    if content:
                        if is_text_blocked(content):
                            return blocked_visit_message()
                        return content
                return json.dumps(data, ensure_ascii=False)[: _env_int("DEEPSEARCH_RAW_MAX_CHARS", 12000)]
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                emit_trace(
                    "deepsearch_error",
                    layer="tool_deepsearch",
                    api_url=api_url,
                    attempt=attempt + 1,
                    max_attempts=max_retries,
                    duration_s=round(time.time() - started_at, 3),
                    error=last_error,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_sleep)
        return f"[deepsearch] Research process could not complete after configured retries. Last error: {last_error}"
