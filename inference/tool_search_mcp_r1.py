import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
from qwen_agent.tools.base import BaseTool, register_tool
import asyncio
from typing import Dict, List, Optional, Union
import uuid
import json
import time

import os
from urllib.parse import urlparse

from runtime_env import get_env_float, get_env_int
from tool_http_client import tool_http_config, tool_http_request
from sandbox_gateway import SandboxStateError
from tool_provider_strategy import provider_strategy, strategy_search
from tool_url_guard import (
    GUARD_FORMAT_SERPER_SEARCH_V1,
    blocked_search_message,
    capture_structured_basis,
    guard_capture_item_scope,
    is_search_result_blocked,
    is_url_blocked,
    partial_search_note,
    search_result_guard_reason,
    url_guard_enabled,
)

try:
    from science_agent.tracing import emit_trace
except Exception:  # pragma: no cover
    def emit_trace(event: str, *, layer: str, **fields):
        return None


SERPER_KEY = os.environ.get("SERPER_API_KEY") or os.environ.get("SERPER_KEY_ID")
SERPER_API_URL = os.environ.get("SERPER_API_URL", "https://google.serper.dev/search")
SERPER_TIMEOUT = int(os.environ.get("SERPER_TIMEOUT", 30))

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("DR_CACHE_DIR", os.path.join(PROJECT_DIR, ".cache", "dr"))
DR_CACHE_LOG = os.environ.get("DR_CACHE_LOG", "").strip().lower() in {"1", "true", "yes", "on"}
if DR_CACHE_LOG:
    os.makedirs(CACHE_DIR, exist_ok=True)

successful_calls_search = 0
failed_calls_search = 0

SEARCH_CACHE_FILE = os.path.join(CACHE_DIR, "search.jsonl")
SEARCH_DOMAIN_STATS_FILE = os.path.join(CACHE_DIR, "search_domain_stats.jsonl")

TRUE_VALUES = {"1", "true", "yes", "on"}


def _serper_max_retries() -> int:
    return max(1, get_env_int("SERPER_MAX_RETRIES", default=5))


def _serper_retry_sleep_sec() -> float:
    return max(0.0, get_env_float("SERPER_RETRY_SLEEP_SEC", "TOOL_HTTP_RETRY_SLEEP_SEC", default=1.0))


def write_cache_log(path: str, payload: dict, label: str) -> None:
    if not DR_CACHE_LOG:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            json_line = json.dumps(payload, ensure_ascii=False)
            f.write(json_line + "\n")
    except Exception as exc:
        print(f"[cache-log] Warning: failed to write {label} log to {path}: {exc}")


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def _search_domain_stats_enabled() -> bool:
    return _truthy_env("AGENT_EVAL_SEARCH_DOMAIN_STATS_LOG")


def _search_domain_stats_file() -> str:
    return os.environ.get("AGENT_EVAL_SEARCH_DOMAIN_STATS_FILE", "").strip() or SEARCH_DOMAIN_STATS_FILE


def _host_from_url(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized or normalized == "#":
        return ""

    parsed = urlparse(normalized)
    if not parsed.netloc and "://" not in normalized:
        parsed = urlparse("//" + normalized)
    host = (parsed.netloc or "").split("@")[-1].split(":")[0]
    if "." not in host:
        return ""
    return host


def _result_url_candidates(result: dict) -> list[str]:
    values: list[str] = []
    for key in ("link", "url", "sourceUrl", "source", "displayedLink"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value)
    return values


def _domain_status_from_result(result: dict) -> tuple[str, bool, str] | None:
    if not isinstance(result, dict):
        return None

    candidates = _result_url_candidates(result)
    values = [str(value or "").strip() for value in candidates if str(value or "").strip()]
    if not values:
        return None

    blocked_values = [value for value in values if is_url_blocked(value)]
    reason = search_result_guard_reason(result)
    representative = blocked_values[0] if blocked_values else values[0]
    host = _host_from_url(representative)
    if not host:
        return None
    return host, bool(reason), reason or "allowed"


def _search_domain_stats_payload(query: str, results: dict) -> dict:
    blocked_domains: Counter[str] = Counter()
    allowed_domains: Counter[str] = Counter()
    blocked_reasons: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    def record(source: str, result: dict) -> None:
        status = _domain_status_from_result(result)
        if not status:
            return
        host, blocked, reason = status
        source_counts[source] += 1
        if blocked:
            blocked_domains[host] += 1
            blocked_reasons[reason] += 1
        else:
            allowed_domains[host] += 1

    answer = results.get("answerBox")
    if isinstance(answer, dict):
        record("answerBox", answer)

    organic_results = results.get("organic")
    if isinstance(organic_results, list):
        for item in organic_results[:10]:
            if not isinstance(item, dict):
                continue
            record("organic", item)

    blocked_results = sum(blocked_domains.values())
    allowed_results = sum(allowed_domains.values())
    return {
        "query": query,
        "benchmark": os.environ.get("AGENT_EVAL_BENCHMARK", "").strip(),
        "guard_enabled": url_guard_enabled(),
        "total_results": blocked_results + allowed_results,
        "blocked_results": blocked_results,
        "allowed_results": allowed_results,
        "blocked_domains": dict(sorted(blocked_domains.items())),
        "allowed_domains": dict(sorted(allowed_domains.items())),
        "blocked_reasons": dict(sorted(blocked_reasons.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }


def write_search_domain_stats_log(query: str, results: dict) -> None:
    if not _search_domain_stats_enabled():
        return
    path = _search_domain_stats_file()
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = _search_domain_stats_payload(query, results)
        with open(path, "a", encoding="utf-8") as f:
            json_line = json.dumps(payload, ensure_ascii=False)
            f.write(json_line + "\n")
    except Exception as exc:
        print(f"[search-domain-stats] Warning: failed to write search domain stats to {path}: {exc}")


def _results_for_cache(results: dict) -> dict:
    if not url_guard_enabled() or not isinstance(results, dict):
        return results

    sanitized: dict = {}
    answer = results.get("answerBox")
    if isinstance(answer, dict) and not is_search_result_blocked(answer):
        sanitized["answerBox"] = answer

    organic = results.get("organic")
    if isinstance(organic, list):
        sanitized["organic"] = [
            item
            for item in organic
            if isinstance(item, dict) and not is_search_result_blocked(item)
        ]

    return sanitized


@register_tool("google_search-google_search", allow_overwrite=True)
class Search(BaseTool):
    name = "google_search-google_search"
    description = "Search Google for science/research questions when external facts are uncertain or candidate source URLs are needed; prefer precise queries and authoritative sources."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        
    def google_search_with_serp(self, query: str):
        global successful_calls_search, failed_calls_search, SEARCH_CACHE_FILE
        strategy = provider_strategy()
        if strategy != "original":
            return strategy_search(query)

        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)
        if contains_chinese_basic(query):
            payload = {
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn"
            }
            
        else:
            payload = {
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en"
            }
        
        headers = {"Content-Type": "application/json"}
        if SERPER_KEY:
            headers["X-API-KEY"] = SERPER_KEY
        
        max_attempts = _serper_max_retries()
        retry_sleep = _serper_retry_sleep_sec()
        for i in range(max_attempts):
            request_started_at = time.time()
            http_config = tool_http_config("SERPER")
            emit_trace(
                "serper_request_start",
                layer="tool_search",
                query=query,
                api_url=SERPER_API_URL,
                attempt=i + 1,
                max_attempts=max_attempts,
                backend=http_config["backend"],
                connect_timeout_s=http_config["connect_timeout"],
                read_timeout_s=http_config["read_timeout"],
                max_time_s=http_config["max_time"],
            )
            try:
                res = tool_http_request(
                    prefix="SERPER",
                    method="POST",
                    url=SERPER_API_URL,
                    headers=headers,
                    json_payload=payload,
                    trace_layer="tool_search",
                    trace_event="serper",
                    trace_fields={
                        "query": query,
                        "api_url": SERPER_API_URL,
                        "attempt": i + 1,
                        "max_attempts": max_attempts,
                    },
                )
                emit_trace(
                    "serper_request_end",
                    layer="tool_search",
                    query=query,
                    api_url=SERPER_API_URL,
                    attempt=i + 1,
                    max_attempts=max_attempts,
                    status_code=res.status_code,
                    backend=res.backend,
                    response_chars=len(res.text or ""),
                    duration_s=round(time.time() - request_started_at, 3),
                )
                res.raise_for_status()
                break
            except Exception as e:
                emit_trace(
                    "serper_request_error",
                    layer="tool_search",
                    query=query,
                    api_url=SERPER_API_URL,
                    attempt=i + 1,
                    max_attempts=max_attempts,
                    duration_s=round(time.time() - request_started_at, 3),
                    error_type=type(e).__name__,
                    error=str(e),
                )
                print(e)
                if isinstance(e, SandboxStateError):
                    failed_calls_search += 1
                    return f"Google search unavailable: {e}"
                if i == max_attempts - 1:
                    failed_calls_search += 1
                    if not SERPER_KEY and "google.serper.dev" in SERPER_API_URL:
                        return "Google search failed: missing Serper API key. Set SERPER_API_KEY."
                    return (
                        "Google search could not complete after configured retries. "
                        "Please try a more specific query or retry later."
                    )
                time.sleep(retry_sleep)
                continue
    
        results = res.json()
        capture_structured_basis(
            "search",
            results,
            request={"query": query},
            formatter=GUARD_FORMAT_SERPER_SEARCH_V1,
        )
        write_search_domain_stats_log(query, results)

        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            blocked_result_count = 0
            if "answerBox" in results:
                answer = results["answerBox"]
                if is_search_result_blocked(answer):
                    answer_box_info = ""
                    blocked_result_count += 1
                else:
                    title = answer.get("title", "Answer")
                    snippet = answer.get("snippet", "N/A").replace("\n", " ")
                    answer_box_info = f"## {title}\n{snippet}\n\n"
            else:
                answer_box_info = ""

            if "organic" in results:
                organic_info = ""
                max_results = max(1, int(os.environ.get("SCIENCE_AGENT_SEARCH_MAX_RESULTS", "10")))
                snippet_chars = max(0, int(os.environ.get("SCIENCE_AGENT_SEARCH_SNIPPET_CHARS", "0")))
                for i, item in enumerate(results["organic"][:max_results], 1):
                    title = item.get("title", "N/A")
                    link = item.get("link", "#")
                    if is_search_result_blocked(item):
                        blocked_result_count += 1
                        continue
                    snippet = item.get("snippet", "N/A").replace("\n", " " ).strip()
                    if snippet_chars > 0 and len(snippet) > snippet_chars:
                        snippet = snippet[:snippet_chars].rsplit(" ", 1)[0] + "..."
                    organic_info += f"{i}. **[{title}]({link})**\n   - {snippet}\n"
                # for page in results["organic"]:
                #     idx += 1
                #     date_published = ""
                #     if "date" in page:
                #         date_published = "\nDate published: " + page["date"]

                #     source = ""
                #     if "source" in page:
                #         source = "\nSource: " + page["source"]

                #     snippet = ""
                #     if "snippet" in page:
                #         snippet = "\n" + page["snippet"]

                #     redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                #     redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                #     web_snippets.append(redacted_version)

            if blocked_result_count and not answer_box_info and not organic_info:
                successful_calls_search += 1
                write_cache_log(
                    SEARCH_CACHE_FILE,
                    {"query": query, "result": _results_for_cache(results)},
                    "search",
                )
                return blocked_search_message(query=query)

            content = f"# Search Results for: '{query}'\n\n {answer_box_info} ## Web Results\n\n" + organic_info
            if blocked_result_count:
                content += "\n" + partial_search_note()
            successful_calls_search += 1
            write_cache_log(
                SEARCH_CACHE_FILE,
                {"query": query, "result": _results_for_cache(results)},
                "search",
            )
            return content
        except:
            failed_calls_search += 1
            return f"No results found for '{query}'. Try with a more general query."


    
    def search_with_serp(self, query: str):
        result = self.google_search_with_serp(query)
        return result

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            # 单个查询
            with guard_capture_item_scope(0):
                response = self.search_with_serp(query)
        else:
            # 多个查询
            assert isinstance(query, List)
            responses = []
            for index, q in enumerate(query):
                with guard_capture_item_scope(index):
                    responses.append(self.search_with_serp(q))
            response = "\n=======\n".join(responses)
            
        return response


if __name__ == "__main__":
    tool = Search()
    params = {
        "query": "Find information about a joke removed from Wikipedia Dragon page on leap day before 2008"
    }
    result = tool.call(params)
    print(result)
