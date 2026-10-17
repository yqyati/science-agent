import os
import json
import time
from typing import Union, List
from qwen_agent.tools.base import BaseTool, register_tool
from concurrent.futures import ThreadPoolExecutor
from runtime_env import get_env_float, get_env_int
from tool_http_client import tool_http_config, tool_http_request
from sandbox_gateway import SandboxStateError
from tool_provider_strategy import provider_strategy, strategy_scholar
from tool_url_guard import (
    GUARD_FORMAT_SERPER_SCHOLAR_V1,
    bind_guard_capture_context,
    blocked_search_message,
    capture_structured_basis,
    guard_capture_item_scope,
    is_search_result_blocked,
    partial_search_note,
)

try:
    from science_agent.tracing import emit_trace
except Exception:  # pragma: no cover
    def emit_trace(event: str, *, layer: str, **fields):
        return None


SERPER_KEY = os.environ.get("SERPER_API_KEY") or os.environ.get("SERPER_KEY_ID")
SERPER_SCHOLAR_API_URL = os.environ.get(
    "SERPER_SCHOLAR_API_URL",
    "https://google.serper.dev/scholar",
)

try:
    SERPER_TIMEOUT = int(os.environ.get("SERPER_TIMEOUT", "30"))
except ValueError:
    SERPER_TIMEOUT = 30

try:
    SERPER_MAX_RETRIES = int(os.environ.get("SERPER_MAX_RETRIES", "5"))
except ValueError:
    SERPER_MAX_RETRIES = 5


def _serper_max_retries() -> int:
    return max(1, get_env_int("SERPER_MAX_RETRIES", default=SERPER_MAX_RETRIES))


def _serper_retry_sleep_sec() -> float:
    return max(0.0, get_env_float("SERPER_RETRY_SLEEP_SEC", "TOOL_HTTP_RETRY_SLEEP_SEC", default=1.0))


@register_tool("google_scholar", allow_overwrite=True)
class Scholar(BaseTool):
    name = "google_scholar"
    description = "Search Google Scholar when peer-reviewed papers or scholarly evidence are more appropriate than general web results for science/research questions."
    parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string", "description": "The search query."},
                    "minItems": 1,
                    "description": "The list of search queries for Google Scholar."
                },
            },
        "required": ["query"],
    }

    def google_scholar_with_serp(self, query: str):
        strategy = provider_strategy()
        if strategy != "original":
            return strategy_scholar(query)

        payload = {
            "q": query,
        }
        headers = {
            "Content-Type": "application/json",
        }
        if SERPER_KEY:
            headers["X-API-KEY"] = SERPER_KEY

        last_error = None
        max_retries = _serper_max_retries()
        retry_sleep = _serper_retry_sleep_sec()
        for i in range(max_retries):
            request_started_at = time.time()
            http_config = tool_http_config("SERPER")
            emit_trace(
                "serper_scholar_request_start",
                layer="tool_scholar",
                query=query,
                api_url=SERPER_SCHOLAR_API_URL,
                attempt=i + 1,
                max_attempts=max_retries,
                backend=http_config["backend"],
                connect_timeout_s=http_config["connect_timeout"],
                read_timeout_s=http_config["read_timeout"],
                max_time_s=http_config["max_time"],
            )
            try:
                res = tool_http_request(
                    prefix="SERPER",
                    method="POST",
                    url=SERPER_SCHOLAR_API_URL,
                    headers=headers,
                    json_payload=payload,
                    trace_layer="tool_scholar",
                    trace_event="serper_scholar",
                    trace_fields={
                        "query": query,
                        "api_url": SERPER_SCHOLAR_API_URL,
                        "attempt": i + 1,
                        "max_attempts": max_retries,
                    },
                )
                emit_trace(
                    "serper_scholar_request_end",
                    layer="tool_scholar",
                    query=query,
                    api_url=SERPER_SCHOLAR_API_URL,
                    attempt=i + 1,
                    max_attempts=max_retries,
                    status_code=res.status_code,
                    backend=res.backend,
                    response_chars=len(res.text or ""),
                    duration_s=round(time.time() - request_started_at, 3),
                )
                res.raise_for_status()
                break
            except Exception as e:
                last_error = e
                emit_trace(
                    "serper_scholar_request_error",
                    layer="tool_scholar",
                    query=query,
                    api_url=SERPER_SCHOLAR_API_URL,
                    attempt=i + 1,
                    max_attempts=max_retries,
                    duration_s=round(time.time() - request_started_at, 3),
                    error_type=type(e).__name__,
                    error=str(e),
                )
                print(e)
                if isinstance(e, SandboxStateError):
                    return f"Google Scholar unavailable: {e}"
                if i == max_retries - 1:
                    if not SERPER_KEY and "google.serper.dev" in SERPER_SCHOLAR_API_URL:
                        return (
                            "Google Scholar failed: missing Serper API key. "
                            "Set SERPER_API_KEY."
                        )
                    if hasattr(e, "response") and e.response is not None:
                        return (
                            f"Google Scholar HTTP {e.response.status_code}: "
                            f"{e.response.text[:300]}"
                        )
                    return (
                        "Google Scholar could not complete after configured retries. "
                        f"Please try again later. Last error: {last_error}"
                    )
                time.sleep(retry_sleep)
                continue

        try:
            results = res.json()
        except (json.JSONDecodeError, ValueError):
            return f"Google Scholar returned invalid JSON: {res.text[:300]}"

        capture_structured_basis(
            "google_scholar",
            results,
            request={"query": query},
            formatter=GUARD_FORMAT_SERPER_SCHOLAR_V1,
        )

        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            blocked_result_count = 0
            if "organic" in results:
                max_results = max(0, int(os.environ.get("SCIENCE_AGENT_SCHOLAR_MAX_RESULTS", "0")))
                snippet_chars = max(0, int(os.environ.get("SCIENCE_AGENT_SCHOLAR_SNIPPET_CHARS", "0")))
                for page in results["organic"][:max_results or None]:
                    if is_search_result_blocked(page):
                        blocked_result_count += 1
                        continue

                    idx = len(web_snippets) + 1
                    title = page.get("title", "Untitled")

                    date_published = ""
                    if "year" in page:
                        date_published = "\nDate published: " + str(page["year"])

                    publicationInfo = ""
                    if "publicationInfo" in page:
                        publicationInfo = "\npublicationInfo: " + str(page["publicationInfo"])

                    snippet = ""
                    if "snippet" in page:
                        snippet_text = str(page["snippet"]).strip()
                        if snippet_chars > 0 and len(snippet_text) > snippet_chars:
                            snippet_text = snippet_text[:snippet_chars].rsplit(" ", 1)[0] + "..."
                        snippet = "\n" + snippet_text
                    
                    link_info = "no available link"
                    if "link" in page:
                        link_info = page["link"]
                    elif "pdfUrl" in page:
                        link_info = page["pdfUrl"]
                    
                    citedBy = ""
                    if "citedBy" in page:
                        citedBy = "\ncitedBy: " + str(page["citedBy"])
                    
                    redacted_version = f"{idx}. [{title}]({link_info}){publicationInfo}{date_published}{citedBy}\n{snippet}"

                    redacted_version = redacted_version.replace("Your browser can't play this video.", "") 
                    web_snippets.append(redacted_version)

            if blocked_result_count and not web_snippets:
                return blocked_search_message()

            content = f"A Google scholar for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)
            if blocked_result_count:
                content += "\n" + partial_search_note()
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."


    def call(self, params: Union[str, dict], **kwargs) -> str:
        # assert GOOGLE_SEARCH_KEY is not None, "Please set the IDEALAB_SEARCH_KEY environment variable."
        try:
            params = self._verify_json_format_args(params)
            query = params["query"]
        except:
            return "[google_scholar] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            with guard_capture_item_scope(0):
                response = self.google_scholar_with_serp(query)
        else:
            assert isinstance(query, List)
            callbacks = []
            for index, item in enumerate(query):
                def invoke(index=index, item=item):
                    with guard_capture_item_scope(index):
                        return self.google_scholar_with_serp(item)

                callbacks.append(bind_guard_capture_context(invoke))
            with ThreadPoolExecutor(max_workers=3) as executor:
                response = list(executor.map(lambda callback: callback(), callbacks))
            response = "\n=======\n".join(response)
        return response
