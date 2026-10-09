import json
import os
import re
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
from qwen_agent.tools.base import BaseTool, register_tool
from prompt import EXTRACTOR_PROMPT 
from openai import OpenAI
import random
from urllib.parse import urlparse, unquote
import time 
import httpx
from runtime_env import get_env_float, get_env_int
from tool_http_client import tool_http_config, tool_http_request
from sandbox_gateway import shared_sandbox_run
from tool_provider_strategy import provider_strategy, strategy_visit
from tool_url_guard import (
    GUARD_FORMAT_VISIT_FINAL_V1,
    blocked_visit_message,
    capture_structured_basis,
    guard_capture_item_scope,
    guard_capture_url_scope,
    is_text_blocked,
    is_url_blocked,
    is_value_blocked,
    current_guard_context, capture_guard_context, guard_fingerprint,
)

try:
    from science_agent.tracing import emit_trace
except Exception:  # pragma: no cover
    def emit_trace(event: str, *, layer: str, **fields):
        return None


VISIT_SERVER_TIMEOUT = get_env_int("VISIT_SERVER_TIMEOUT", default=200)
VISIT_READ_MAX_RETRIES = max(
    1,
    get_env_int("VISIT_READ_MAX_RETRIES", default=3),
)
VISIT_RETRY_SLEEP_SEC = max(
    0.0,
    get_env_float("VISIT_RETRY_SLEEP_SEC", default=0.5),
)
SUMMARY_REQUEST_TIMEOUT_SEC = get_env_float(
    "SUMMARY_REQUEST_TIMEOUT_SEC",
    "OPENAI_REQUEST_TIMEOUT",
    "LLM_REQUEST_TIMEOUT_SEC",
    default=600.0,
)
SUMMARY_RETRY_SLEEP_SEC = max(
    0.0,
    get_env_float("SUMMARY_RETRY_SLEEP_SEC", default=10.0),
)
VISIT_SERVER_MAX_RETRIES = max(
    1,
    get_env_int("VISIT_SERVER_MAX_RETRIES", default=1),
)
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("DR_CACHE_DIR", os.path.join(PROJECT_DIR, ".cache", "dr"))
DR_CACHE_LOG = os.environ.get("DR_CACHE_LOG", "").strip().lower() in {"1", "true", "yes", "on"}

JINA_API_KEYS = [
    key.strip()
    for key in (os.getenv("JINA_API_KEYS") or os.getenv("JINA_API_KEY", "")).split(",")
    if key.strip()
]
JINA_READER_URL_TEMPLATE = os.getenv(
    "JINA_READER_URL_TEMPLATE",
    "https://r.jina.ai/{full_url}",
)


successful_calls_visit = 0
failed_calls_visit = 0
successful_calls_summarize = 0
failed_calls_summarize = 0

VISIT_CACHE_FILE = os.path.join(CACHE_DIR, "visit.jsonl")
SUMMARY_CACHE_FILE = os.path.join(CACHE_DIR, "summary.jsonl")


def _first_env_pool(name: str) -> str:
    raw_value = os.environ.get(name, "")
    for part in str(raw_value).split(","):
        value = part.strip()
        if value:
            return value
    return ""


def _jina_reader_url_template() -> str:
    return (
        os.environ.get("JINA_READER_URL_TEMPLATE", "").strip()
        or _first_env_pool("JINA_READER_URL_TEMPLATES")
        or JINA_READER_URL_TEMPLATE
        or "https://r.jina.ai/{full_url}"
    )


def _visit_read_max_retries() -> int:
    return max(1, get_env_int("VISIT_READ_MAX_RETRIES", default=VISIT_READ_MAX_RETRIES))


def _visit_retry_sleep_sec() -> float:
    return max(0.0, get_env_float("VISIT_RETRY_SLEEP_SEC", "JINA_RETRY_SLEEP_SEC", "TOOL_HTTP_RETRY_SLEEP_SEC", default=VISIT_RETRY_SLEEP_SEC))


def _visit_html_read_max_attempts() -> int:
    return max(1, get_env_int("VISIT_HTML_READ_MAX_ATTEMPTS", default=1))


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


def _is_blocked_visit_response(value: object) -> bool:
    return isinstance(value, str) and value == blocked_visit_message()


def _evidence_units(text: str, max_units: int) -> str:
    """Keep complete evidence sentences grouped by paragraph."""
    paragraphs = [part.strip() for part in str(text or "").split("\n\n") if part.strip()]
    selected = []
    for paragraph in paragraphs[:max_units]:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        selected.append(" ".join(sentences[:3]).strip())
    return "\n\n".join(selected)


def _structured_visit_return(url: str, goal: str, evidence: str, summary: str) -> str:
    max_evidence = max(1, int(os.environ.get("SCIENCE_AGENT_READ_PAGE_MAX_EVIDENCE_BLOCKS", "4")))
    evidence = _evidence_units(evidence, max_evidence)
    return (
        f"The useful information in {url} for user goal {goal} as follows:\n\n"
        f"Evidence in page:\n{evidence}\n\nSummary:\n{summary}\n"
    )


def _clean_page_paragraphs(content: str) -> list[str]:
    """Split cleaned page text into complete, non-boilerplate paragraphs."""
    content = re.sub(r"\r\n?", "\n", str(content or ""))
    content = re.sub(r"[ \t]+", " ", content)
    paragraphs = []
    for raw in re.split(r"\n\s*\n+", content):
        text = re.sub(r"\s*\n\s*", " ", raw).strip()
        if len(text) < 40:
            continue
        lowered = text.lower()
        if lowered.count("|") >= 3 and len(text) < 500:
            continue
        if re.fullmatch(r"(?:home|menu|search|log in|sign in|subscribe|contact|about)[\s|·-]*", lowered):
            continue
        paragraphs.append(text)
    return paragraphs


def _extractive_page(content: str, goal: str, *, max_chars: int) -> str:
    """Return the full page below budget, otherwise select relevant whole paragraphs."""
    paragraphs = _clean_page_paragraphs(content)
    cleaned = "\n\n".join(paragraphs)
    if len(cleaned) <= max_chars:
        return cleaned

    terms = [t.lower() for t in re.findall(r"[\w\u4e00-\u9fff]{2,}", str(goal or ""))]
    terms = list(dict.fromkeys(terms))
    phrases = [p.lower().strip() for p in re.findall(r"\"([^\"]+)\"", str(goal or "")) if p.strip()]
    scored = []
    for index, paragraph in enumerate(paragraphs):
        lower = paragraph.lower()
        term_hits = sum(lower.count(term) for term in terms)
        phrase_hits = sum(1 for phrase in phrases if phrase in lower)
        number_hits = sum(1 for number in re.findall(r"\b\d+(?:\.\d+)?%?\b", str(goal or "")) if number in paragraph)
        heading_bonus = 2 if paragraph.startswith(("#", "##", "###")) else 0
        boilerplate_penalty = 4 if any(x in lower for x in ("privacy policy", "cookie settings", "all rights reserved")) else 0
        score = term_hits + phrase_hits * 4 + number_hits * 2 + heading_bonus - boilerplate_penalty
        scored.append((score, -index, index, paragraph))

    selected = []
    used = 0
    # Prefer relevant paragraphs, but retain the page title/opening context when possible.
    for score, _, index, paragraph in sorted(scored, reverse=True):
        if score <= 0 and selected:
            continue
        extra = len(paragraph) + (2 if selected else 0)
        if used + extra > max_chars:
            continue
        selected.append((index, paragraph))
        used += extra

    if not selected:
        first_sentences = re.split(r"(?<=[.!?])\s+", paragraphs[0])
        fallback = []
        used = 0
        for sentence in first_sentences:
            extra = len(sentence) + (1 if fallback else 0)
            if used + extra > max_chars:
                break
            fallback.append(sentence)
            used += extra
        selected = [(0, " ".join(fallback))] if fallback else [(0, paragraphs[0])]
    selected.sort(key=lambda item: item[0])
    return "\n\n".join(paragraph for _, paragraph in selected)


def _extractive_visit_return(url: str, goal: str, content: str) -> str:
    max_chars = max(1000, int(os.environ.get("SCIENCE_AGENT_READ_PAGE_MAX_CHARS", "8000")))
    evidence = _extractive_page(content, goal, max_chars=max_chars)
    return (
        f"Source: {url}\n\n"
        f"Extractive evidence selected for goal: {goal}\n\n"
        f"{evidence}\n\n"
        "Note: no secondary LLM summary was generated; the evidence above is selected from the page text."
    )


def normalize_url_input(url: Union[str, List[str]]) -> Union[str, List[str]]:
    def normalize_single_url(single_url: str) -> str:
        single_url = single_url.strip()
        if single_url.startswith('["') and single_url.endswith('"]'):
            return single_url[2:-2]
        return single_url

    if isinstance(url, str):
        return normalize_single_url(url)

    if isinstance(url, list):
        return [
            normalize_single_url(item) if isinstance(item, str) else item
            for item in url
        ]

    return url

OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


@register_tool('read-page-server-read_page', allow_overwrite=True)
class Visit(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    name = 'read-page-server-read_page'
    description = (
        "Fetch a known webpage URL after search or when a URL is given. In Sandbox file mode, "
        "save the complete reader-provided Markdown in this Agent's /workspace/pages directory. "
        "Return the file path, source URL, character/line counts, and a short preview; "
        "the response is not the full page and no LLM summary is generated. "
        "Next use the Python tool to read the returned relative path from /workspace with UTF-8. "
        "Search for relevant terms or inspect line ranges, and print original evidence with line numbers "
        "and surrounding context. Do not assume information absent from the preview is absent from the page. "
        "Keep each Python output short; if it reports truncation, read smaller ranges to recover omitted evidence. "
        "Repeated visits reuse the saved page within this Agent run."
    )
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {
                    "type": "string"
                    },
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
        },
        "goal": {
                "type": "string",
                "description": "The research question or evidence to look for. In file mode the page is saved in full; use Python afterward to locate evidence for this goal."
        }
        },
        "required": ["url", "goal"]
    }
    # The `call` method is the main function of the tool.
    def call(self, params: Union[str, dict], **kwargs) -> str:

        try:
            url = params["url"]
            goal = params["goal"]
            url = normalize_url_input(url)
        except:
            return "[read_page] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        start_time = time.time()
        
        # Create log folder if it doesn't exist
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)

        if isinstance(url, str):
            with guard_capture_item_scope(0), guard_capture_url_scope(url, goal=goal):
                if is_url_blocked(url):
                    response = blocked_visit_message()
                else:
                    strategy = provider_strategy()
                    if strategy == "original":
                        response = self.readpage_jina(url, goal)
                    else:
                        response = strategy_visit(url, goal, original_visit=self)
                capture_structured_basis(
                    "visit",
                    {"content": response},
                    request={"url": url, "goal": goal},
                    stage="final",
                    formatter=GUARD_FORMAT_VISIT_FINAL_V1,
                )
        else:
            response = []
            assert isinstance(url, List)
            start_time = time.time()
            for index, u in enumerate(url):
                with guard_capture_item_scope(index), guard_capture_url_scope(u, goal=goal):
                    if is_url_blocked(u):
                        cur_response = blocked_visit_message()
                    elif time.time() - start_time > 900:
                        cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                        cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                        cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                    else:
                        try:
                            strategy = provider_strategy()
                            if strategy == "original":
                                cur_response = self.readpage_jina(u, goal)
                            else:
                                cur_response = strategy_visit(u, goal, original_visit=self)
                        except Exception as e:
                            cur_response = f"Error fetching {u}: {str(e)}"
                    capture_structured_basis(
                        "visit",
                        {"content": cur_response},
                        request={"url": u, "goal": goal},
                        stage="final",
                        formatter=GUARD_FORMAT_VISIT_FINAL_V1,
                    )
                    response.append(cur_response)
            response = "\n=======\n".join(response)
        
        print(f'Read page response chars: {len(response)}')
        return response.strip()
        
    def call_server(self, msgs, max_retries=2):
        global successful_calls_summarize, failed_calls_summarize, SUMMARY_CACHE_FILE
        api_key = os.environ.get("API_KEY")
        url_llm = os.environ.get("API_BASE")
        model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
        reasoning_effort = str(os.environ.get("SUMMARY_REASONING_EFFORT") or "").strip().lower()
        if reasoning_effort not in {"low", "medium", "high"}:
            reasoning_effort = ""
        sandbox_http = os.getenv("SCIENCE_AGENT_TOOL_HTTP_BACKEND", "").strip().lower() == "gateway_sandbox"
        if sandbox_http and not all((api_key, url_llm, model_name)):
            raise ValueError("Sandbox readpage requires API_BASE, API_KEY and SUMMARY_MODEL_NAME in sandbox.local.env")
        client = None if sandbox_http else OpenAI(
            api_key=api_key,
            base_url=url_llm,
            timeout=SUMMARY_REQUEST_TIMEOUT_SEC,
            http_client=httpx.Client(verify=False)
        )
        for attempt in range(max_retries):
            request_started_at = time.time()
            emit_trace(
                "summary_request_start",
                layer="tool_visit",
                api_base=url_llm,
                model=model_name,
                attempt=attempt + 1,
                max_attempts=max_retries,
                timeout_s=SUMMARY_REQUEST_TIMEOUT_SEC,
                reasoning_effort=reasoning_effort,
                prompt_chars=sum(len(str(msg.get("content", ""))) for msg in msgs if isinstance(msg, dict)),
            )
            try:
                request_kwargs = {
                    "model": model_name,
                    "messages": msgs,
                    "temperature": 1.0,
                }
                if reasoning_effort:
                    request_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
                if sandbox_http:
                    payload = dict(request_kwargs)
                    payload.update(payload.pop("extra_body", {}))
                    response = tool_http_request(
                        prefix="SUMMARY", method="POST",
                        url=url_llm.rstrip("/") + "/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json_payload=payload,
                        trace_layer="tool_visit", trace_event="summary_request",
                    )
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"].get("content")
                else:
                    chat_response = client.chat.completions.create(**request_kwargs)
                    content = chat_response.choices[0].message.content
                emit_trace(
                    "summary_request_end",
                    layer="tool_visit",
                    api_base=url_llm,
                    model=model_name,
                    attempt=attempt + 1,
                    max_attempts=max_retries,
                    reasoning_effort=reasoning_effort,
                    content_chars=len(str(content or "")),
                    duration_s=round(time.time() - request_started_at, 3),
                )
                if content:
                    if is_text_blocked(content, stage="summary"):
                        return blocked_visit_message()
                    try:
                        json.loads(content)
                    except:
                        # extract json from string 
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    successful_calls_summarize += 1
                    write_cache_log(
                        SUMMARY_CACHE_FILE,
                        {"messages": msgs, "summary": content},
                        "summary",
                    )
                    return content
            except Exception as e:
                emit_trace(
                    "summary_request_error",
                    layer="tool_visit",
                    api_base=url_llm,
                    model=model_name,
                    attempt=attempt + 1,
                    max_attempts=max_retries,
                    duration_s=round(time.time() - request_started_at, 3),
                    error_type=type(e).__name__,
                    error=str(e),
                )
                time.sleep(SUMMARY_RETRY_SLEEP_SEC)
                print(e)
                if attempt == (max_retries - 1):
                    failed_calls_summarize += 1
                    return ""
                continue

    def _build_jina_request(self, url: str):
        full_url = url.strip()
        url_without_scheme = full_url.removeprefix("https://").removeprefix("http://")
        template = _jina_reader_url_template()
        request_url = template.format(
            url=url_without_scheme,
            full_url=full_url,
        )
        headers = {}
        if JINA_API_KEYS:
            headers["Authorization"] = f"Bearer {random.choice(JINA_API_KEYS)}"
        return request_url, headers


    def jina_readpage(self, url: str) -> str:
        global successful_calls_visit, failed_calls_visit, VISIT_CACHE_FILE
        """
        Read webpage content using Jina service.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        max_retries = _visit_read_max_retries()
        retry_sleep = _visit_retry_sleep_sec()
        
        for attempt in range(max_retries):
            request_started_at = time.time()
            try:
                request_url, headers = self._build_jina_request(url)
                http_config = tool_http_config("JINA")
                emit_trace(
                    "jina_request_start",
                    layer="tool_visit",
                    url=url,
                    request_url=request_url,
                    attempt=attempt + 1,
                    max_attempts=max_retries,
                    backend=http_config["backend"],
                    connect_timeout_s=http_config["connect_timeout"],
                    read_timeout_s=http_config["read_timeout"],
                    max_time_s=http_config["max_time"],
                )
                response = tool_http_request(
                    prefix="JINA",
                    method="GET",
                    url=request_url,
                    headers=headers,
                    trace_layer="tool_visit",
                    trace_event="jina",
                    trace_fields={
                        "url": url,
                        "request_url": request_url,
                        "attempt": attempt + 1,
                        "max_attempts": max_retries,
                    },
                )
                emit_trace(
                    "jina_request_end",
                    layer="tool_visit",
                    url=url,
                    request_url=request_url,
                    attempt=attempt + 1,
                    max_attempts=max_retries,
                    status_code=response.status_code,
                    backend=response.backend,
                    response_chars=len(response.text or ""),
                    duration_s=round(time.time() - request_started_at, 3),
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    if is_text_blocked(webpage_content, stage="page"):
                        return blocked_visit_message()
                    successful_calls_visit += 1
                    write_cache_log(
                        VISIT_CACHE_FILE,
                        {"url": url, "content": webpage_content},
                        "visit",
                    )
                    return webpage_content
                else:
                    print(f"Jina API Error: Status {response.status_code}, Response: {response.text}")
                    if attempt == max_retries - 1:
                        failed_calls_visit += 1
                        return f"[read_page] Failed to read page: HTTP {response.status_code}"
            except Exception as e:
                emit_trace(
                    "jina_request_error",
                    layer="tool_visit",
                    url=url,
                    attempt=attempt + 1,
                    max_attempts=max_retries,
                    duration_s=round(time.time() - request_started_at, 3),
                    error_type=type(e).__name__,
                    error=str(e),
                )
                time.sleep(retry_sleep)
                print(f"Jina API Request Error (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    failed_calls_visit += 1
                    return f"[read_page] Failed to read page: {str(e)}"
                
        return "[read_page] Failed to read page after all retries"

    def html_readpage_jina(self, url: str) -> str:
        max_attempts = _visit_html_read_max_attempts()
        for attempt in range(max_attempts):
            content = self.jina_readpage(url)
            service = "jina"     
            print(service)
            if _is_blocked_visit_response(content):
                return content
            if content and not content.startswith("[read_page] Failed") and content != "[read_page] Empty content." and not content.startswith("[document_parser]"):
                return content
        return "[read_page] Failed to read page."

    def readpage_jina(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
   
        if os.getenv("SCIENCE_AGENT_READ_PAGE_MODE", "").strip().lower() == "file":
            if os.getenv("SCIENCE_AGENT_TOOL_HTTP_BACKEND") != "gateway_sandbox":
                return "[read_page] File mode requires gateway_sandbox"
            context = current_guard_context() or capture_guard_context()
            request_url, headers = self._build_jina_request(url)
            headers["X-Return-Format"] = "markdown"
            for attempt in range(_visit_read_max_retries()):
                try:
                    result = shared_sandbox_run().run_http({
                        "method": "GET", "url": request_url, "headers": headers,
                        "save_page": True, "source_url": url,
                        "guard_snapshot": context.snapshot.to_dict(),
                        "guard_fingerprint": guard_fingerprint(context),
                    }, timeout=tool_http_config("JINA")["max_time"])
                    if result["status_code"] == 200:
                        page = result["page"]
                        if page.get("blocked"):
                            return blocked_visit_message()
                        emit_trace("read_page_saved", layer="tool_visit", url=url,
                                   path=page["path"], page_chars=page["characters"], cached=page.get("cached", False))
                        return json.dumps(dict(page, instruction="This is a file description and preview, not the full page. Use Python to read the relative path from /workspace with UTF-8. Print relevant original lines with line numbers and context. If output is truncated, read smaller line ranges."), ensure_ascii=False)
                    error = f"HTTP {result['status_code']}: {result.get('text', '')[:500]}"
                except Exception as exc:
                    error = str(exc)
                if attempt + 1 < _visit_read_max_retries():
                    time.sleep(_visit_retry_sleep_sec())
            return "[read_page] Failed to save page: " + error

        pipeline_started_at = time.time()
        emit_trace(
            "read_page_pipeline_start",
            layer="tool_visit",
            url=url,
            goal_chars=len(str(goal or "")),
        )
        summary_page_func = self.call_server
        max_retries = VISIT_SERVER_MAX_RETRIES

        try:
            content = self.html_readpage_jina(url)

            if _is_blocked_visit_response(content) or is_text_blocked(content, stage="page"):
                return blocked_visit_message()

            if content and not content.startswith("[read_page] Failed") and content != "[read_page] Empty content." and not content.startswith("[document_parser]"):
                if os.getenv("SCIENCE_AGENT_READ_PAGE_MODE", "").strip().lower() == "extractive":
                    return _extractive_visit_return(url, goal, content)
                content = content[:WEBCONTENT_MAXLENGTH]
                messages = [{"role":"user","content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
                parse_retry_times = 0
                raw = summary_page_func(messages, max_retries=max_retries)
                if _is_blocked_visit_response(raw) or is_text_blocked(raw, stage="summary"):
                    return blocked_visit_message()
                summary_retries = 3
                while len(raw) < 10 and summary_retries >= 0:
                    truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                    status_msg = (
                        f"[read_page] Summary url[{url}] " 
                        f"attempt {3 - summary_retries + 1}/3, "
                        f"content length: {len(content)}, "
                        f"truncating to {truncate_length} chars"
                    ) if summary_retries > 0 else (
                        f"[read_page] Summary url[{url}] failed after 3 attempts, "
                        f"final truncation to 25000 chars"
                    )
                    print(status_msg)
                    content = content[:truncate_length]
                    extraction_prompt = EXTRACTOR_PROMPT.format(
                        webpage_content=content,
                        goal=goal
                    )
                    messages = [{"role": "user", "content": extraction_prompt}]
                    raw = summary_page_func(messages, max_retries=max_retries)
                    if _is_blocked_visit_response(raw) or is_text_blocked(raw, stage="summary"):
                        return blocked_visit_message()
                    summary_retries -= 1

                parse_retry_times = 2
                if isinstance(raw, str):
                    raw = raw.replace("```json", "").replace("```", "").strip()
                while parse_retry_times < 3:
                    try:
                        raw = json.loads(raw)
                        break
                    except:
                        raw = summary_page_func(messages, max_retries=max_retries)
                        if _is_blocked_visit_response(raw) or is_text_blocked(raw, stage="summary"):
                            return blocked_visit_message()
                        parse_retry_times += 1
                
                if parse_retry_times >= 3:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    if is_value_blocked(raw, stage="summary"):
                        return blocked_visit_message()
                    useful_information = _structured_visit_return(
                        url, goal, str(raw["evidence"]), str(raw["summary"])
                    )

                if len(useful_information) < 10 and summary_retries < 0:
                    print("[read_page] Could not generate valid summary after maximum retries")
                    useful_information = "[read_page] Failed to read page"

                if is_text_blocked(useful_information, stage="final"):
                    return blocked_visit_message()
                
                return useful_information

            # If no valid content was obtained after all retries
            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                return useful_information
        except Exception as exc:
            emit_trace(
                "read_page_pipeline_error",
                layer="tool_visit",
                url=url,
                duration_s=round(time.time() - pipeline_started_at, 3),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            emit_trace(
                "read_page_pipeline_end",
                layer="tool_visit",
                url=url,
                duration_s=round(time.time() - pipeline_started_at, 3),
            )


if __name__ == "__main__":
    tool = Visit()
    params = {
        "url": "https://learningforward.org/journal/the-time-dilemma/the-role-of-collaborative-learning-in-teacher-retention/",
        "goal": "Find the author of the article and verify if they received an award for remedial education in the publication year."
    }
    result = tool.call(params)
    print(result)
    
