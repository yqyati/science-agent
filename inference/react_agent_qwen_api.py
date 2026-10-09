import copy
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import httpx
from sandbox_gateway import sandbox_run_scope
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import Message
from qwen_agent.tools import BaseTool

from prompt import TOOLS_DEEPSEEK_V32
from multimodal_input import (
    build_initial_user_message,
    build_runtime_multimodal_support_error,
)
from runtime_env import get_env_bool, get_env_float, get_env_int
from deepseek_v4_config import (
    deepseek_v4_reasoning_opt_in_enabled,
    deepseek_v4_request_kwargs,
    extract_reasoning_text,
    is_deepseek_v4_model,
    split_deepseek_v4_response_text,
)
from science_agent.tracing import emit_trace
from science_agent.tool_call_parsing import extract_text_tool_calls
from tool_search_mcp_r1 import Search
from tool_visit_mcp_r1 import Visit
from tool_runtime_registry import compat_tool_schemas_for_base_url

try:
    import json5 as json_parser
except ImportError:
    json_parser = json

OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"

CONTEXT_MANAGEMENT_STRATEGY = "none"
MAX_LLM_CALL_PER_RUN = get_env_int("MAX_LLM_CALL_PER_RUN", default=100)
MAX_TOOL_CALLS_PER_RUN = max(
    0,
    get_env_int("MAX_TOOL_CALLS_PER_RUN", default=0),
)
TRAJECTORY_MAX_TOKENS = max(
    1,
    get_env_int("SCIENCE_TRAJECTORY_MAX_TOKENS", default=32768),
)
# A length-truncated completion is deterministic for the same request and
# retrying it wastes the rollout budget. Network/API failures still use the
# normal retry path below. Set this to 1 only for compatibility experiments.
RETRY_INCOMPLETE_OUTPUT = get_env_bool(
    "LLM_RETRY_INCOMPLETE_OUTPUT",
    default=False,
)
TASK_TIMEOUT_SEC = get_env_float("TASK_TIMEOUT_SEC", default=9000.0)
AGENT_EVAL_FORCE_FINALIZE = os.getenv("AGENT_EVAL_FORCE_FINALIZE", "0").lower() in {
    "1",
    "true",
    "yes",
}
AGENT_EVAL_FORCE_FINAL_DISABLE_THINKING = get_env_bool(
    "AGENT_EVAL_FORCE_FINAL_DISABLE_THINKING",
    default=False,
)
AGENT_EVAL_FORCE_FINAL_MAX_OUTPUT_TOKENS = max(
    0,
    get_env_int("AGENT_EVAL_FORCE_FINAL_MAX_OUTPUT_TOKENS", default=0),
)
AGENT_EVAL_FORCE_FINAL_RECOVERY_MAX_OUTPUT_TOKENS = max(
    1,
    get_env_int(
        "AGENT_EVAL_FORCE_FINAL_RECOVERY_MAX_OUTPUT_TOKENS",
        default=2048,
    ),
)
AGENT_EVAL_FORCE_FINAL_RECOVERY_DRAFT_CHARS = max(
    1000,
    get_env_int("AGENT_EVAL_FORCE_FINAL_RECOVERY_DRAFT_CHARS", default=32000),
)

# Opt-in context compaction for context-length errors. Disabled by default so
# existing runtime logic is unaffected. When enabled, older conversation history
# is summarized on context-length errors and the request is retried.
AGENT_CONTEXT_COMPACTION = get_env_bool("AGENT_CONTEXT_COMPACTION", default=False)
AGENT_CONTEXT_COMPACT_RECENT_TURNS = max(
    1,
    get_env_int("AGENT_CONTEXT_COMPACT_RECENT_TURNS", default=8),
)
AGENT_CONTEXT_COMPACT_SUMMARY_MODEL = os.getenv("AGENT_CONTEXT_COMPACT_SUMMARY_MODEL", "")
AGENT_CONTEXT_COMPACT_MAX_ATTEMPTS = max(
    1,
    get_env_int("AGENT_CONTEXT_COMPACT_MAX_ATTEMPTS", default=3),
)

# Opt-in evaluation-only observation masking.  The authoritative ``messages``
# history remains unchanged for tool execution and diagnostics; masking is
# applied only to the request View sent to the model.  This is deliberately
# disabled by default so existing training and evaluation processes are
# unaffected unless they explicitly opt in.
AGENT_OBSERVATION_MASKING = get_env_bool("AGENT_OBSERVATION_MASKING", default=False)
AGENT_OBSERVATION_MASKING_WINDOW = max(
    0,
    get_env_int("AGENT_OBSERVATION_MASKING_WINDOW", default=8),
)
AGENT_OBSERVATION_MASKING_PLACEHOLDER = os.getenv(
    "AGENT_OBSERVATION_MASKING_PLACEHOLDER",
    "<MASKED>",
)

VERBOSE_AGENT_LOGS = os.getenv("VERBOSE_AGENT_LOGS", "0").lower() in {
    "1",
    "true",
    "yes",
}
PYTHON_TOOL_TRACE_CODE_PREVIEW_CHARS = max(
    0,
    get_env_int("PYTHON_TOOL_TRACE_CODE_PREVIEW_CHARS", default=2048),
)
if CONTEXT_MANAGEMENT_STRATEGY != "none":
    MAX_LLM_CALL_PER_RUN = 1e9

TOOL_CLASS = [
    Visit(),
    Search(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

regex_pattern = r"[\s\S]*<\/think>"


def debug_print(*args, **kwargs):
    if VERBOSE_AGENT_LOGS:
        print(*args, **kwargs)


def _tool_arguments_hash(raw_tool_args: str) -> str:
    return hashlib.sha256(str(raw_tool_args or "").encode("utf-8", errors="replace")).hexdigest()


def _python_tool_code_preview(tool_name: str, tool_args: Any, raw_tool_args: str) -> dict[str, Any]:
    if tool_name not in {"PythonInterpreter", "python", "code_executor"}:
        return {}
    code: Any = ""
    if isinstance(tool_args, dict):
        code = tool_args.get("code") or tool_args.get("raw") or ""
    if not isinstance(code, str):
        code = str(code)
    preview_chars = PYTHON_TOOL_TRACE_CODE_PREVIEW_CHARS
    preview = code[:preview_chars] if preview_chars > 0 else ""
    return {
        "python_code_chars": len(code),
        "python_code_preview_chars": len(preview),
        "python_code_preview": preview,
    }


RESPONSES_MODE_ENABLED_VALUES = {"1", "true", "yes", "on", "responses", "always", "portable"}
RESPONSES_MODE_DISABLED_VALUES = {
    "",
    "0",
    "false",
    "no",
    "off",
    "never",
    "chat",
    "chat_completions",
    "chat-completions",
}
REASONING_EFFORT_ALIAS_RE = re.compile(
    r"^(?P<model>.+)-(?P<effort>xhigh|high|medium|low|minimal)$",
    flags=re.IGNORECASE,
)


def _chat_tools_with_reasoning_effort_error(error_text: str) -> bool:
    lowered = str(error_text or "").lower()
    return (
        "function tools" in lowered
        and "reasoning_effort" in lowered
        and "/v1/chat/completions" in lowered
        and "/v1/responses" in lowered
    )


def _extract_responses_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"output_text", "input_text", "text"}:
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts).strip()


def _responses_unsupported_sampling_params(served_model_name: str) -> set[str]:
    model_name = str(served_model_name or "").strip().lower()
    match = REASONING_EFFORT_ALIAS_RE.match(model_name)
    base_model = match.group("model") if match else model_name
    # Use server sampling defaults for all GPT models in Responses mode.
    if base_model.startswith("gpt-"):
        return {"temperature", "top_p"}
    return set()


def terminal_print(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class APIResponseError(RuntimeError):
    def __init__(
        self,
        message: str,
        retryable: bool = True,
        *,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


class ServedModelSelection(NamedTuple):
    model: str
    generation: int
    primary_probe: bool
    active_model: str


class ServedModelTransition(NamedTuple):
    action: str
    from_model: str
    to_model: str


class ServedModelFailoverState:
    """Thread-safe request-level model failover shared by task workers."""

    def __init__(
        self,
        primary_model: str,
        fallback_models: tuple[str, ...],
        primary_recovery_interval_sec: float,
    ):
        self.primary_model = primary_model
        self.fallback_models = fallback_models
        self.primary_recovery_interval_sec = max(
            0.0, float(primary_recovery_interval_sec)
        )
        self.current_model = primary_model
        self.generation = 0
        self.last_primary_probe_at = 0.0
        self.primary_probe_inflight = False
        self._lock = threading.Lock()

    def select(self, now: Optional[float] = None) -> ServedModelSelection:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            active_model = self.current_model
            if (
                active_model != self.primary_model
                and not self.primary_probe_inflight
                and now - self.last_primary_probe_at
                >= self.primary_recovery_interval_sec
            ):
                self.primary_probe_inflight = True
                self.last_primary_probe_at = now
                return ServedModelSelection(
                    self.primary_model,
                    self.generation,
                    True,
                    active_model,
                )
            return ServedModelSelection(
                active_model,
                self.generation,
                False,
                active_model,
            )

    def failure(
        self,
        selection: ServedModelSelection,
        *,
        eligible: bool,
        now: Optional[float] = None,
    ) -> Optional[ServedModelTransition]:
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            if selection.primary_probe:
                self.primary_probe_inflight = False
            if not eligible or selection.generation != self.generation:
                return None

            if selection.model == self.primary_model:
                if self.current_model != self.primary_model:
                    return None
                next_model = self.fallback_models[0]
                previous_model = self.current_model
                self.current_model = next_model
                self.generation += 1
                self.last_primary_probe_at = now
                return ServedModelTransition(
                    "fallback", previous_model, next_model
                )

            if (
                selection.model == self.current_model
                and len(self.fallback_models) > 1
            ):
                current_index = self.fallback_models.index(selection.model)
                next_model = self.fallback_models[
                    (current_index + 1) % len(self.fallback_models)
                ]
                if next_model != self.current_model:
                    previous_model = self.current_model
                    self.current_model = next_model
                    self.generation += 1
                    return ServedModelTransition(
                        "fallback_rotate", previous_model, next_model
                    )
            return None

    def success(
        self,
        selection: ServedModelSelection,
    ) -> Optional[ServedModelTransition]:
        with self._lock:
            if selection.primary_probe:
                self.primary_probe_inflight = False
                if (
                    selection.generation == self.generation
                    and self.current_model != self.primary_model
                ):
                    previous_model = self.current_model
                    self.current_model = self.primary_model
                    self.generation += 1
                    return ServedModelTransition(
                        "primary_recovered", previous_model, self.primary_model
                    )
            return None


_SERVED_MODEL_FAILOVER_STATES: dict[
    tuple[str, str, tuple[str, ...], float], ServedModelFailoverState
] = {}
_SERVED_MODEL_FAILOVER_STATES_LOCK = threading.Lock()


def _parse_served_model_fallbacks(raw_value: str, primary_model: str) -> tuple[str, ...]:
    fallbacks: list[str] = []
    for raw_model in str(raw_value or "").split(","):
        model = raw_model.strip()
        if not model or model == primary_model or model in fallbacks:
            continue
        fallbacks.append(model)
    return tuple(fallbacks)


def get_served_model_failover_state(
    base_url: str,
    primary_model: str,
    fallback_models: tuple[str, ...],
    primary_recovery_interval_sec: float,
) -> Optional[ServedModelFailoverState]:
    if not fallback_models:
        return None
    key = (
        str(base_url or "").rstrip("/"),
        primary_model,
        fallback_models,
        float(primary_recovery_interval_sec),
    )
    with _SERVED_MODEL_FAILOVER_STATES_LOCK:
        state = _SERVED_MODEL_FAILOVER_STATES.get(key)
        if state is None:
            state = ServedModelFailoverState(
                primary_model,
                fallback_models,
                primary_recovery_interval_sec,
            )
            _SERVED_MODEL_FAILOVER_STATES[key] = state
        return state


def clear_served_model_failover_states() -> None:
    """Clear shared state for deterministic tests and fresh embedded runs."""
    with _SERVED_MODEL_FAILOVER_STATES_LOCK:
        _SERVED_MODEL_FAILOVER_STATES.clear()


def _error_status_code(error: Exception) -> Optional[int]:
    value = getattr(error, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def is_served_model_failover_error(
    error: Exception,
    error_text: Optional[str] = None,
) -> bool:
    status_code = _error_status_code(error)
    if status_code in {400, 401, 403}:
        return False
    if status_code == 404 or status_code in {408, 409, 429}:
        return True
    if status_code is not None and status_code >= 500:
        return True
    if isinstance(
        error,
        (
            APIConnectionError,
            APITimeoutError,
            TimeoutError,
            httpx.TimeoutException,
            httpx.TransportError,
        ),
    ):
        return True

    lowered = str(error_text if error_text is not None else error).lower()
    service_failure_markers = (
        "get_channel_failed",
        "available channel",
        "no channel",
        "service unavailable",
        "temporarily unable",
        "rate limit",
        "too many requests",
        "timed out",
        "timeout",
        "connection reset",
        "connection refused",
        "gateway timeout",
        "overloaded",
        "overload",
        "try again later",
        "可用渠道不存在",
        "暂无可用渠道",
    )
    return any(marker in lowered for marker in service_failure_markers)


def served_model_failover_reason(
    error: Exception,
    error_text: Optional[str] = None,
) -> str:
    status_code = _error_status_code(error)
    if status_code is not None:
        return f"http_{status_code}"
    if isinstance(error, (APITimeoutError, TimeoutError, httpx.TimeoutException)):
        return "timeout"
    if isinstance(error, (APIConnectionError, httpx.TransportError)):
        return "connection_error"
    lowered = str(error_text if error_text is not None else error).lower()
    if "get_channel_failed" in lowered or "可用渠道不存在" in lowered:
        return "channel_unavailable"
    if "rate limit" in lowered or "too many requests" in lowered:
        return "rate_limited"
    return "service_unavailable"


class IncompleteOutputError(RuntimeError):
    def __init__(
        self,
        attempts: int,
        finish_reason: str,
        *,
        partial_response: Any = None,
        usage: Optional[dict[str, int]] = None,
        attempt_metadata: Optional[list[dict[str, Any]]] = None,
        request_messages: Any = None,
        max_output_tokens: Optional[int] = None,
    ):
        self.attempts = attempts
        self.finish_reason = normalize_finish_reason(finish_reason)
        self.partial_response = partial_response
        self.usage = dict(usage or {})
        self.attempt_metadata = list(attempt_metadata or [])
        self.request_messages = request_messages
        self.max_output_tokens = max_output_tokens
        super().__init__(
            "incomplete output after "
            f"{attempts} attempts (finish_reason={self.finish_reason})"
        )


class TruncatedResponseMessage:
    """Proxy a partial provider response without treating it as a runner error."""

    _molt_truncated = True

    def __init__(self, response_message):
        self._response_message = response_message

    def __getattr__(self, name):
        return getattr(self._response_message, name)


def normalize_finish_reason(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def extract_choice_finish_reason(choice: Any) -> str:
    if isinstance(choice, dict):
        value = choice.get("finish_reason") or choice.get("stop_reason")
    else:
        value = getattr(choice, "finish_reason", None)
        if value is None:
            value = getattr(choice, "stop_reason", None)
    return normalize_finish_reason(value)


class ContextLengthErrorInfo(NamedTuple):
    max_context_tokens: int
    input_tokens: int
    requested_output_tokens: int


class ContextLengthAdjustment(NamedTuple):
    max_context_tokens: int
    input_tokens: int
    requested_output_tokens: int
    adjusted_output_tokens: int
    min_output_tokens: int
    backend_input_tokens: Optional[int] = None
    local_input_tokens: Optional[int] = None
    strategy: str = "budget"
    previous_backend_input_tokens: Optional[int] = None
    previous_requested_output_tokens: Optional[int] = None


_TOKEN_ENCODING = None
_TOKEN_ENCODING_LOAD_ATTEMPTED = False


def _parse_int_token_value(value: str) -> int:
    return int(value.replace(",", ""))


def _first_int_match(patterns: Tuple[str, ...], text: str) -> Optional[int]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return _parse_int_token_value(match.group(1))
    return None


def parse_context_length_error(error_text: str) -> Optional[ContextLengthErrorInfo]:
    if not error_text:
        return None

    max_context_tokens = _first_int_match(
        (
            r"maximum context length is\s+([0-9,]+)\s+tokens",
            r"maximum context length of\s+([0-9,]+)\s+tokens",
            r"max(?:imum)? context(?: length)?(?: is|:| of)?\s+([0-9,]+)\s+tokens",
            r"model['’]?s context length\s*\(([0-9,]+)\s+tokens\)",
        ),
        error_text,
    )
    input_tokens = _first_int_match(
        (
            r"input\s*\(([0-9,]+)\s+tokens\)\s+is longer than",
            r"prompt contains at least\s+([0-9,]+)\s+input tokens",
            r"prompt contains\s+([0-9,]+)\s+input tokens",
            r"([0-9,]+)\s+tokens from the input messages",
            r"([0-9,]+)\s+in the messages",
            r"input_tokens[^0-9]+([0-9,]+)",
        ),
        error_text,
    )
    requested_output_tokens = _first_int_match(
        (
            r"requested\s+([0-9,]+)\s+output tokens",
            r"([0-9,]+)\s+tokens for the completion",
            r"([0-9,]+)\s+in the completion",
            r"max_tokens[^0-9]+([0-9,]+)",
        ),
        error_text,
    )
    if (
        requested_output_tokens is None
        and max_context_tokens is not None
        and input_tokens is not None
        and re.search(
            r"input\s*\([0-9,]+\s+tokens\)\s+is longer than\s+the\s+model['’]?s\s+context\s+length",
            error_text,
            flags=re.IGNORECASE,
        )
    ):
        # SGLang reports input-only overflow without a requested output budget.
        # There is no max_tokens value that can make this request fit.
        requested_output_tokens = 0

    if (
        max_context_tokens is None
        or input_tokens is None
        or requested_output_tokens is None
    ):
        return None

    return ContextLengthErrorInfo(
        max_context_tokens=max_context_tokens,
        input_tokens=input_tokens,
        requested_output_tokens=requested_output_tokens,
    )


def _get_token_encoding():
    global _TOKEN_ENCODING, _TOKEN_ENCODING_LOAD_ATTEMPTED
    if _TOKEN_ENCODING_LOAD_ATTEMPTED:
        return _TOKEN_ENCODING

    _TOKEN_ENCODING_LOAD_ATTEMPTED = True
    try:
        import tiktoken

        _TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _TOKEN_ENCODING = None
    return _TOKEN_ENCODING


def _json_for_token_estimate(payload) -> str:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
    except Exception:
        return str(payload)


def estimate_request_input_tokens(messages, tool_schemas=None) -> int:
    payload = {"messages": messages}
    if tool_schemas:
        payload["tools"] = tool_schemas

    text = _json_for_token_estimate(payload)
    encoding = _get_token_encoding()
    if encoding is not None:
        try:
            return len(encoding.encode(text))
        except Exception:
            pass

    # Conservative fallback for environments without tiktoken.
    return max(1, (len(text) + 2) // 3)


def _is_lower_bound_context_error(
    max_context_tokens: int,
    input_tokens: int,
    requested_output_tokens: int,
) -> bool:
    return input_tokens + requested_output_tokens == max_context_tokens + 1


def _is_lower_bound_adjustment(adjustment: ContextLengthAdjustment) -> bool:
    return _is_lower_bound_context_error(
        adjustment.max_context_tokens,
        adjustment.input_tokens,
        adjustment.requested_output_tokens,
    )


def _is_lower_bound_drift(
    previous_adjustment: Optional[ContextLengthAdjustment],
    current_info: ContextLengthErrorInfo,
) -> bool:
    if previous_adjustment is None:
        return False
    if not _is_lower_bound_adjustment(previous_adjustment):
        return False
    if not _is_lower_bound_context_error(
        current_info.max_context_tokens,
        current_info.input_tokens,
        current_info.requested_output_tokens,
    ):
        return False

    input_delta = current_info.input_tokens - previous_adjustment.input_tokens
    output_delta = (
        previous_adjustment.requested_output_tokens
        - current_info.requested_output_tokens
    )
    if input_delta <= 0 or output_delta <= 0:
        return False

    tolerance = max(16, int(output_delta * 0.05))
    return abs(input_delta - output_delta) <= tolerance


def calculate_context_length_adjustment(
    error_text: str,
    *,
    current_max_output_tokens: int,
    min_output_tokens: int,
    local_input_tokens: Optional[int] = None,
    previous_adjustment: Optional[ContextLengthAdjustment] = None,
) -> Optional[ContextLengthAdjustment]:
    info = parse_context_length_error(error_text)
    if info is None:
        return None

    sanitized_local_input_tokens = (
        max(0, int(local_input_tokens))
        if local_input_tokens is not None
        else None
    )
    input_tokens_for_budget = info.input_tokens
    adjusted_output_tokens = min(
        current_max_output_tokens,
        info.requested_output_tokens,
        info.max_context_tokens - input_tokens_for_budget - 1,
    )
    strategy = "budget"

    if _is_lower_bound_drift(previous_adjustment, info):
        binary_output_tokens = max(
            min_output_tokens,
            current_max_output_tokens // 2,
        )
        adjusted_output_tokens = min(
            adjusted_output_tokens,
            binary_output_tokens,
        )
        strategy = "lower_bound_binary"

    return ContextLengthAdjustment(
        max_context_tokens=info.max_context_tokens,
        input_tokens=input_tokens_for_budget,
        requested_output_tokens=info.requested_output_tokens,
        adjusted_output_tokens=adjusted_output_tokens,
        min_output_tokens=min_output_tokens,
        backend_input_tokens=info.input_tokens,
        local_input_tokens=sanitized_local_input_tokens,
        strategy=strategy,
        previous_backend_input_tokens=(
            previous_adjustment.backend_input_tokens
            if previous_adjustment is not None
            else None
        ),
        previous_requested_output_tokens=(
            previous_adjustment.requested_output_tokens
            if previous_adjustment is not None
            else None
        ),
    )


class ContextCompactor:
    """Summarize older conversation history when a context-length error occurs.

    This is a purely additive recovery path. When ``AGENT_CONTEXT_COMPACTION``
    is disabled (the default), this class is never instantiated and existing
    code paths remain unchanged.
    """

    def __init__(
        self,
        summary_model_name: str,
        recent_turns: int,
        max_attempts: int,
    ) -> None:
        self.summary_model_name = summary_model_name
        self.recent_turns = recent_turns
        self.max_attempts = max_attempts

    def compact(
        self,
        messages: list[dict[str, Any]],
        error_text: str,
        attempt: int,
        client: OpenAI,
        request_model: str,
        tool_schemas: Optional[list[dict[str, Any]]],
        request_timeout: float,
    ) -> Optional[list[dict[str, Any]]]:
        if attempt >= self.max_attempts:
            return None

        info = parse_context_length_error(error_text)
        if info is None:
            return None

        new_messages = self._compact_messages(
            messages,
            client,
            request_model,
            tool_schemas,
            request_timeout,
            info.max_context_tokens,
        )
        if new_messages is None:
            return None

        # Verify compaction actually reduced the estimated token count.
        old_tokens = estimate_request_input_tokens(messages, tool_schemas)
        new_tokens = estimate_request_input_tokens(new_messages, tool_schemas)
        if new_tokens >= old_tokens:
            debug_print(
                "Context compaction did not reduce token count "
                f"({old_tokens} -> {new_tokens}); aborting."
            )
            return None

        return new_messages

    def _compact_messages(
        self,
        messages: list[dict[str, Any]],
        client: OpenAI,
        request_model: str,
        tool_schemas: Optional[list[dict[str, Any]]],
        request_timeout: float,
        max_context_tokens: int,
    ) -> Optional[list[dict[str, Any]]]:
        if not messages:
            return None

        # System prompt is always preserved at the start.
        system_message: Optional[dict[str, Any]] = None
        start_idx = 0
        if messages[0].get("role") == "system":
            system_message = messages[0]
            start_idx = 1

        # The current prompt is the last message.
        current_prompt = messages[-1]
        middle_messages = messages[start_idx:-1]
        if not middle_messages:
            return None

        # Identify assistant markers so we can reason in whole turns.
        assistant_indices = [
            i for i, m in enumerate(middle_messages) if m.get("role") == "assistant"
        ]
        if len(assistant_indices) <= self.recent_turns:
            return None

        # Keep everything from the Nth most recent assistant turn onward.
        # The cutoff must be the user message that prompted that assistant,
        # so the preserved segment starts with a user message and the assistant
        # still has its preceding query. This also keeps any multi-tool results
        # that belong to the same turn intact.
        recent_assistant_idx = assistant_indices[-self.recent_turns]
        recent_cutoff = max(0, recent_assistant_idx - 1)
        if middle_messages[recent_cutoff].get("role") != "user":
            # Malformed history: fall back to keeping the assistant itself.
            # The caller will likely fail later, but we avoid making it worse.
            recent_cutoff = recent_assistant_idx
        old_messages = middle_messages[:recent_cutoff]
        recent_messages = middle_messages[recent_cutoff:]
        if not old_messages:
            return None

        # Find turn boundaries within the older history. A turn starts at the
        # user message immediately before each assistant message.
        old_assistant_indices = [
            i for i, m in enumerate(old_messages) if m.get("role") == "assistant"
        ]
        if not old_assistant_indices:
            return None

        # Summarize the oldest half of the old turns, keep the rest.
        split_turns = max(1, len(old_assistant_indices) // 2)
        if split_turns >= len(old_assistant_indices):
            split_idx = 0
        else:
            split_idx = max(0, old_assistant_indices[split_turns] - 1)
        to_compact = old_messages[:split_idx]
        to_keep = old_messages[split_idx:]
        if not to_compact:
            return None

        summary_message = self._summarize_messages(
            to_compact,
            client,
            request_model,
            tool_schemas,
            request_timeout,
            max_context_tokens,
        )
        if summary_message is None:
            return None

        new_messages: list[dict[str, Any]] = []
        if system_message is not None:
            new_messages.append(system_message)
        if to_keep:
            new_messages.extend(to_keep)
        new_messages.append(summary_message)
        new_messages.extend(recent_messages)
        new_messages.append(current_prompt)
        return new_messages

    def _summarize_messages(
        self,
        messages: list[dict[str, Any]],
        client: OpenAI,
        request_model: str,
        tool_schemas: Optional[list[dict[str, Any]]],
        request_timeout: float,
        max_context_tokens: int,
    ) -> Optional[dict[str, Any]]:
        del tool_schemas
        model = self.summary_model_name or request_model
        # Adapt summary budget to the model context limit; no user-facing knob.
        summary_max_tokens = min(max(2048, max_context_tokens // 4), 16384)

        summary_prompt = {
            "role": "system",
            "content": (
                "You are compressing a long agent conversation to fit within a "
                "context window. Summarize the following earlier conversation "
                "turns into a dense, state-preserving summary. Keep: the original "
                "task, key decisions, key facts or data obtained from tools, "
                "current progress, and any partial or final answers. Drop: "
                "verbatim raw tool output, repetitive reasoning, and failed "
                "intermediate attempts. Use concise bullet points. The summary "
                "must contain everything needed to continue the task from where "
                "it left off."
            ),
        }
        summary_user = {
            "role": "user",
            "content": "Earlier conversation turns to summarize:\n"
            + json.dumps(messages, ensure_ascii=False, default=str),
        }

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[summary_prompt, summary_user],
                temperature=0.1,
                max_tokens=summary_max_tokens,
                timeout=request_timeout,
            )
            content = response.choices[0].message.content
            if not content:
                return None
            return {
                "role": "assistant",
                "content": f"[Context summary]\n{content}",
            }
        except Exception as exc:
            debug_print(f"Context compaction summary model call failed: {exc}")
            return None


_CONTEXT_COMPACTOR: Optional[ContextCompactor] = None


def _get_context_compactor() -> Optional[ContextCompactor]:
    if not AGENT_CONTEXT_COMPACTION:
        return None
    global _CONTEXT_COMPACTOR
    if _CONTEXT_COMPACTOR is None:
        _CONTEXT_COMPACTOR = ContextCompactor(
            summary_model_name=AGENT_CONTEXT_COMPACT_SUMMARY_MODEL,
            recent_turns=AGENT_CONTEXT_COMPACT_RECENT_TURNS,
            max_attempts=AGENT_CONTEXT_COMPACT_MAX_ATTEMPTS,
        )
    return _CONTEXT_COMPACTOR


def today_date():
    return datetime.today().strftime("%Y-%m-%d")


def safe_strip(text) -> str:
    return text.strip() if isinstance(text, str) else ""


TOOL_MULTIMODAL_RESPONSE_ERROR_MARKERS = (
    "tool",
    "image_url",
    "unsupported content",
    "expected a string",
    "content type",
    "multimodal",
    "role=tool",
)


def _content_part_has_image(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == "image_url"


def tool_content_has_image(content: Any) -> bool:
    return isinstance(content, list) and any(_content_part_has_image(part) for part in content)


def tool_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    image_count = 0
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text)
        elif part.get("type") == "image_url":
            image_count += 1

    if image_count:
        parts.append(
            f"[{image_count} visual tool image(s) omitted from this text fallback. "
            "Use the artifact path(s) above if provided.]"
        )
    return "\n".join(parts).strip()


def _messages_have_tool_image(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and tool_content_has_image(message.get("content"))
        for message in messages
    )


def _tool_multimodal_error_hint(error_text: str) -> bool:
    lowered = str(error_text or "").lower()
    return any(marker in lowered for marker in TOOL_MULTIMODAL_RESPONSE_ERROR_MARKERS)


def extract_usage_dict(chat_response) -> dict[str, int]:
    usage = getattr(chat_response, "usage", None)
    if usage is None:
        return {}

    if hasattr(usage, "model_dump"):
        try:
            payload = usage.model_dump(exclude_none=True)
        except TypeError:
            payload = usage.model_dump()
    elif isinstance(usage, dict):
        payload = usage
    else:
        payload = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    usage_dict: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            usage_dict[key] = int(value)
        except (TypeError, ValueError):
            continue
    return usage_dict


def merge_usage_dict(target: dict[str, int], usage: dict[str, int]) -> bool:
    updated = False
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is None:
            continue
        target[key] = target.get(key, 0) + int(value)
        updated = True
    return updated


class MultiTurnReactAgent(FnCallAgent):
    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
        llm: Optional[Union[Dict, BaseChatModel]] = None,
        **kwargs,
    ):
        del function_list, kwargs
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self.served_model_name = llm.get("served_model_name") or self.llm_local_path
        self._served_model_fallbacks_raw = os.getenv(
            "SERVED_MODEL_NAME_FALLBACKS", ""
        )
        self.served_model_name_fallbacks = _parse_served_model_fallbacks(
            self._served_model_fallbacks_raw,
            self.served_model_name,
        )
        self.served_model_primary_recovery_interval_sec = max(
            0.0,
            get_env_float(
                "SERVED_MODEL_PRIMARY_RECOVERY_INTERVAL_SEC",
                default=300.0,
            ),
        )
        self.served_model_names_used: list[str] = []
        self.served_model_name_actual: Optional[str] = None
        self.max_output_tokens = int(
            os.getenv("QWEN_API_MAX_OUTPUT_TOKENS")
            or os.getenv("MAX_OUTPUT_TOKENS")
            or "96000"
        )
        self.request_timeout = get_env_float(
            "OPENAI_REQUEST_TIMEOUT",
            "LLM_REQUEST_TIMEOUT_SEC",
            default=600.0,
        )
        self.llm_api_max_retries = max(
            1,
            get_env_int("LLM_API_MAX_RETRIES", default=10),
        )
        self.llm_retry_base_sleep_sec = max(
            0.0,
            get_env_float("LLM_RETRY_BASE_SLEEP_SEC", default=1.0),
        )
        self.llm_retry_max_sleep_sec = max(
            0.0,
            get_env_float("LLM_RETRY_MAX_SLEEP_SEC", default=30.0),
        )
        self.llm_initial_jitter_max_sec = max(
            0.0,
            get_env_float("LLM_INITIAL_JITTER_MAX_SEC", default=0.5),
        )
        self.auto_adjust_max_output_tokens_on_context_error = get_env_bool(
            "LLM_AUTO_ADJUST_MAX_OUTPUT_TOKENS_ON_CONTEXT_ERROR",
            default=False,
        )
        self.context_error_min_output_tokens = max(
            1,
            get_env_int("LLM_CONTEXT_ERROR_MIN_OUTPUT_TOKENS", default=4096),
        )
        self.task_timeout_sec = TASK_TIMEOUT_SEC
        self.adaptive_context_mgr = None
        self._tool_multimodal_response_fallback = False
        self._fallback_message_lists = []
        self._context_tokenizer = None
        self._context_tokenizer_unavailable = False

    def _infer_tool_status(self, tool_response: str) -> str:
        text = (tool_response or "").lower()
        failure_markers = (
            "error",
            "invalid request",
            "traceback",
            "failed",
            "timeout",
            "exception",
            "not found",
        )
        return "failed" if any(marker in text for marker in failure_markers) else "success"

    def _apply_adaptive_context(
        self,
        base_messages,
        round_idx: int,
        assistant_content: str,
        tool_response: str,
        tool_names,
        tool_args,
        raw_messages,
    ):
        del base_messages, round_idx, assistant_content, tool_response, tool_names, tool_args, raw_messages
        return None

    def _configure_task_tools(
        self,
        item: dict,
        item_base_dir: str,
        data: dict,
        *,
        user_message: Optional[dict] = None,
    ) -> str:
        del item, item_base_dir, data, user_message
        self._tool_multimodal_response_fallback = False
        return ""

    def _active_tool_schemas_for_current_task(self):
        return None

    def _set_tool_text_response_fallback(self, enabled: bool) -> None:
        del enabled

    def _append_text_to_user_message(self, user_message: dict, text: str) -> None:
        if not text:
            return
        content = user_message.get("content")
        if isinstance(content, str):
            user_message["content"] = content + text
            return
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = str(part.get("text") or "") + text
                    return
            content.insert(0, {"type": "text", "text": text.strip()})

    def _set_fallback_message_lists(self, *message_lists) -> None:
        self._fallback_message_lists = [item for item in message_lists if isinstance(item, list)]

    @staticmethod
    def _observation_masked_view(messages):
        """Return the optional model-facing view with old tool observations masked.

        ``messages`` is the authoritative conversation history used for tool
        execution, result serialization, and retries.  This helper always
        deep-copies it, so the default-disabled path and the caller's history
        are unchanged.  Only ``role=tool`` message content is eligible; the
        assistant tool call, tool name, and call id remain visible so the
        model still sees the conversation structure.
        """
        if not AGENT_OBSERVATION_MASKING or not isinstance(messages, list):
            return messages

        tool_indexes = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        keep = AGENT_OBSERVATION_MASKING_WINDOW
        mask_indexes = tool_indexes[:-keep] if keep else tool_indexes
        if not mask_indexes:
            return messages

        view = copy.deepcopy(messages)
        for index in mask_indexes:
            message = view[index]
            # A fixed text placeholder avoids leaking old tool output while
            # preserving a valid OpenAI tool-message shape.
            message["content"] = AGENT_OBSERVATION_MASKING_PLACEHOLDER
        return view

    def _maybe_downgrade_multimodal_tool_messages(
        self,
        error_text: str,
        *,
        request_messages=None,
    ) -> bool:
        if str(os.getenv("VISUAL_TOOL_RESPONSE_MODE") or "auto").strip().lower() == "multimodal":
            return False
        if not _tool_multimodal_error_hint(error_text):
            return False

        message_lists = []
        if isinstance(request_messages, list):
            message_lists.append(request_messages)
        message_lists.extend(self._fallback_message_lists)

        changed = False
        seen_message_ids: set[int] = set()
        for messages in message_lists:
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if id(message) in seen_message_ids:
                    continue
                seen_message_ids.add(id(message))
                if message.get("role") != "tool":
                    continue
                content = message.get("content")
                if not tool_content_has_image(content):
                    continue
                message["content"] = tool_content_to_text(content)
                changed = True

        if changed:
            self._tool_multimodal_response_fallback = True
            self._set_tool_text_response_fallback(True)
        return changed

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content

    def _get_api_key(self) -> str:
        return (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("API_KEY")
            or ""
        )

    def _get_api_base(self) -> str:
        return (
            os.getenv("OPENAI_API_BASE")
            or os.getenv("BASE_URL")
            or os.getenv("API_BASE")
            or ""
        )

    def _build_system_prompt(self) -> str:
        return ""

    def _fetch_available_models(self, openai_api_base: str) -> List[str]:
        try:
            response = httpx.get(f"{openai_api_base}/models", timeout=10.0)
            response.raise_for_status()
            payload = response.json()
            return [item.get("id", "") for item in payload.get("data", [])]
        except Exception as exc:
            return [f"<failed to fetch /v1/models: {exc}>"]

    def _format_model_resolution_error(
        self, openai_api_base: str, requested_model: str, error: Exception
    ) -> str:
        available_models = self._fetch_available_models(openai_api_base)
        return (
            "API model resolution failed. "
            f"base_url={openai_api_base}, requested_model={requested_model}, "
            f"available_models={available_models}, error={error}"
        )

    def _extract_reasoning(self, response_message) -> str:
        return extract_reasoning_text(response_message)

    def _deepseek_v4_force_enabled_for_model(self, model_name: str) -> bool:
        return (
            deepseek_v4_reasoning_opt_in_enabled()
            and str(model_name or "").strip()
            == str(self.served_model_name or "").strip()
        )

    def _deepseek_v4_profile_model_name(self, model_name: str) -> str:
        active_model = str(model_name or "").strip()
        if is_deepseek_v4_model(active_model):
            return active_model
        if (
            active_model == str(self.served_model_name or "").strip()
            and is_deepseek_v4_model(self.llm_local_path)
        ):
            return str(self.llm_local_path)
        return active_model

    def _deepseek_v4_response_enabled(self) -> bool:
        model_name = self.served_model_name_actual or self.served_model_name
        profile_model_name = self._deepseek_v4_profile_model_name(model_name)
        return is_deepseek_v4_model(profile_model_name) or (
            deepseek_v4_reasoning_opt_in_enabled()
            and str(model_name or "").strip()
            == str(self.served_model_name or "").strip()
        )

    def _extract_response_text_fields(
        self,
        response_message,
    ) -> tuple[str, str]:
        reasoning = self._extract_reasoning(response_message)
        content = safe_strip(getattr(response_message, "content", ""))
        if not self._deepseek_v4_response_enabled():
            model_names = f"{self.llm_local_path} {self.served_model_name}".lower()
            if AGENT_OBSERVATION_MASKING and "qwen" in model_names:
                # Molt decodes raw completion tokens; framing belongs to the
                # chat template, not to the replayed assistant body.
                content = re.sub(r"(?:\s*<\|(?:im_end|endoftext)\|>)+\s*$", "", content).strip()
                if not reasoning and "</think>" in content:
                    reasoning, content = content.split("</think>", 1)
                    reasoning = reasoning.removeprefix("<think>").strip()
                    content = content.strip()
            return reasoning, content

        split_fields = split_deepseek_v4_response_text(content, reasoning)
        if split_fields.fallback_split_used:
            emit_trace(
                "deepseek_v4_reasoning_fallback_split",
                layer="agent_llm",
                served_model=str(
                    self.served_model_name_actual or self.served_model_name
                ),
                reasoning_chars=len(split_fields.reasoning_content),
                content_chars=len(split_fields.content),
            )
        return split_fields.reasoning_content, split_fields.content

    def _responses_mode_enabled(self) -> bool:
        raw_value = os.getenv("OPENAI_COMPAT_RESPONSES_MODE", "").strip().lower()
        if raw_value in RESPONSES_MODE_ENABLED_VALUES:
            return True
        if raw_value in RESPONSES_MODE_DISABLED_VALUES:
            return False
        raise RuntimeError(
            "Invalid OPENAI_COMPAT_RESPONSES_MODE="
            f"{raw_value!r}. Use 'responses' to opt in, 'portable' for visible-history "
            "replay without server-bound reasoning state, or leave it unset/0 "
            "to keep the default /v1/chat/completions path."
        )

    def _responses_portable_replay_enabled(self) -> bool:
        return os.getenv("OPENAI_COMPAT_RESPONSES_MODE", "").strip().lower() == "portable"

    def _responses_model_and_reasoning(
        self,
        served_model_name: Optional[str] = None,
    ) -> tuple[str, dict[str, str]]:
        model_name = str(served_model_name or self.served_model_name or "").strip()
        match = REASONING_EFFORT_ALIAS_RE.match(model_name)
        if match and match.group("model").lower().startswith("gpt-"):
            return match.group("model"), {"effort": match.group("effort").lower()}
        return model_name, {}

    def _responses_sampling_params(
        self,
        served_model_name: Optional[str] = None,
    ) -> tuple[dict[str, Any], list[str]]:
        active_model_name = served_model_name or self.served_model_name
        unsupported_params = _responses_unsupported_sampling_params(active_model_name)
        payload: dict[str, Any] = {}
        omitted_params: list[str] = []
        for param_name in ("temperature", "top_p"):
            if param_name == "top_p" and self.llm_generate_cfg.get("omit_top_p"):
                omitted_params.append(param_name)
                continue
            value = self.llm_generate_cfg.get(param_name)
            if value in (None, ""):
                continue
            if param_name in unsupported_params:
                omitted_params.append(param_name)
                continue
            payload[param_name] = value
        return payload, omitted_params

    def _responses_omitted_sampling_params(
        self,
        served_model_name: Optional[str] = None,
    ) -> list[str]:
        _, omitted_params = self._responses_sampling_params(served_model_name)
        return omitted_params

    def _format_responses_required_error(
        self,
        openai_api_base: str,
        served_model_name: Optional[str] = None,
    ) -> str:
        requested_model = served_model_name or self.served_model_name
        return (
            "The backend rejected /v1/chat/completions with function tools and "
            "reasoning_effort for this model. Set "
            "OPENAI_COMPAT_RESPONSES_MODE=responses to explicitly route this run "
            f"through /v1/responses; base_url={openai_api_base}, "
            f"requested_model={requested_model}."
        )

    def _responses_tool_schemas(self, tool_schemas):
        if not tool_schemas:
            return []

        response_tools = []
        for tool_schema in tool_schemas:
            if not isinstance(tool_schema, dict):
                continue
            if tool_schema.get("type") != "function":
                response_tools.append(copy.deepcopy(tool_schema))
                continue

            function_schema = tool_schema.get("function")
            if not isinstance(function_schema, dict):
                response_tools.append(copy.deepcopy(tool_schema))
                continue

            response_tool = {"type": "function"}
            for key in ("name", "description", "parameters", "strict"):
                if key in function_schema:
                    response_tool[key] = copy.deepcopy(function_schema[key])
            response_tools.append(response_tool)
        return response_tools

    def _responses_content_for_message(self, role: str, content):
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return "" if content is None else str(content)

        if role != "user":
            return tool_content_to_text(content)

        converted = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                converted.append(
                    {"type": "input_text", "text": str(part.get("text") or "")}
                )
            elif part_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    image_url = image_url.get("url")
                if image_url:
                    converted.append({"type": "input_image", "image_url": image_url})
            elif part_type in {"input_text", "input_image"}:
                converted.append(copy.deepcopy(part))
        return converted or tool_content_to_text(content)

    def _responses_function_call_item(self, tool_call: dict) -> dict | None:
        if not isinstance(tool_call, dict):
            return None
        function = tool_call.get("function")
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        call_id = str(tool_call.get("id") or tool_call.get("call_id") or "").strip()
        if not call_id:
            return None
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        return {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }

    def _responses_replay_output_items(self, output_items: list) -> list:
        items = copy.deepcopy(output_items)
        if not self._responses_portable_replay_enabled():
            return items

        replay_items = []
        for item in items:
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type == "reasoning":
                # Keep the original trajectory intact, but never replay opaque
                # reasoning state that may belong to a different upstream resource.
                continue
            if item_type not in ("message", "function_call"):
                raise APIResponseError(
                    "OPENAI_COMPAT_RESPONSES_MODE=portable cannot replay "
                    f"Responses output item type {item_type!r}. Only message and "
                    "function_call items can be replayed without server state; "
                    "reasoning items are omitted. Use responses mode with a "
                    "consistent upstream resource for other item types.",
                    retryable=False,
                )
            # Item IDs are server-side references; call_id still connects each
            # self-contained function call to its function_call_output.
            item.pop("id", None)
            replay_items.append(item)
        return replay_items

    def _messages_to_responses_input(self, messages):
        input_items = []
        instructions = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "system":
                text = self._responses_content_for_message(role, message.get("content"))
                if text:
                    instructions.append(str(text))
                continue

            if role == "tool":
                call_id = str(message.get("tool_call_id") or "").strip()
                if call_id:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": tool_content_to_text(message.get("content")),
                        }
                    )
                continue

            if role == "assistant":
                raw_response_items = message.get("_responses_output")
                if isinstance(raw_response_items, list) and raw_response_items:
                    input_items.extend(self._responses_replay_output_items(raw_response_items))
                    continue

                content = self._responses_content_for_message(role, message.get("content"))
                if content:
                    input_items.append({"role": "assistant", "content": content})
                for tool_call in message.get("tool_calls") or []:
                    call_item = self._responses_function_call_item(tool_call)
                    if call_item:
                        input_items.append(call_item)
                continue

            if role in {"user", "developer"}:
                content = self._responses_content_for_message(role, message.get("content"))
                if content:
                    input_items.append({"role": role, "content": content})

        return input_items, "\n\n".join(instructions).strip()

    def _extract_responses_usage_dict(self, payload: dict[str, Any]) -> dict[str, int]:
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if not isinstance(usage, dict):
            return {}

        mapping = {
            "input_tokens": "prompt_tokens",
            "output_tokens": "completion_tokens",
            "total_tokens": "total_tokens",
        }
        usage_dict: dict[str, int] = {}
        for source_key, target_key in mapping.items():
            value = usage.get(source_key)
            if value is None:
                continue
            try:
                usage_dict[target_key] = int(value)
            except (TypeError, ValueError):
                continue
        return usage_dict

    def _responses_message_from_payload(self, payload: dict[str, Any]):
        output_items = payload.get("output") if isinstance(payload, dict) else None
        if not isinstance(output_items, list):
            output_items = []

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                text = _extract_responses_text(item.get("content"))
                if text:
                    content_parts.append(text)
            elif item_type == "function_call":
                call_id = str(item.get("call_id") or item.get("id") or "").strip()
                if not call_id:
                    continue
                tool_calls.append(
                    SimpleNamespace(
                        type="function",
                        id=call_id,
                        function=SimpleNamespace(
                            name=str(item.get("name") or ""),
                            arguments=(
                                item.get("arguments")
                                if isinstance(item.get("arguments"), str)
                                else json.dumps(item.get("arguments") or {}, ensure_ascii=False)
                            ),
                        ),
                    )
                )
            elif item_type == "reasoning":
                summary = item.get("summary")
                if isinstance(summary, list):
                    for summary_item in summary:
                        if isinstance(summary_item, dict):
                            text = summary_item.get("text")
                            if isinstance(text, str) and text:
                                reasoning_parts.append(text)
                text = item.get("text")
                if isinstance(text, str) and text:
                    reasoning_parts.append(text)

        if not content_parts:
            output_text = payload.get("output_text") if isinstance(payload, dict) else None
            if isinstance(output_text, str) and output_text:
                content_parts.append(output_text)

        return SimpleNamespace(
            content="\n".join(content_parts).strip(),
            reasoning_content="\n".join(reasoning_parts).strip(),
            tool_calls=tool_calls or None,
            _responses_output=copy.deepcopy(output_items),
        )

    def _create_responses_completion(
        self,
        *,
        openai_api_key: str,
        openai_api_base: str,
        messages,
        tool_schemas,
        max_output_tokens: Optional[int],
        served_model_name: Optional[str] = None,
    ) -> tuple[object, dict[str, int]]:
        response_input, instructions = self._messages_to_responses_input(messages)
        model_name, reasoning = self._responses_model_and_reasoning(served_model_name)
        payload: dict[str, Any] = {
            "model": model_name,
            "input": response_input,
        }
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if self._responses_portable_replay_enabled():
            payload["store"] = False
        if instructions:
            payload["instructions"] = instructions
        response_tools = self._responses_tool_schemas(tool_schemas)
        if response_tools:
            payload["tools"] = response_tools
        if reasoning:
            payload["reasoning"] = reasoning
        sampling_params, _ = self._responses_sampling_params(served_model_name)
        payload.update(sampling_params)

        response = httpx.post(
            f"{openai_api_base.rstrip('/')}/responses",
            headers={
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.request_timeout,
        )
        if response.status_code >= 400:
            try:
                error_payload = response.json()
            except Exception:
                error_payload = response.text
            raise APIResponseError(
                "Responses API error: "
                f"status_code={response.status_code}, payload={error_payload}",
                retryable=response.status_code in {408, 409, 429, 500, 502, 503, 504},
                status_code=response.status_code,
            )

        payload = response.json()
        if isinstance(payload, dict) and payload.get("status") == "incomplete":
            details = payload.get("incomplete_details") or {}
            reason = details.get("reason") if isinstance(details, dict) else ""
            if str(reason).lower() in {"max_output_tokens", "length"}:
                # In compaction mode we accept the model/backend output cap
                # and do not treat incomplete output as an error.
                if AGENT_CONTEXT_COMPACTION:
                    return (
                        self._responses_message_from_payload(payload),
                        self._extract_responses_usage_dict(payload),
                    )
                partial_response = self._responses_message_from_payload(payload)
                usage = self._extract_responses_usage_dict(payload)
                raise IncompleteOutputError(
                    1,
                    "length",
                    partial_response=partial_response,
                    usage=usage,
                    attempt_metadata=[
                        {
                            "attempt": 1,
                            "finish_reason": "length",
                            "max_output_tokens": max_output_tokens,
                            "usage": usage,
                        }
                    ],
                    max_output_tokens=max_output_tokens,
                )
        return (
            self._responses_message_from_payload(payload),
            self._extract_responses_usage_dict(payload),
        )

    def _normalize_response_error(self, response_error) -> Dict[str, str]:
        if response_error is None:
            return {}

        if hasattr(response_error, "model_dump"):
            try:
                response_error = response_error.model_dump(exclude_none=True)
            except TypeError:
                response_error = response_error.model_dump()
        elif not isinstance(response_error, dict):
            candidate = {
                "message": getattr(response_error, "message", None),
                "type": getattr(response_error, "type", None),
                "code": getattr(response_error, "code", None),
                "param": getattr(response_error, "param", None),
            }
            if any(value is not None for value in candidate.values()):
                response_error = candidate
            else:
                response_error = {"message": str(response_error)}

        return {
            str(key): "" if value is None else str(value)
            for key, value in response_error.items()
        }

    def _attach_usage_fields(
        self,
        result: dict,
        usage_totals: dict[str, int],
        has_usage: bool,
    ) -> dict:
        if has_usage:
            result["prompt_tokens"] = usage_totals.get("prompt_tokens")
            result["completion_tokens"] = usage_totals.get("completion_tokens")
            result["total_tokens"] = usage_totals.get("total_tokens")
        return result

    def _extract_embedded_error(self, chat_response) -> Dict[str, str]:
        response_error = self._normalize_response_error(
            getattr(chat_response, "error", None)
        )
        if response_error:
            return response_error

        if hasattr(chat_response, "model_dump"):
            try:
                payload = chat_response.model_dump(exclude_none=False)
            except TypeError:
                payload = chat_response.model_dump()
            return self._normalize_response_error(payload.get("error"))

        return {}

    def _dump_response_preview(self, chat_response, max_chars: int = 800) -> str:
        try:
            if hasattr(chat_response, "model_dump"):
                payload = chat_response.model_dump(exclude_none=False)
            elif isinstance(chat_response, dict):
                payload = chat_response
            else:
                payload = {"repr": repr(chat_response)}
            preview = json.dumps(payload, ensure_ascii=False)
        except Exception as exc:
            preview = f"<failed to serialize response: {exc}; repr={chat_response!r}>"

        if len(preview) > max_chars:
            preview = preview[:max_chars] + "...(truncated)"
        return preview

    def _is_retryable_response_error(self, response_error: Dict[str, str]) -> bool:
        haystack = " ".join(
            [
                response_error.get("type", ""),
                response_error.get("code", ""),
                response_error.get("message", ""),
            ]
        ).lower()
        retry_markers = (
            "rate",
            "limit",
            "burst",
            "timeout",
            "tempor",
            "busy",
            "overload",
            "again later",
            "try again",
        )
        return any(marker in haystack for marker in retry_markers)

    def _format_response_diagnostic(
        self,
        chat_response,
        response_error: Optional[Dict[str, str]] = None,
    ) -> str:
        if response_error is None:
            response_error = self._extract_embedded_error(chat_response)

        parts = []
        if response_error:
            parts.append(
                "embedded_error=" + json.dumps(response_error, ensure_ascii=False)
            )
        parts.append("payload=" + self._dump_response_preview(chat_response))
        return ", ".join(parts)

    def _format_context_adjustment_failure(
        self,
        adjustment: ContextLengthAdjustment,
        current_max_tokens: int,
    ) -> str:
        available_output_tokens = (
            adjustment.max_context_tokens
            - adjustment.input_tokens
            - 1
        )
        return (
            "Context length error cannot be fixed by lowering max_tokens: "
            f"max_context_tokens={adjustment.max_context_tokens}, "
            f"budget_input_tokens={adjustment.input_tokens}, "
            f"backend_input_tokens={adjustment.backend_input_tokens}, "
            f"local_input_tokens={adjustment.local_input_tokens}, "
            f"requested_output_tokens={adjustment.requested_output_tokens}, "
            f"current_max_output_tokens={current_max_tokens}, "
            f"strategy={adjustment.strategy}, "
            f"available_output_tokens={available_output_tokens}, "
            f"min_output_tokens={adjustment.min_output_tokens}."
        )

    def _extract_api_error_text(self, error: Exception) -> str:
        parts = [str(error)]
        body = getattr(error, "body", None)
        if body:
            try:
                parts.append(json.dumps(body, ensure_ascii=False, default=str))
            except TypeError:
                parts.append(str(body))
        response = getattr(error, "response", None)
        response_text = getattr(response, "text", None)
        if response_text:
            parts.append(str(response_text))
        return "\n".join(part for part in parts if part)

    def _context_length_adjustment_from_error(
        self,
        error_text: str,
        current_max_output_tokens: int,
        local_input_tokens: Optional[int],
        previous_adjustment: Optional[ContextLengthAdjustment],
    ) -> Optional[ContextLengthAdjustment]:
        if not self.auto_adjust_max_output_tokens_on_context_error:
            return None
        return calculate_context_length_adjustment(
            error_text,
            current_max_output_tokens=current_max_output_tokens,
            min_output_tokens=self.context_error_min_output_tokens,
            local_input_tokens=local_input_tokens,
            previous_adjustment=previous_adjustment,
        )

    def _context_length_adjustment_or_raise(
        self,
        error_text: str,
        current_max_output_tokens: int,
        adjustment_count: int,
        local_input_tokens: Optional[int],
        previous_adjustment: Optional[ContextLengthAdjustment],
    ) -> Optional[ContextLengthAdjustment]:
        adjustment = self._context_length_adjustment_from_error(
            error_text,
            current_max_output_tokens,
            local_input_tokens,
            previous_adjustment,
        )
        if adjustment is None:
            return None

        if current_max_output_tokens <= adjustment.min_output_tokens:
            raise RuntimeError(
                "Context length error persisted at the minimum allowed "
                "max_tokens; the input prompt is too long to fix by lowering "
                "max_tokens further: "
                + self._format_context_adjustment_failure(
                    adjustment,
                    current_max_output_tokens,
                )
            )

        if adjustment.adjusted_output_tokens < adjustment.min_output_tokens:
            raise RuntimeError(
                self._format_context_adjustment_failure(
                    adjustment,
                    current_max_output_tokens,
                )
            )

        if adjustment.adjusted_output_tokens >= current_max_output_tokens:
            raise RuntimeError(
                "Context length error was parsed, but no smaller max_tokens value "
                "is available: "
                f"max_context_tokens={adjustment.max_context_tokens}, "
                f"budget_input_tokens={adjustment.input_tokens}, "
                f"backend_input_tokens={adjustment.backend_input_tokens}, "
                f"local_input_tokens={adjustment.local_input_tokens}, "
                f"requested_output_tokens={adjustment.requested_output_tokens}, "
                f"current_max_output_tokens={current_max_output_tokens}, "
                f"strategy={adjustment.strategy}."
            )

        return adjustment

    def _log_context_adjustment_retry(
        self,
        adjustment: ContextLengthAdjustment,
        adjustment_count: int,
        previous_max_output_tokens: int,
        effective_max_output_tokens: int,
    ) -> None:
        terminal_print(
            "Context length error detected; "
            f"context adjustment {adjustment_count}; "
            "lowering max_tokens from "
            f"{previous_max_output_tokens} to {effective_max_output_tokens} "
            f"(max_context_tokens={adjustment.max_context_tokens}, "
            f"strategy={adjustment.strategy}, "
            f"budget_input_tokens={adjustment.input_tokens}, "
            f"backend_input_tokens={adjustment.backend_input_tokens}, "
            f"local_input_tokens={adjustment.local_input_tokens}, "
            f"previous_backend_input_tokens={adjustment.previous_backend_input_tokens}, "
            f"previous_requested_output_tokens={adjustment.previous_requested_output_tokens}, "
            f"min_output_tokens={adjustment.min_output_tokens}) and retrying immediately."
        )

    def _served_model_failover_state_for_base(
        self,
        openai_api_base: str,
    ) -> Optional[ServedModelFailoverState]:
        return get_served_model_failover_state(
            openai_api_base,
            self.served_model_name,
            self.served_model_name_fallbacks,
            self.served_model_primary_recovery_interval_sec,
        )

    def _emit_served_model_transition(
        self,
        transition: ServedModelTransition,
        *,
        openai_api_base: str,
        attempt: int,
        reason: str,
    ) -> None:
        emit_trace(
            "served_model_failover",
            layer="agent_llm",
            action=transition.action,
            attempt=attempt,
            base_url=str(openai_api_base),
            configured_served_model=str(self.served_model_name),
            from_model=transition.from_model,
            to_model=transition.to_model,
            reason=reason,
        )
        terminal_print(
            "Served model failover: "
            f"{transition.from_model} -> {transition.to_model} "
            f"({reason}, action={transition.action})."
        )

    def _record_served_model_failure(
        self,
        state: Optional[ServedModelFailoverState],
        selection: ServedModelSelection,
        error: Exception,
        *,
        openai_api_base: str,
        attempt: int,
        error_text: Optional[str] = None,
        eligible_override: Optional[bool] = None,
    ) -> bool:
        eligible = (
            is_served_model_failover_error(error, error_text)
            if eligible_override is None
            else bool(eligible_override)
        )
        if state is None:
            return eligible
        transition = state.failure(selection, eligible=eligible)
        if transition is not None:
            self._emit_served_model_transition(
                transition,
                openai_api_base=openai_api_base,
                attempt=attempt,
                reason=served_model_failover_reason(error, error_text),
            )
        return eligible

    def _record_served_model_success(
        self,
        state: Optional[ServedModelFailoverState],
        selection: ServedModelSelection,
        *,
        openai_api_base: str,
        attempt: int,
    ) -> None:
        if selection.model not in self.served_model_names_used:
            self.served_model_names_used.append(selection.model)
        self.served_model_name_actual = selection.model
        if state is None:
            return
        transition = state.success(selection)
        if transition is not None:
            self._emit_served_model_transition(
                transition,
                openai_api_base=openai_api_base,
                attempt=attempt,
                reason="primary_probe_succeeded",
            )

    def call_server(
        self,
        msgs,
        planning_port,
        max_tries=None,
        tool_schemas=None,
        max_output_tokens=None,
        request_extra_body=None,
        retry_incomplete_output=None,
        request_temperature=None,
        request_presence_penalty=None,
        request_stop=None,
    ) -> Tuple[object, dict[str, int]]:
        del planning_port

        openai_api_key = self._get_api_key()
        openai_api_base = self._get_api_base()
        if not openai_api_key:
            raise RuntimeError(
                "Missing API key. Set OPENAI_API_KEY or API_KEY."
            )
        if not openai_api_base:
            raise RuntimeError(
                "Missing API base. Set OPENAI_API_BASE, API_BASE, or BASE_URL."
            )

        use_responses_api = self._responses_mode_enabled()
        responses_trace_fields = (
            {"responses_replay_mode": "portable"}
            if use_responses_api and self._responses_portable_replay_enabled()
            else {}
        )
        client = None
        if not use_responses_api:
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
                timeout=self.request_timeout,
            )

        max_tries = self.llm_api_max_retries if max_tries is None else max(1, int(max_tries))
        if retry_incomplete_output is None:
            retry_incomplete_output = RETRY_INCOMPLETE_OUTPUT
        base_sleep_time = self.llm_retry_base_sleep_sec
        effective_max_output_tokens = (
            self.max_output_tokens
            if max_output_tokens is None
            else max(1, int(max_output_tokens))
        )
        # In context-compaction mode we rely on the model/backend output cap
        # instead of enforcing a client-side MAX_OUTPUT_TOKENS limit.
        if AGENT_CONTEXT_COMPACTION:
            effective_max_output_tokens = None
        request_extra_body = dict(request_extra_body or {})
        request_tool_schemas = TOOLS_DEEPSEEK_V32 if tool_schemas is None else tool_schemas
        request_tool_schemas = compat_tool_schemas_for_base_url(
            request_tool_schemas,
            openai_api_base,
        )
        local_input_tokens = estimate_request_input_tokens(msgs, request_tool_schemas)
        context_adjustment_count = 0
        context_compact_count = 0
        previous_context_adjustment = None
        failover_state = self._served_model_failover_state_for_base(openai_api_base)
        attempt = 0
        length_attempts: list[dict[str, Any]] = []
        while attempt < max_tries:
            selection = (
                failover_state.select()
                if failover_state is not None
                else ServedModelSelection(
                    self.served_model_name,
                    0,
                    False,
                    self.served_model_name,
                )
            )
            request_model = selection.model
            omitted_sampling_params = (
                self._responses_omitted_sampling_params(request_model)
                if use_responses_api
                else (["top_p"] if self.llm_generate_cfg.get("omit_top_p") else [])
            )
            if selection.primary_probe:
                emit_trace(
                    "served_model_failover",
                    layer="agent_llm",
                    action="primary_recovery_probe",
                    attempt=attempt + 1,
                    base_url=str(openai_api_base),
                    configured_served_model=str(self.served_model_name),
                    from_model=selection.active_model,
                    to_model=selection.model,
                    reason="recovery_interval_elapsed",
                )
                terminal_print(
                    "Served model recovery probe: "
                    f"{selection.active_model} -> {selection.model}."
                )
            try:
                if attempt == 0:
                    # Spread the first wave of requests so the backend is less likely to
                    # return burst-rate limited responses with choices=null.
                    if self.llm_initial_jitter_max_sec > 0:
                        time.sleep(random.uniform(0, self.llm_initial_jitter_max_sec))

                debug_print(
                    f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---"
                )
                request_started_at = time.time()
                emit_trace(
                    "llm_request_start",
                    layer="agent_llm",
                    attempt=attempt + 1,
                    max_attempts=max_tries,
                    base_url=str(openai_api_base),
                    served_model=str(request_model),
                    configured_served_model=str(self.served_model_name),
                    primary_recovery_probe=selection.primary_probe,
                    api_path="/v1/responses" if use_responses_api else "/v1/chat/completions",
                    messages=len(msgs),
                    tools=len(request_tool_schemas or []),
                    local_input_tokens=local_input_tokens,
                    max_tokens=effective_max_output_tokens,
                    omitted_sampling_params=omitted_sampling_params,
                    request_extra_body_keys=sorted(request_extra_body),
                    **responses_trace_fields,
                )
                try:
                    if use_responses_api:
                        response_message, response_usage = self._create_responses_completion(
                            openai_api_key=openai_api_key,
                            openai_api_base=openai_api_base,
                            messages=msgs,
                            tool_schemas=request_tool_schemas,
                            max_output_tokens=effective_max_output_tokens,
                            served_model_name=request_model,
                        )
                    else:
                        deepseek_profile_model = (
                            self._deepseek_v4_profile_model_name(request_model)
                        )
                        request_kwargs = deepseek_v4_request_kwargs(
                            deepseek_profile_model,
                            force_enabled=self._deepseek_v4_force_enabled_for_model(
                                request_model
                            ),
                        )
                        top_k = self.llm_generate_cfg.get("top_k")
                        if top_k not in (None, ""):
                            extra_body = dict(request_kwargs.get("extra_body") or {})
                            extra_body["top_k"] = int(top_k)
                            request_kwargs["extra_body"] = extra_body
                        if request_extra_body:
                            extra_body = dict(request_kwargs.get("extra_body") or {})
                            extra_body.update(copy.deepcopy(request_extra_body))
                            request_kwargs["extra_body"] = extra_body

                        create_kwargs = {
                            "model": request_model,
                            "messages": msgs,
                            "temperature": (
                                self.llm_generate_cfg.get("temperature", 0.6)
                                if request_temperature is None
                                else float(request_temperature)
                            ),
                            "presence_penalty": (
                                self.llm_generate_cfg.get("presence_penalty", 1.1)
                                if request_presence_penalty is None
                                else float(request_presence_penalty)
                            ),
                        }
                        if effective_max_output_tokens is not None:
                            create_kwargs["max_tokens"] = effective_max_output_tokens
                        if request_stop:
                            create_kwargs["stop"] = list(request_stop)
                        if request_tool_schemas:
                            create_kwargs["tools"] = request_tool_schemas
                        if not self.llm_generate_cfg.get("omit_top_p"):
                            create_kwargs["top_p"] = self.llm_generate_cfg.get("top_p", 0.95)
                        create_kwargs.update(request_kwargs)
                        chat_response = client.chat.completions.create(**create_kwargs)
                except Exception as exc:  # noqa: BLE001
                    emit_trace(
                        "llm_request_error",
                        layer="agent_llm",
                        attempt=attempt + 1,
                        max_attempts=max_tries,
                        base_url=str(openai_api_base),
                        served_model=str(request_model),
                        configured_served_model=str(self.served_model_name),
                        duration_s=round(time.time() - request_started_at, 3),
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    raise
                emit_trace(
                    "llm_request_end",
                    layer="agent_llm",
                    attempt=attempt + 1,
                    max_attempts=max_tries,
                    base_url=str(openai_api_base),
                    served_model=str(request_model),
                    configured_served_model=str(self.served_model_name),
                    api_path="/v1/responses" if use_responses_api else "/v1/chat/completions",
                    duration_s=round(time.time() - request_started_at, 3),
                )
                if use_responses_api:
                    self._record_served_model_success(
                        failover_state,
                        selection,
                        openai_api_base=openai_api_base,
                        attempt=attempt + 1,
                    )
                    return response_message, response_usage

                response_error = self._extract_embedded_error(chat_response)
                if response_error:
                    diagnostic = self._format_response_diagnostic(
                        chat_response,
                        response_error,
                    )
                    multimodal_error = build_runtime_multimodal_support_error(
                        msgs,
                        model_path=self.llm_local_path,
                        served_model_name=request_model,
                        error_text=diagnostic,
                        base_url=openai_api_base,
                    )
                    if multimodal_error:
                        raise APIResponseError(multimodal_error, retryable=False)
                    raise APIResponseError(
                        diagnostic,
                        retryable=self._is_retryable_response_error(response_error),
                    )

                choices = getattr(chat_response, "choices", None)
                if not choices:
                    diagnostic = self._format_response_diagnostic(chat_response)
                    raise APIResponseError(
                        "Response does not contain choices. " + diagnostic,
                        retryable=True,
                    )

                first_choice = choices[0]
                response_message = getattr(first_choice, "message", None)
                if response_message is None:
                    diagnostic = self._format_response_diagnostic(chat_response)
                    raise APIResponseError(
                        "Response choice[0].message is empty. " + diagnostic,
                        retryable=True,
                    )

                finish_reason = extract_choice_finish_reason(first_choice)
                if finish_reason == "length":
                    # In compaction mode we accept the model/backend output cap
                    # and do not retry on finish_reason=length.
                    if AGENT_CONTEXT_COMPACTION:
                        response_usage = extract_usage_dict(chat_response)
                        self._record_served_model_success(
                            failover_state,
                            selection,
                            openai_api_base=openai_api_base,
                            attempt=attempt + 1,
                        )
                        return response_message, response_usage
                    response_usage = extract_usage_dict(chat_response)
                    raise IncompleteOutputError(
                        attempt + 1,
                        finish_reason,
                        partial_response=response_message,
                        usage=response_usage,
                        attempt_metadata=[
                            {
                                "attempt": attempt + 1,
                                "finish_reason": finish_reason,
                                "max_output_tokens": effective_max_output_tokens,
                                "duration_sec": round(time.time() - request_started_at, 3),
                                "usage": response_usage,
                            }
                        ],
                        max_output_tokens=effective_max_output_tokens,
                    )

                self._record_served_model_success(
                    failover_state,
                    selection,
                    openai_api_base=openai_api_base,
                    attempt=attempt + 1,
                )
                return response_message, extract_usage_dict(chat_response)

            except IncompleteOutputError as exc:
                self._record_served_model_failure(
                    failover_state,
                    selection,
                    exc,
                    openai_api_base=openai_api_base,
                    attempt=attempt + 1,
                    eligible_override=False,
                )
                current_metadata = copy.deepcopy(exc.attempt_metadata) or [
                    {
                        "finish_reason": exc.finish_reason,
                        "max_output_tokens": effective_max_output_tokens,
                        "usage": dict(exc.usage),
                    }
                ]
                for item in current_metadata:
                    item["attempt"] = attempt + 1
                length_attempts.extend(current_metadata)
                enriched = IncompleteOutputError(
                    attempt + 1,
                    exc.finish_reason,
                    partial_response=exc.partial_response,
                    usage=exc.usage,
                    attempt_metadata=copy.deepcopy(length_attempts),
                    request_messages=copy.deepcopy(msgs),
                    max_output_tokens=exc.max_output_tokens or effective_max_output_tokens,
                )
                if not retry_incomplete_output:
                    terminal_print(f"Warning: {enriched}; keeping partial response as truncated")
                    if enriched.partial_response is not None:
                        self._record_served_model_success(
                            failover_state,
                            selection,
                            openai_api_base=openai_api_base,
                            attempt=attempt + 1,
                        )
                        return TruncatedResponseMessage(enriched.partial_response), enriched.usage
                    raise enriched
                if attempt >= max_tries - 1:
                    terminal_print(f"Error: {enriched}")
                    raise enriched
                terminal_print(
                    "Warning: "
                    f"{enriched}; retrying the same LLM request immediately."
                )
                attempt += 1
                continue

            except APIResponseError as exc:
                response_error_text = str(exc)
                failover_eligible = self._record_served_model_failure(
                    failover_state,
                    selection,
                    exc,
                    openai_api_base=openai_api_base,
                    attempt=attempt + 1,
                    error_text=response_error_text,
                )
                if _chat_tools_with_reasoning_effort_error(str(exc)):
                    diagnostic = self._format_responses_required_error(
                        openai_api_base,
                        request_model,
                    )
                    terminal_print(diagnostic)
                    raise RuntimeError(diagnostic) from exc
                if self._maybe_downgrade_multimodal_tool_messages(
                    str(exc),
                    request_messages=msgs,
                ):
                    terminal_print(
                        "Tool multimodal response rejected by backend; "
                        "downgraded visual tool observations to text and retrying."
                    )
                    max_tries += 1
                    attempt += 1
                    continue
                retry_kind = "retryable" if exc.retryable else "fatal"
                terminal_print(
                    "Error: Attempt "
                    f"{attempt + 1} failed with an API response error ({retry_kind}): {exc}"
                )

                compactor = _get_context_compactor()
                if compactor is not None and parse_context_length_error(response_error_text):
                    summary_client = client or OpenAI(
                        api_key=openai_api_key,
                        base_url=openai_api_base,
                        timeout=self.request_timeout,
                    )
                    compacted = compactor.compact(
                        msgs,
                        response_error_text,
                        context_compact_count,
                        summary_client,
                        request_model,
                        request_tool_schemas,
                        self.request_timeout,
                    )
                    if compacted is not None:
                        msgs = compacted
                        local_input_tokens = estimate_request_input_tokens(
                            msgs, request_tool_schemas
                        )
                        context_compact_count += 1
                        max_tries += 1
                        attempt += 1
                        terminal_print(
                            "Context compacted "
                            f"({context_compact_count}); retrying with {len(msgs)} messages, "
                            f"~{local_input_tokens} input tokens."
                        )
                        continue
                    terminal_print(
                        "Context compaction failed or exhausted; raising context-length error."
                    )
                    raise RuntimeError(
                        "Context-length error and compaction exhausted or ineffective."
                    ) from exc

                try:
                    adjustment = self._context_length_adjustment_or_raise(
                        str(exc),
                        effective_max_output_tokens,
                        context_adjustment_count,
                        local_input_tokens,
                        previous_context_adjustment,
                    )
                except RuntimeError as context_exc:
                    terminal_print(str(context_exc))
                    raise RuntimeError(str(context_exc)) from exc
                if adjustment is not None:
                    previous_max_output_tokens = effective_max_output_tokens
                    effective_max_output_tokens = adjustment.adjusted_output_tokens
                    previous_context_adjustment = adjustment
                    context_adjustment_count += 1
                    max_tries += 1
                    self._log_context_adjustment_retry(
                        adjustment,
                        context_adjustment_count,
                        previous_max_output_tokens,
                        effective_max_output_tokens,
                    )
                    attempt += 1
                    continue
                if not exc.retryable and not (
                    failover_state is not None and failover_eligible
                ):
                    raise

            except APIError as exc:
                error_text = self._extract_api_error_text(exc)
                failover_eligible = self._record_served_model_failure(
                    failover_state,
                    selection,
                    exc,
                    openai_api_base=openai_api_base,
                    attempt=attempt + 1,
                    error_text=error_text,
                )
                if _chat_tools_with_reasoning_effort_error(error_text):
                    diagnostic = self._format_responses_required_error(
                        openai_api_base,
                        request_model,
                    )
                    terminal_print(diagnostic)
                    raise RuntimeError(diagnostic) from exc
                if self._maybe_downgrade_multimodal_tool_messages(
                    error_text,
                    request_messages=msgs,
                ):
                    terminal_print(
                        "Tool multimodal response rejected by backend; "
                        "downgraded visual tool observations to text and retrying."
                    )
                    max_tries += 1
                    attempt += 1
                    continue
                if _chat_tools_with_reasoning_effort_error(str(exc)):
                    diagnostic = self._format_responses_required_error(
                        openai_api_base,
                        request_model,
                    )
                    terminal_print(diagnostic)
                    raise RuntimeError(diagnostic) from exc
                multimodal_error = build_runtime_multimodal_support_error(
                    msgs,
                    model_path=self.llm_local_path,
                    served_model_name=request_model,
                    error_text=error_text,
                    base_url=openai_api_base,
                )
                if multimodal_error:
                    terminal_print(multimodal_error)
                    raise RuntimeError(multimodal_error) from exc
                if getattr(exc, "status_code", None) == 404:
                    if failover_state is None or not failover_eligible:
                        diagnostic = self._format_model_resolution_error(
                            openai_api_base,
                            request_model,
                            exc,
                        )
                        terminal_print(diagnostic)
                        raise RuntimeError(diagnostic) from exc
                if getattr(exc, "status_code", None) == 400:
                    compactor = _get_context_compactor()
                    if compactor is not None and parse_context_length_error(error_text):
                        summary_client = client or OpenAI(
                            api_key=openai_api_key,
                            base_url=openai_api_base,
                            timeout=self.request_timeout,
                        )
                        compacted = compactor.compact(
                            msgs,
                            error_text,
                            context_compact_count,
                            summary_client,
                            request_model,
                            request_tool_schemas,
                            self.request_timeout,
                        )
                        if compacted is not None:
                            msgs = compacted
                            local_input_tokens = estimate_request_input_tokens(
                                msgs, request_tool_schemas
                            )
                            context_compact_count += 1
                            max_tries += 1
                            attempt += 1
                            terminal_print(
                                "Context compacted "
                                f"({context_compact_count}); retrying with {len(msgs)} messages, "
                                f"~{local_input_tokens} input tokens."
                            )
                            continue
                        terminal_print(
                            "Context compaction failed or exhausted; raising context-length error."
                        )
                        raise RuntimeError(
                            "Context-length error and compaction exhausted or ineffective."
                        ) from exc

                    try:
                        adjustment = self._context_length_adjustment_or_raise(
                            error_text,
                            effective_max_output_tokens,
                            context_adjustment_count,
                            local_input_tokens,
                            previous_context_adjustment,
                        )
                    except RuntimeError as context_exc:
                        terminal_print(str(context_exc))
                        raise RuntimeError(str(context_exc)) from exc
                    if adjustment is not None:
                        previous_max_output_tokens = effective_max_output_tokens
                        effective_max_output_tokens = adjustment.adjusted_output_tokens
                        previous_context_adjustment = adjustment
                        context_adjustment_count += 1
                        max_tries += 1
                        self._log_context_adjustment_retry(
                            adjustment,
                            context_adjustment_count,
                            previous_max_output_tokens,
                            effective_max_output_tokens,
                        )
                        attempt += 1
                        continue
                terminal_print(
                    f"Error: Attempt {attempt + 1} failed with an API or network error: {exc}"
                )
            except (APIConnectionError, APITimeoutError) as exc:
                self._record_served_model_failure(
                    failover_state,
                    selection,
                    exc,
                    openai_api_base=openai_api_base,
                    attempt=attempt + 1,
                )
                terminal_print(
                    f"Error: Attempt {attempt + 1} failed with an API or network error: {exc}"
                )
            except Exception as exc:
                self._record_served_model_failure(
                    failover_state,
                    selection,
                    exc,
                    openai_api_base=openai_api_base,
                    attempt=attempt + 1,
                )
                if self._maybe_downgrade_multimodal_tool_messages(
                    str(exc),
                    request_messages=msgs,
                ):
                    terminal_print(
                        "Tool multimodal response rejected by backend; "
                        "downgraded visual tool observations to text and retrying."
                    )
                    max_tries += 1
                    attempt += 1
                    continue
                multimodal_error = build_runtime_multimodal_support_error(
                    msgs,
                    model_path=self.llm_local_path,
                    served_model_name=request_model,
                    error_text=str(exc),
                    base_url=openai_api_base,
                )
                if multimodal_error:
                    terminal_print(multimodal_error)
                    raise RuntimeError(multimodal_error) from exc
                terminal_print(
                    f"Error: Attempt {attempt + 1} failed with an unexpected error: {exc}"
                )

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, self.llm_retry_max_sleep_sec)
                terminal_print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                terminal_print("Error: All retry attempts have been exhausted. The call has failed.")
            attempt += 1

        raise RuntimeError(
            "api server error after "
            f"{max_tries} attempts; base_url={openai_api_base}, "
            f"requested_model={self.served_model_name}"
        )

    def count_tokens(self, messages):
        """Count the request tokens used for context management.

        Prefer the model's local chat-template tokenizer so the count includes
        the actual role/tool formatting. Fall back to the API request estimator
        when the tokenizer is unavailable; context management must remain best
        effort rather than breaking a rollout during tokenizer initialization.
        """
        try:
            tool_schemas = self._active_tool_schemas_for_current_task()
        except Exception:
            tool_schemas = None
        if not self._context_tokenizer_unavailable and self._context_tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._context_tokenizer = AutoTokenizer.from_pretrained(
                    self.llm_local_path,
                    local_files_only=True,
                    trust_remote_code=True,
                )
            except Exception:
                self._context_tokenizer_unavailable = True
        if self._context_tokenizer is not None:
            try:
                kwargs = {
                    "tokenize": True,
                    "add_generation_prompt": True,
                }
                if tool_schemas:
                    kwargs["tools"] = tool_schemas
                encoded = self._context_tokenizer.apply_chat_template(
                    messages,
                    **kwargs,
                )
                if isinstance(encoded, dict):
                    encoded = encoded.get("input_ids", [])
                if hasattr(encoded, "shape"):
                    return int(encoded.shape[-1])
                return len(encoded)
            except Exception:
                pass
        return estimate_request_input_tokens(messages, tool_schemas)

    def _estimate_trajectory_tokens(self, messages):
        return self.count_tokens(messages)

    def _append_budget_notice(self, content, messages):
        used_tokens = self._estimate_trajectory_tokens(
            messages + [{"role": "tool", "content": content}]
        )
        remaining = max(0, TRAJECTORY_MAX_TOKENS - used_tokens)
        notice = (
            "\n\n[Budget] Approximately "
            f"{remaining} tokens remain in the total trajectory budget. "
            "Keep the next response concise and provide the final answer before the budget is exhausted."
        )
        if isinstance(content, str):
            return content + notice
        if isinstance(content, list):
            return [*content, {"type": "text", "text": notice}]
        return f"{tool_content_to_text(content)}{notice}"

    def get_tool_related_msg_indices(self, messages):
        tool_related_indices = []
        for idx in range(2, len(messages)):
            tool_related_indices.append(idx)
        return tool_related_indices

    def discard_75_percent(self, messages):
        tool_indices = self.get_tool_related_msg_indices(messages)
        if not tool_indices:
            return messages

        keep_num = max(1, int(len(tool_indices) * 0.25))
        keep_indices = tool_indices[-keep_num:]

        new_messages = messages[:2]
        for idx in keep_indices:
            new_messages.append(messages[idx])
        return new_messages

    def discard_all(self, messages):
        return messages[:2]

    def _build_assistant_request_message(
        self,
        reasoning: str,
        content: str,
        serialized_tool_calls,
        response_output_items=None,
    ) -> Dict:
        message = {
            "role": "assistant",
            "reasoning": reasoning,
            "content": content,
            "tool_calls": copy.deepcopy(serialized_tool_calls),
        }
        # Some OpenAI-compatible thinking endpoints require the exact
        # `reasoning_content` key when replaying an assistant tool-call turn.
        if not AGENT_OBSERVATION_MASKING or reasoning or self._deepseek_v4_response_enabled():
            message["reasoning_content"] = reasoning or ""
        if isinstance(response_output_items, list) and response_output_items:
            message["_responses_output"] = copy.deepcopy(response_output_items)
        return message

    def _schema_tool_name(self, tool_name: str) -> str:
        return tool_name

    def _normalize_serialized_tool_calls(self, serialized_tool_calls):
        if not isinstance(serialized_tool_calls, list):
            return serialized_tool_calls
        normalized = copy.deepcopy(serialized_tool_calls)
        for tool_call in normalized:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if isinstance(name, str) and name.strip():
                function["name"] = self._schema_tool_name(name)
        return normalized

    def _recover_text_tool_calls(
        self,
        content: str,
        reasoning: str,
        round_idx: int,
    ):
        parsed, cleaned_content = extract_text_tool_calls(content)
        cleaned_reasoning = reasoning
        source_field = "content"
        if not parsed:
            parsed, cleaned_reasoning = extract_text_tool_calls(reasoning)
            source_field = "reasoning"
        if not parsed:
            return None, content, reasoning

        recovered = []
        for index, payload in enumerate(parsed):
            name = self._schema_tool_name(str(payload["name"]))
            arguments = json.dumps(
                payload["arguments"],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(
                f"{round_idx}:{index}:{name}:{arguments}".encode("utf-8")
            ).hexdigest()[:24]
            recovered.append(
                SimpleNamespace(
                    type="function",
                    id=f"call_text_{digest}",
                    function=SimpleNamespace(name=name, arguments=arguments),
                )
            )
        emit_trace(
            "text_tool_calls_recovered",
            layer="agent_loop",
            round_idx=round_idx,
            source_field=source_field,
            tool_calls=len(recovered),
            tool_names=[call.function.name for call in recovered],
        )
        return recovered, cleaned_content, cleaned_reasoning

    def _clean_limit_apology(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = text.strip()
        if not cleaned:
            return ""
        cleaned = re.sub(
            r"(?:\s|\r|\n)*Sorry,\s*the number of llm calls exceeds the limit\.?\s*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    def _extract_answer_block(self, text: str) -> str:
        cleaned = self._clean_limit_apology(text)
        if "<answer>" in cleaned and "</answer>" in cleaned:
            return cleaned.split("<answer>")[-1].split("</answer>")[0].strip()
        return ""

    def _extract_last_boxed(self, text: str) -> str:
        cleaned = self._clean_limit_apology(text)
        spans = []
        index = 0
        while index < len(cleaned):
            match = re.search(r"\\?boxed\s*\{", cleaned[index:])
            if not match:
                break
            start = index + match.start()
            brace = cleaned.find("{", start)
            if brace == -1:
                break
            depth = 0
            pos = brace
            while pos < len(cleaned):
                if cleaned[pos] == "{":
                    depth += 1
                elif cleaned[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        content = cleaned[brace + 1 : pos].strip()
                        if content:
                            spans.append(f"\\boxed{{{content}}}")
                        index = pos + 1
                        break
                pos += 1
            else:
                break
        return spans[-1] if spans else ""

    def _extract_final_answer_block(self, text: str) -> str:
        cleaned = self._clean_limit_apology(text)
        matches = list(
            re.finditer(r"final[_ ]answer\s*:?\s*", cleaned, flags=re.IGNORECASE)
        )
        if not matches:
            return ""
        return cleaned[matches[-1].end() :].strip()

    def _extract_terminal_answer(self, text: str) -> str:
        answer_block = self._extract_answer_block(text)
        if answer_block:
            return answer_block

        boxed = self._extract_last_boxed(text)
        if boxed:
            return boxed

        return self._extract_final_answer_block(text)

    def _last_substantive_assistant_content(self, messages) -> str:
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            cleaned = self._clean_limit_apology(
                safe_strip(message.get("content", ""))
            )
            if cleaned:
                return cleaned
        return ""

    def _question_requires_boxed(self, question: str) -> bool:
        lowered = str(question).lower()
        return "\\boxed" in lowered or "boxed" in lowered

    def _build_forced_finalization_user_message(
        self,
        question: str,
        reason: str,
    ) -> dict:
        if self._question_requires_boxed(question):
            format_line = (
                "Inside <answer>, keep the required \\boxed{...} format for the final answer."
            )
        else:
            format_line = (
                "Inside <answer>, preserve any benchmark-specific final-answer format already requested."
            )
        return {
            "role": "user",
            "content": (
                "Tool-use budget is exhausted or nearly exhausted for this problem. "
                "Do not call any more tools. Based only on the information already gathered, "
                "provide your best final response now.\n\n"
                "Requirements:\n"
                "1. Do not produce another reasoning trace or a <think> section; the prior "
                "reasoning is already preserved.\n"
                "2. Return only the final response inside <answer></answer>.\n"
                f"3. {format_line}\n"
                "4. Keep the final response under 800 words. State only the result and the "
                "minimum equations or evidence needed to make it self-contained; do not "
                "repeat the research history, tool outputs, or alternative derivations.\n"
                "5. Do not apologize or mention tool limits.\n"
                "6. If tools were used and their observations contain source URLs, "
                "cite supported claims with [^n^] and append matching [^n^]: URL "
                "definitions using only URLs that were actually observed.\n"
                f"7. Trigger for this instruction: {reason}."
            ),
        }

    def _attempt_forced_finalization(
        self,
        passin_messages,
        planning_port,
        question: str,
        reason: str,
    ) -> Tuple[Optional[dict], dict[str, int]]:
        if not AGENT_EVAL_FORCE_FINALIZE:
            return None, {}

        final_messages = copy.deepcopy(passin_messages)
        final_messages.append(
            self._build_forced_finalization_user_message(question, reason)
        )
        partial_usage: dict[str, int] = {}
        try:
            forced_max_output_tokens = (
                min(self.max_output_tokens, AGENT_EVAL_FORCE_FINAL_MAX_OUTPUT_TOKENS)
                if AGENT_EVAL_FORCE_FINAL_MAX_OUTPUT_TOKENS > 0
                else self.max_output_tokens
            )
            forced_extra_body = (
                {"thinking": {"type": "disabled"}}
                if AGENT_EVAL_FORCE_FINAL_DISABLE_THINKING
                else None
            )
            response, usage = self.call_server(
                self._observation_masked_view(final_messages),
                planning_port,
                tool_schemas=[],
                max_output_tokens=forced_max_output_tokens,
                request_extra_body=forced_extra_body,
                retry_incomplete_output=False,
                request_temperature=0.0,
                request_presence_penalty=0.0,
            )
        except IncompleteOutputError as exc:
            partial_response = exc.partial_response
            partial_reasoning, partial_content = (
                self._extract_response_text_fields(partial_response)
                if partial_response is not None
                else ("", "")
            )
            emit_trace(
                "forced_finalization_incomplete",
                layer="agent_loop",
                reason=reason,
                max_output_tokens=exc.max_output_tokens,
                reasoning_chars=len(partial_reasoning),
                content_chars=len(partial_content),
                has_answer_open="<answer>" in partial_content,
                has_answer_close="</answer>" in partial_content,
                reasoning_has_answer_open="<answer>" in partial_reasoning,
                reasoning_has_answer_close="</answer>" in partial_reasoning,
            )
            recovered_answer = ""
            for candidate in (partial_content, partial_reasoning):
                answer_start = candidate.find("<answer>")
                answer_end = candidate.find(
                    "</answer>",
                    answer_start + len("<answer>"),
                )
                if answer_start >= 0 and answer_end >= 0:
                    recovered_answer = candidate[
                        answer_start : answer_end + len("</answer>")
                    ]
                    break
            if recovered_answer:
                response = SimpleNamespace(
                    content=recovered_answer,
                    reasoning_content="",
                    tool_calls=None,
                )
                usage = dict(exc.usage)
                emit_trace(
                    "forced_finalization_partial_recovered",
                    layer="agent_loop",
                    reason=reason,
                    content_chars=len(recovered_answer),
                )
            else:
                merge_usage_dict(partial_usage, exc.usage)
                draft = (partial_reasoning or partial_content).strip()
                if not draft:
                    raise
                draft = draft[-AGENT_EVAL_FORCE_FINAL_RECOVERY_DRAFT_CHARS:]
                recovery_messages = [
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": draft,
                    },
                    {
                        "role": "user",
                        "content": (
                            "The analysis above is already complete. Do not restart or extend "
                            "the derivation and do not call tools. Return only a self-contained "
                            "final response inside <answer></answer>, using at most 600 words. "
                            "Preserve any required benchmark answer format and only retain "
                            "citations whose URLs already appear in the analysis."
                        ),
                    },
                ]
                emit_trace(
                    "forced_finalization_recovery_start",
                    layer="agent_loop",
                    reason=reason,
                    draft_chars=len(draft),
                    max_output_tokens=(
                        AGENT_EVAL_FORCE_FINAL_RECOVERY_MAX_OUTPUT_TOKENS
                    ),
                )
                response, recovery_usage = self.call_server(
                    recovery_messages,
                    planning_port,
                    tool_schemas=[],
                    max_output_tokens=(
                        AGENT_EVAL_FORCE_FINAL_RECOVERY_MAX_OUTPUT_TOKENS
                    ),
                    request_extra_body={
                        "thinking": {"type": "disabled"},
                    },
                    retry_incomplete_output=False,
                    request_temperature=0.0,
                    request_presence_penalty=0.0,
                )
                merge_usage_dict(partial_usage, recovery_usage)
                usage = partial_usage
                emit_trace(
                    "forced_finalization_recovery_end",
                    layer="agent_loop",
                    reason=reason,
                )
        except Exception as exc:
            terminal_print(
                f"Warning: forced finalization failed after {reason}: {exc}"
            )
            return None, {}

        reasoning, content = self._extract_response_text_fields(response)
        content = self._clean_limit_apology(content)
        answer_start = content.find("<answer>")
        answer_end = content.find("</answer>", answer_start + len("<answer>"))
        if answer_start >= 0:
            if answer_end >= 0:
                content = content[answer_start : answer_end + len("</answer>")]
            else:
                content = content[answer_start:] + "</answer>"
        elif content:
            content = f"<answer>{content}</answer>"
        tool_calls = getattr(response, "tool_calls", None)
        if not content and not reasoning:
            return None, usage

        serialized_tool_calls = (
            [
                {
                    "type": tool_call.type,
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in tool_calls
            ]
            if tool_calls is not None
            else None
        )
        serialized_tool_calls = self._normalize_serialized_tool_calls(serialized_tool_calls)
        return {
            "role": "assistant",
            "reasoning": reasoning,
            "content": content,
            "tool_calls": serialized_tool_calls,
        }, usage

    @sandbox_run_scope
    def _run(self, data: str, served_model_name: Optional[str] = None, **kwargs) -> List[List[Message]]:
        del kwargs
        if served_model_name:
            self.served_model_name = served_model_name
        self.served_model_name_fallbacks = _parse_served_model_fallbacks(
            self._served_model_fallbacks_raw,
            self.served_model_name,
        )
        self.served_model_names_used = []
        self.served_model_name_actual = None
        try:
            question = data["item"]["question"]
        except Exception:
            raw_msg = data["item"]["messages"][1]["content"]
            question = raw_msg.split("User:", 1)[1].strip() if "User:" in raw_msg else raw_msg

        start_time = time.time()
        planning_port = data.get("planning_port")
        # answer = data["item"].get("answer", "")
        answer = data['item'].get('answer', data['item'].get('ground_truth', ''))
        self.user_prompt = question
        item_base_dir = data["item"].get("dataset_base_dir") or os.getcwd()

        user_message = build_initial_user_message(
            data["item"],
            question,
            model_path=self.llm_local_path,
            served_model_name=self.served_model_name,
            base_dir=item_base_dir,
        )
        tool_context_hint = self._configure_task_tools(
            data["item"],
            item_base_dir,
            data,
            user_message=user_message,
        )
        if tool_context_hint:
            self._append_text_to_user_message(
                user_message,
                "\n\n" + tool_context_hint,
            )
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            user_message,
        ]
        passin_messages = copy.deepcopy(messages)
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round_idx = 0
        tool_call_count = 0
        prediction = None
        termination = None
        response_was_truncated = False
        trajectory_budget_exhausted = False
        usage_totals: dict[str, int] = {}
        has_usage = False

        def finish(result: dict) -> dict:
            prediction = str(result.get("prediction") or "").strip()
            if prediction.casefold() in {"your answer", "your final answer"}:
                prediction = ""
            result["prediction"] = prediction
            result["generation_truncated"] = response_was_truncated
            result["trajectory_budget_exhausted"] = trajectory_budget_exhausted
            result["final_answer_valid"] = bool(prediction)
            result["truncated"] = response_was_truncated or trajectory_budget_exhausted
            if trajectory_budget_exhausted and not prediction:
                result["termination"] = "trajectory budget exhausted without valid final answer"
            return self._attach_usage_fields(result, usage_totals, has_usage)

        while num_llm_calls_available > 0:
            if time.time() - start_time > self.task_timeout_sec:
                return finish({
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": "",
                    "termination": "task timeout",
                })

            round_idx += 1
            num_llm_calls_available -= 1
            self._set_fallback_message_lists(messages, passin_messages)
            request_messages = self._observation_masked_view(passin_messages)
            response, usage = self.call_server(
                request_messages,
                planning_port,
                tool_schemas=self._active_tool_schemas_for_current_task(),
            )
            has_usage = merge_usage_dict(usage_totals, usage) or has_usage
            debug_print(response)
            reasoning, content = self._extract_response_text_fields(response)
            response_truncated = bool(getattr(response, "_molt_truncated", False))
            response_was_truncated = response_was_truncated or response_truncated
            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                tool_calls, content, reasoning = self._recover_text_tool_calls(
                    content,
                    reasoning,
                    round_idx,
                )
            debug_print(f"Round {round_idx}\n{reasoning=}\n{content=}\n{tool_calls=}")

            serialized_tool_calls = (
                [
                    {
                        "type": tool_call.type,
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ]
                if tool_calls is not None
                else None
            )
            serialized_tool_calls = self._normalize_serialized_tool_calls(serialized_tool_calls)

            if not content and not reasoning and not serialized_tool_calls:
                terminal_print(
                    "Warning: empty assistant response after stripping special "
                    "tokens; stopping current task."
                )
                return finish({
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": "",
                    "termination": "empty model response",
                })

            messages.append(
                {
                    "role": "assistant",
                    "reasoning": reasoning,
                    "content": content,
                    "tool_calls": serialized_tool_calls,
                }
            )
            passin_messages.append(
                self._build_assistant_request_message(
                    reasoning,
                    content,
                    serialized_tool_calls,
                    getattr(response, "_responses_output", None),
                )
            )

            if tool_calls is not None and len(tool_calls) > 0:
                turn_tool_messages = []
                turn_tool_names = []
                turn_tool_args = []
                turn_tool_responses = []
                for tool in tool_calls:
                    raw_tool_args = ""
                    try:
                        tool_name = tool.function.name
                        raw_tool_args = tool.function.arguments
                        tool_args = json_parser.loads(raw_tool_args)
                        tool_started_at = time.time()
                        trace_tool_args = {
                            "arguments_chars": len(str(raw_tool_args)),
                            "arguments_sha256": _tool_arguments_hash(raw_tool_args),
                        }
                        trace_tool_args.update(
                            _python_tool_code_preview(tool_name, tool_args, raw_tool_args)
                        )
                        emit_trace(
                            "model_tool_call_start",
                            layer="agent_loop",
                            round_idx=round_idx,
                            tool_call_id=getattr(tool, "id", ""),
                            tool=tool_name,
                            **trace_tool_args,
                        )
                        tool_response = self.custom_call_tool(tool_name, tool_args)
                        emit_trace(
                            "model_tool_call_end",
                            layer="agent_loop",
                            round_idx=round_idx,
                            tool_call_id=getattr(tool, "id", ""),
                            tool=tool_name,
                            duration_s=round(time.time() - tool_started_at, 3),
                            response_chars=len(str(tool_response)),
                        )
                    except Exception as exc:
                        tool_name = getattr(getattr(tool, "function", None), "name", "")
                        tool_args = raw_tool_args
                        tool_response = "Error: " + str(exc)
                        emit_trace(
                            "model_tool_call_error",
                            layer="agent_loop",
                            round_idx=round_idx,
                            tool_call_id=getattr(tool, "id", ""),
                            tool=tool_name,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                    schema_name = (
                        self._schema_tool_name(tool_name)
                        if hasattr(self, "_schema_tool_name")
                        else tool_name
                    )
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool.id,
                        "name": schema_name,
                        "content": self._append_budget_notice(tool_response, passin_messages),
                    }
                    tool_response_text = tool_content_to_text(tool_response)
                    turn_tool_names.append(schema_name)
                    turn_tool_args.append(copy.deepcopy(tool_args))
                    turn_tool_responses.append(tool_response_text)
                    turn_tool_messages.append(copy.deepcopy(tool_message))
                    messages.append(tool_message)
                    passin_messages.append(copy.deepcopy(tool_message))
                tool_call_count += len(tool_calls)

            else:
                terminal_answer = self._extract_terminal_answer(content)
                if terminal_answer:
                    prediction = terminal_answer
                    termination = "answer"
                    break

                # Without a structured final answer, return an empty prediction
                # so the reward adapter can skip the judge and assign 0.
                prediction = ""
                termination = "answer not found"
                break

                if (
                    not AGENT_EVAL_FORCE_FINALIZE
                    and num_llm_calls_available <= 0
                    and "<answer>" not in content
                ):
                    messages[-1]["content"] = "Sorry, the number of llm calls exceeds the limit."

            finalization_reason = ""
            if (
                MAX_TOOL_CALLS_PER_RUN > 0
                and tool_call_count >= MAX_TOOL_CALLS_PER_RUN
            ):
                num_llm_calls_available = 0
                finalization_reason = "tool call budget reached"
            elif num_llm_calls_available <= 0:
                finalization_reason = "llm call budget reached"

            if (
                AGENT_EVAL_FORCE_FINALIZE
                and finalization_reason
                and "<answer>" not in content
            ):
                forced_message, forced_usage = self._attempt_forced_finalization(
                    passin_messages,
                    planning_port,
                    question,
                    reason=finalization_reason,
                )
                has_usage = merge_usage_dict(usage_totals, forced_usage) or has_usage
                if forced_message is not None:
                    messages.append(forced_message)
                    passin_messages.append(
                        self._build_assistant_request_message(
                            forced_message.get("reasoning", ""),
                            forced_message.get("content", ""),
                            forced_message.get("tool_calls"),
                        )
                    )
                final_content = self._last_substantive_assistant_content(messages)
                prediction = self._extract_terminal_answer(final_content) or ""
                termination = (
                    f"forced finalization after {finalization_reason}"
                    if forced_message is not None
                    else "exceed available llm calls"
                )
                return finish({
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                })

            max_tokens = TRAJECTORY_MAX_TOKENS
            request_messages = self._observation_masked_view(passin_messages)
            token_count = self.count_tokens(request_messages)
            debug_print(f"round: {round_idx}, token count: {token_count}")

            if token_count > max_tokens:
                if CONTEXT_MANAGEMENT_STRATEGY == "none":
                    trajectory_budget_exhausted = True
                    terminal_print(f"Warning: token quantity exceeds the limit: {token_count} > {max_tokens}")

                    if AGENT_EVAL_FORCE_FINALIZE:
                        forced_message, forced_usage = self._attempt_forced_finalization(
                            passin_messages,
                            planning_port,
                            question,
                            reason="context token limit reached",
                        )
                        has_usage = (
                            merge_usage_dict(usage_totals, forced_usage) or has_usage
                        )
                        if forced_message is not None:
                            messages.append(forced_message)
                            passin_messages.append(
                                self._build_assistant_request_message(
                                    forced_message.get("reasoning", ""),
                                    forced_message.get("content", ""),
                                    forced_message.get("tool_calls"),
                                )
                            )
                        final_content = self._last_substantive_assistant_content(messages)
                        prediction = self._extract_terminal_answer(final_content) or ""
                        termination = (
                            "forced finalization as token limit reached"
                            if forced_message is not None
                            else "token limit reached without final answer"
                        )
                        return finish({
                            "question": question,
                            "answer": answer,
                            "messages": messages,
                            "prediction": prediction,
                            "termination": termination,
                        })

                    messages[-1]["content"] = (
                        "You have now reached the maximum context length you can handle. "
                        "You should stop making tool calls and, based on all the "
                        "information above, think again and provide what you consider "
                        "the most likely actual answer enclosed in <answer> and </answer> tags. "
                        "Do not copy template placeholders or make further tool calls."
                    )
                    passin_messages[-1]["content"] = messages[-1]["content"]
                    self._set_fallback_message_lists(messages, passin_messages)
                    choice, usage = self.call_server(
                        self._observation_masked_view(passin_messages),
                        planning_port,
                        tool_schemas=self._active_tool_schemas_for_current_task(),
                    )
                    has_usage = merge_usage_dict(usage_totals, usage) or has_usage
                    reasoning, content = self._extract_response_text_fields(choice)
                    response_was_truncated = response_was_truncated or bool(getattr(choice, "_molt_truncated", False))
                    tool_calls = getattr(choice, "tool_calls", None)
                    serialized_tool_calls = (
                        [
                            {
                                "type": tool_call.type,
                                "id": tool_call.id,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in tool_calls
                        ]
                        if tool_calls is not None
                        else None
                    )
                    serialized_tool_calls = self._normalize_serialized_tool_calls(serialized_tool_calls)

                    messages.append(
                        {
                            "role": "assistant",
                            "reasoning": reasoning,
                            "content": content,
                            "tool_calls": serialized_tool_calls,
                        }
                    )
                    passin_messages.append(
                        self._build_assistant_request_message(
                            reasoning,
                            content,
                            serialized_tool_calls,
                            getattr(choice, "_responses_output", None),
                        )
                    )

                    terminal_answer = self._extract_terminal_answer(content)
                    if terminal_answer:
                        prediction = terminal_answer
                        termination = "generate an answer as token limit reached"
                    else:
                        prediction = ""
                        termination = "format error: generate an answer as token limit reached"
                    return finish({
                        "question": question,
                        "answer": answer,
                        "messages": messages,
                        "prediction": prediction,
                        "termination": termination,
                    })
                elif CONTEXT_MANAGEMENT_STRATEGY == "discard_75":
                    passin_messages = self.discard_75_percent(passin_messages)
                    new_token_count = self.count_tokens(passin_messages)
                    debug_print(f"token count after discarding 75%：{new_token_count} ({token_count})")
                    continue
                elif CONTEXT_MANAGEMENT_STRATEGY == "discard_all":
                    passin_messages = self.discard_all(passin_messages)
                    new_token_count = self.count_tokens(passin_messages)
                    debug_print(f"token count after discarding all：{new_token_count} ({token_count})")
                    continue
                else:
                    raise ValueError("Invalid CONTEXT_MANAGEMENT_STRATEGY value.")

        if termination == "answer" and prediction is not None:
            return finish({
                "question": question,
                "answer": answer,
                "messages": messages,
                "prediction": prediction,
                "termination": termination,
            })

        extracted_answer = self._last_substantive_assistant_content(messages)
        terminal_answer = self._extract_terminal_answer(extracted_answer)
        if terminal_answer:
            prediction = terminal_answer
            termination = "answer"
        else:
            # Do not promote unstructured trailing assistant text to a
            # prediction. The reward adapter treats an empty prediction as an
            # invalid answer and assigns reward 0 without calling the judge.
            prediction = ""
            termination = "answer not found"
            if num_llm_calls_available == 0:
                termination = "exceed available llm calls"

        return finish({
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
        })

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
            return raw_result if isinstance(raw_result, (str, list)) else str(raw_result)
        return f"Error: Tool {tool_name} not found"
