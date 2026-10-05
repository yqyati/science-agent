import os
import re
from typing import Any, Dict, NamedTuple, Optional


class DeepSeekV4Profile(NamedTuple):
    family: str
    reasoning_effort: str
    is_local_alias: bool


class DeepSeekV4ResponseText(NamedTuple):
    reasoning_content: str
    content: str
    fallback_split_used: bool


_EFFORT_MODEL_RE = re.compile(
    r"deepseek-v4-(pro|flash)-(max|high)",
    flags=re.IGNORECASE,
)
_OFFICIAL_MODEL_RE = re.compile(
    r"deepseek-v4-(pro|flash)",
    flags=re.IGNORECASE,
)
_LEADING_EMPTY_THINK_END_RE = re.compile(
    r"^(?:\s*</think>\s*)+",
    flags=re.IGNORECASE,
)
_THINK_END_RE = re.compile(r"</think>", flags=re.IGNORECASE)
_THINK_START_RE = re.compile(r"^\s*<think>\s*", flags=re.IGNORECASE)
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"", "0", "false", "no", "off"}
_SUPPORTED_REASONING_EFFORTS = {"high", "max"}
DEEPSEEK_V4_REASONING_ENV = "SCIENCE_AGENT_DEEPSEEK_V4_REASONING"
DEEPSEEK_V4_REASONING_EFFORT_ENV = "SGLANG_DSV4_REASONING_EFFORT"


def normalize_model_name_for_profile(model_name: str) -> str:
    value = str(model_name or "").strip()
    value = value.rsplit("/", 1)[-1]
    value = os.path.basename(value.rstrip("/"))
    return value.lower()


def parse_deepseek_v4_profile(model_name: str) -> Optional[DeepSeekV4Profile]:
    normalized = normalize_model_name_for_profile(model_name)

    effort_match = _EFFORT_MODEL_RE.search(normalized)
    if effort_match:
        family, reasoning_effort = effort_match.groups()
        return DeepSeekV4Profile(
            family=family.lower(),
            reasoning_effort=reasoning_effort.lower(),
            is_local_alias=True,
        )

    official_match = _OFFICIAL_MODEL_RE.search(normalized)
    if official_match:
        family = official_match.group(1)
        return DeepSeekV4Profile(
            family=family.lower(),
            reasoning_effort="max",
            is_local_alias=False,
        )

    return None


def is_deepseek_v4_model(model_name: str) -> bool:
    return parse_deepseek_v4_profile(model_name) is not None


def deepseek_v4_reasoning_opt_in_enabled(value: Any = None) -> bool:
    raw_value = (
        os.getenv(DEEPSEEK_V4_REASONING_ENV, "")
        if value is None
        else str(value)
    )
    normalized = str(raw_value or "").strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid {DEEPSEEK_V4_REASONING_ENV}={raw_value!r}; "
        "use 1/0, true/false, yes/no, or on/off."
    )


def _resolve_reasoning_effort(
    profile: Optional[DeepSeekV4Profile],
    *,
    force_enabled: bool,
    reasoning_effort: Optional[str],
) -> str:
    if reasoning_effort is not None:
        value = str(reasoning_effort).strip().lower()
    elif force_enabled:
        value = os.getenv(DEEPSEEK_V4_REASONING_EFFORT_ENV, "").strip().lower()
        if not value:
            value = profile.reasoning_effort if profile is not None else "max"
    else:
        value = profile.reasoning_effort if profile is not None else "max"
    if value not in _SUPPORTED_REASONING_EFFORTS:
        supported = ", ".join(sorted(_SUPPORTED_REASONING_EFFORTS))
        raise ValueError(
            f"Invalid DeepSeek V4 reasoning effort {value!r}; "
            f"supported values: {supported}."
        )
    return value


def deepseek_v4_request_kwargs(
    model_name: str,
    *,
    force_enabled: bool = False,
    reasoning_effort: Optional[str] = None,
) -> Dict[str, Any]:
    profile = parse_deepseek_v4_profile(model_name)
    if profile is None and not force_enabled:
        return {}
    effort = _resolve_reasoning_effort(
        profile,
        force_enabled=force_enabled,
        reasoning_effort=reasoning_effort,
    )
    return {
        "extra_body": {
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": effort,
            },
            "separate_reasoning": True,
        },
    }


def coerce_openai_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value or "")


SPECIAL_TOKEN_TEXTS = (
    "<|endoftext|>",
    "<|end_of_text|>",
    "<|im_end|>",
    "<|end|>",
)


def strip_special_token_text(value: Any) -> str:
    text = coerce_openai_text(value).strip()
    if not text:
        return ""
    for token in SPECIAL_TOKEN_TEXTS:
        text = text.replace(token, "")
    return text.strip()


def extract_reasoning_text(message: Any) -> str:
    if message is None:
        return ""
    for attr in ("reasoning_content", "reasoning", "reasoning_details"):
        value = getattr(message, attr, None)
        text = strip_special_token_text(value)
        if text:
            return text
    return ""


def split_deepseek_v4_response_text(
    content: Any,
    reasoning_content: Any = "",
) -> DeepSeekV4ResponseText:
    explicit_reasoning = strip_special_token_text(reasoning_content)
    content_text = strip_special_token_text(content)
    if explicit_reasoning:
        return DeepSeekV4ResponseText(
            reasoning_content=explicit_reasoning,
            content=content_text,
            fallback_split_used=False,
        )

    leading_match = _LEADING_EMPTY_THINK_END_RE.match(content_text)
    had_leading_empty_close = leading_match is not None
    candidate = (
        content_text[leading_match.end() :]
        if leading_match is not None
        else content_text
    )
    separator = _THINK_END_RE.search(candidate)
    if separator is None:
        return DeepSeekV4ResponseText(
            reasoning_content="",
            content=candidate.strip() if had_leading_empty_close else content_text,
            fallback_split_used=had_leading_empty_close,
        )

    reasoning = candidate[: separator.start()]
    reasoning = _THINK_START_RE.sub("", reasoning, count=1).strip()
    visible_content = candidate[separator.end() :].strip()
    return DeepSeekV4ResponseText(
        reasoning_content=reasoning,
        content=visible_content,
        fallback_split_used=True,
    )
