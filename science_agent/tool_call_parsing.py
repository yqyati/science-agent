from __future__ import annotations

import json
import re
from typing import Any


TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
FUNCTION_CALL_RE = re.compile(
    r"<function\s*=\s*([A-Za-z0-9_.:-]+)\s*>(.*?)</function>",
    re.IGNORECASE | re.DOTALL,
)
PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*([A-Za-z0-9_.:-]+)\s*>(.*?)</parameter>",
    re.IGNORECASE | re.DOTALL,
)
ARG_PAIR_RE = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*"
    r"<arg_value>\s*(.*?)\s*</arg_value>",
    re.IGNORECASE | re.DOTALL,
)
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _decode_argument_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return stripped


def _parse_tool_call_block(body: str) -> dict[str, Any] | None:
    stripped = body.strip()
    if not stripped:
        return None

    # Qwen/Hermes XML emitted by the model is commonly returned as:
    #   <function=search><parameter=query>...</parameter></function>
    # or as JSON inside the function element.  vLLM may leave this in the
    # assistant content, so normalize it here before the agent loop decides
    # that the turn is a final answer.
    function_match = FUNCTION_CALL_RE.fullmatch(stripped)
    if function_match is not None:
        name = function_match.group(1).strip()
        function_body = function_match.group(2).strip()
        if function_body.startswith("{"):
            try:
                arguments = json.loads(function_body)
            except (TypeError, ValueError, json.JSONDecodeError):
                arguments = None
            if isinstance(arguments, dict):
                return {"name": name, "arguments": arguments}

        arguments: dict[str, Any] = {}
        for match in PARAMETER_RE.finditer(function_body):
            arguments[match.group(1).strip()] = _decode_argument_value(match.group(2))
        if arguments or not function_body:
            return {"name": name, "arguments": arguments}
        return None

    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            name = str(payload.get("name") or "").strip()
            arguments = payload.get("arguments")
            if isinstance(arguments, str):
                arguments = _decode_argument_value(arguments)
            if name and isinstance(arguments, dict):
                return {"name": name, "arguments": arguments}

    first_arg = re.search(r"<arg_key>", stripped, re.IGNORECASE)
    if first_arg is None:
        return None
    name = stripped[: first_arg.start()].strip()
    if not name or not TOOL_NAME_RE.fullmatch(name):
        return None
    arguments: dict[str, Any] = {}
    for match in ARG_PAIR_RE.finditer(stripped[first_arg.start() :]):
        key = match.group(1).strip()
        if not key:
            continue
        arguments[key] = _decode_argument_value(match.group(2))
    if not arguments:
        return None
    return {"name": name, "arguments": arguments}


def extract_text_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """Parse JSON/XML and GLM arg-tag tool calls, preserving malformed blocks."""

    parsed: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        payload = _parse_tool_call_block(match.group(1))
        if payload is None:
            return match.group(0)
        parsed.append(payload)
        return ""

    remaining = TOOL_CALL_BLOCK_RE.sub(replace, str(text or "")).strip()
    return parsed, remaining
