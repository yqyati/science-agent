"""Ordered Judge services shared by rollout grading and startup checks."""

import asyncio
import json
import logging
import math
import os
from pathlib import Path

import httpx

from sandbox_gateway import AgentSandboxRun, SandboxGatewayError, SandboxStateError, shared_sandbox_run

logger = logging.getLogger(__name__)


def load_judge_chain(*, default_max_tokens=1800):
    """Load explicit services, or preserve the legacy Sandbox/direct defaults."""
    config_path = os.environ.get("SCIENCE_AGENT_JUDGE_CONFIG", "").strip()
    if config_path:
        chain = json.loads(Path(config_path).read_text())
    else:
        chain = [{
            "name": "primary", "transport": "sandbox",
            "base_url": os.environ["SCIENCE_AGENT_JUDGE_BASE_URL"],
            "model": os.environ["SCIENCE_AGENT_JUDGE_MODEL"],
            "api_key_env": "SCIENCE_AGENT_JUDGE_API_KEY" if os.getenv("SCIENCE_AGENT_JUDGE_API_KEY") else "API_KEY",
        }]
        fallback_base = os.environ.get("SCIENCE_AGENT_JUDGE_FALLBACK_BASE_URL", "http://s-20261020163637-mpm6m.ailab-agentshared.pjh-service.org.cn/v1")
        if fallback_base:
            chain.append({
                "name": "direct-fallback", "transport": "direct", "base_url": fallback_base,
                "model": os.environ.get("SCIENCE_AGENT_JUDGE_FALLBACK_MODEL", "DeepSeek-V4.1-Flash-1017"),
                "api_key_env": "SCIENCE_AGENT_JUDGE_FALLBACK_API_KEY" if os.getenv("SCIENCE_AGENT_JUDGE_FALLBACK_API_KEY") else "",
            })
    if not isinstance(chain, list) or not chain:
        raise ValueError("Judge config must be a nonempty JSON array")
    names = set()
    allowed = {"name", "transport", "base_url", "model", "api_key_env", "thinking", "thinking_format",
               "reasoning_effort", "max_tokens", "timeout", "retry_delays", "sandbox_replacements"}
    for index, entry in enumerate(chain):
        if not isinstance(entry, dict) or set(entry) - allowed:
            raise ValueError(f"Judge entry {index + 1}: unknown fields or not an object")
        name = entry.setdefault("name", f"judge-{index + 1}")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Judge names must be nonempty and unique")
        names.add(name)
        transport = entry.get("transport")
        if transport not in {"sandbox", "direct"}:
            raise ValueError(f"Judge {name}: transport must be sandbox or direct")
        for field in ("model", "base_url"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f"Judge {name}: {field} is required")
        if not entry["base_url"].startswith(("http://", "https://")):
            raise ValueError(f"Judge {name}: base_url must be HTTP(S)")
        key_env = entry.get("api_key_env", "")
        if not isinstance(key_env, str) or (key_env and not os.getenv(key_env)):
            raise ValueError(f"Judge {name}: api_key_env must reference a populated environment variable")
        entry["api_key"] = os.environ[key_env] if key_env else "EMPTY"
        entry.setdefault("retry_delays", [1, 5, 10, 30, 60, 180] if transport == "sandbox" else [])
        delays = entry["retry_delays"]
        if not isinstance(delays, list) or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 180 for x in delays):
            raise ValueError(f"Judge {name}: retry_delays must contain waits between 0 and 180 seconds")
        entry.setdefault("sandbox_replacements", 1 if transport == "sandbox" else 0)
        if type(entry["sandbox_replacements"]) is not int or entry["sandbox_replacements"] not in (0, 1) or (transport == "direct" and entry["sandbox_replacements"]):
            raise ValueError(f"Judge {name}: sandbox_replacements must be 0 or 1 (0 for direct)")
        thinking = os.environ.get("SCIENCE_AGENT_JUDGE_THINKING", "").strip().lower()
        entry.setdefault("thinking", thinking if thinking in {"enabled", "disabled"} else "")
        entry.setdefault("thinking_format", "chat_template_kwargs" if transport == "direct" else "thinking")
        if entry["thinking"] not in {"", "enabled", "disabled"} or entry["thinking_format"] not in {"thinking", "chat_template_kwargs", "none"}:
            raise ValueError(f"Judge {name}: invalid thinking or thinking_format")
        if entry["thinking_format"] == "none":
            entry["thinking"] = ""
        entry.setdefault("reasoning_effort", 20)
        if type(entry["reasoning_effort"]) is not int or not 0 <= entry["reasoning_effort"] <= 100:
            raise ValueError(f"Judge {name}: reasoning_effort must be an integer between 0 and 100")
        entry.setdefault("max_tokens", int(os.environ.get("SCIENCE_AGENT_JUDGE_MAX_TOKENS", str(default_max_tokens))))
        entry.setdefault("timeout", float(os.environ.get("SCIENCE_AGENT_JUDGE_TIMEOUT", "180")))
        if type(entry["max_tokens"]) is not int or entry["max_tokens"] <= 0:
            raise ValueError(f"Judge {name}: max_tokens must be a positive integer")
        if type(entry["timeout"]) not in (int, float) or not math.isfinite(entry["timeout"]) or entry["timeout"] <= 0:
            raise ValueError(f"Judge {name}: timeout must be positive and finite")
    return chain


async def judge_answer(messages, *, require_score=False, sandbox_run=None):
    chain = load_judge_chain(default_max_tokens=1800 if require_score else 6000)
    judge_result = None
    judge_error = judge_primary_error = judge_response_summary = judge_finish_reason = ""
    judge_attempts = judge_thinking_fallbacks = judge_content_chars = sandbox_restarts = 0
    history = []
    used_task_sandbox = False
    for judge_index, entry in enumerate(chain):
        direct = entry["transport"] == "direct"
        judge_backend = "direct_fallback" if direct and judge_index else "direct" if direct else "gateway_sandbox"
        judge_url = entry["base_url"].rstrip("/") + "/chat/completions"
        judge_model = entry["model"]
        retry_delays = entry["retry_delays"]
        judge_response_summary = judge_finish_reason = ""
        judge_content_chars = 0
        for sandbox_attempt in range(entry["sandbox_replacements"] + 1):
            owned_run = False
            judge_run = None
            if not direct:
                owned_run = used_task_sandbox
                judge_run = AgentSandboxRun() if owned_run else sandbox_run if sandbox_run is not None else shared_sandbox_run()
                used_task_sandbox = True
                sandbox_restarts += int(owned_run)
            judge_connection_failures = 0
            try:
                for attempt in range(len(retry_delays) + 1):
                    judge_input = {
                        "model": judge_model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": entry["max_tokens"],
                        "response_format": {"type": "json_object"},
                    }
                    judge_thinking = entry["thinking"]
                    try:
                        thinking_modes = ("enabled", "disabled") if judge_thinking == "enabled" else (judge_thinking,)
                        for thinking_mode in thinking_modes:
                            if entry["thinking_format"] == "chat_template_kwargs":
                                judge_input["chat_template_kwargs"] = {"thinking": thinking_mode != "disabled"}
                                if thinking_mode != "disabled":
                                    judge_input["chat_template_kwargs"]["reasoning_effort"] = entry["reasoning_effort"]
                            elif entry["thinking_format"] == "thinking" and thinking_mode:
                                judge_input["thinking"] = {"type": thinking_mode}
                            judge_attempts += 1
                            if direct:
                                # Internal service is reachable directly; do not route through environment proxies.
                                async with httpx.AsyncClient(trust_env=False, timeout=entry["timeout"]) as client:
                                    reply = await client.post(judge_url, json=judge_input, headers={
                                        "Authorization": "Bearer " + entry["api_key"],
                                    })
                                response = {"status_code": reply.status_code, "text": reply.text}
                            else:
                                request_task = asyncio.create_task(asyncio.to_thread(
                                    judge_run.run_http,
                                    {
                                        "method": "POST", "url": judge_url, "json": judge_input,
                                        "headers": {
                                            "Content-Type": "application/json",
                                            "Authorization": "Bearer " + entry["api_key"],
                                        },
                                    },
                                    timeout=entry["timeout"],
                                ))
                                try:
                                    response = await asyncio.shield(request_task)
                                except asyncio.CancelledError:
                                    # to_thread keeps running after cancellation. Wait before
                                    # deletion so it cannot create a Sandbox after cleanup.
                                    try:
                                        await request_task
                                    finally:
                                        raise asyncio.CancelledError
                            judge_connection_failures = 0
                            raw_response = response["text"]
                            judge_response_summary = raw_response[:500]
                            if response["status_code"] >= 400:
                                raise RuntimeError(f"Judge HTTP {response['status_code']}: {judge_response_summary}")
                            try:
                                judge_finish_reason = ""
                                judge_content_chars = 0
                                body = json.loads(raw_response)
                                choice = body["choices"][0]
                                judge_finish_reason = str(choice.get("finish_reason") or "")
                                content = choice["message"]["content"]
                                if isinstance(content, list):
                                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                                content = ("" if content is None else str(content)).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                                judge_content_chars = len(content)
                                parsed = json.loads(content)
                                if not isinstance(parsed, dict) or not isinstance(parsed.get("correct"), bool):
                                    raise ValueError(f"invalid judge result: {parsed!r}")
                                if not require_score:
                                    correct = parsed["correct"]
                                    judge_result = {"correct": correct, "score": 1.0 if correct else 0.0, "reason": str(parsed.get("reason", ""))}
                                else:
                                    score = float(parsed["score"])
                                    if score not in (0.0, 1.0) or bool(parsed.get("correct")) != bool(score):
                                        raise ValueError(f"invalid judge result: {parsed!r}")
                                    judge_result = parsed
                            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
                                if thinking_mode != "enabled":
                                    raise
                                judge_thinking_fallbacks += 1
                                logger.warning(
                                    "Science judge returned no valid grade with thinking enabled; "
                                    "retrying with thinking disabled error_type=%s finish_reason=%s content_chars=%d",
                                    type(exc).__name__, judge_finish_reason, judge_content_chars,
                                )
                                continue
                            break
                        judge_error = ""
                        break
                    except Exception as exc:  # noqa: BLE001
                        judge_error = f"{type(exc).__name__}: {exc}"
                        connection_failed = (
                            isinstance(exc, SandboxGatewayError)
                            and "proxy HTTP 502" in str(exc)
                            and "Sandbox upstream connection failed" in str(exc)
                        )
                        judge_connection_failures = judge_connection_failures + 1 if connection_failed else 0
                        # Grading saved text needs no recovered Python state or model acknowledgement.
                        replace_sandbox = isinstance(exc, SandboxStateError) or judge_connection_failures >= 2
                        logger.warning(
                            "Judge failure name=%s transport=%s instance=%d attempt=%d/%d error=%s next_retry_s=%s",
                            entry["name"], entry["transport"], sandbox_attempt + 1, attempt + 1,
                            len(retry_delays) + 1, judge_error,
                            retry_delays[attempt] if not replace_sandbox and attempt < len(retry_delays) else None,
                        )
                        if replace_sandbox:
                            logger.warning("Judge Sandbox unusable; ending retries in this instance error_type=%s", type(exc).__name__)
                            break
                        if attempt < len(retry_delays):
                            await asyncio.sleep(retry_delays[attempt])
            finally:
                if owned_run:
                    try:
                        await asyncio.to_thread(judge_run.close)
                    except Exception:
                        logger.exception("Replacement judge Sandbox cleanup failed")
            if judge_result is not None:
                break
        history.append({"name": entry["name"], "transport": entry["transport"], "model": judge_model,
                        "valid": judge_result is not None, "error": judge_error})
        if judge_result is not None:
            break
        if not judge_primary_error:
            judge_primary_error = judge_error
        logger.warning("Judge service exhausted name=%s transport=%s next_service=%s",
                       entry["name"], entry["transport"], judge_index + 1 < len(chain))
    return {
        "judge_result": judge_result, "judge_error": judge_error, "judge_primary_error": judge_primary_error,
        "judge_response_summary": judge_response_summary, "judge_finish_reason": judge_finish_reason,
        "judge_content_chars": judge_content_chars, "judge_attempts": judge_attempts,
        "judge_thinking_fallbacks": judge_thinking_fallbacks, "judge_backend": judge_backend,
        "judge_model": judge_model, "direct": direct, "sandbox_restarts": sandbox_restarts,
        "judge_name": entry["name"], "judge_index": judge_index, "history": history,
    }
