"""Try other configured Jina keys only when the account has no balance."""

import json
import os
from urllib.parse import urlsplit


def _balance_exhausted(response):
    status = response.get("status_code") if isinstance(response, dict) else response.status_code
    if int(status) != 402:
        return False
    text = response.get("text", "") if isinstance(response, dict) else response.text
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return False
    return isinstance(data, dict) and (
        data.get("name") == "InsufficientBalanceError"
        or "account balance not enough" in str(data.get("message", "")).lower()
    )


def request_with_jina_key_failover(request, *, url, headers=None):
    """Keep the original key first; never retry it after an explicit balance error."""
    current = dict(headers or {})
    if urlsplit(url).hostname not in {"r.jina.ai", "s.jina.ai", "api.jina.ai"}:
        return request(current)
    keys = list(dict.fromkeys(
        key.strip()
        for name in ("JINA_API_KEYS", "JINA_API_KEY")
        for key in os.getenv(name, "").split(",")
        if key.strip()
    ))
    auth_name = next((name for name in current if name.lower() == "authorization"), "Authorization")
    original = current.get(auth_name, "")
    original_key = original[7:] if original.lower().startswith("bearer ") else ""
    alternatives = [key for key in keys if key != original_key]
    response = request(current)
    for key in alternatives:
        if not _balance_exhausted(response):
            break
        current[auth_name] = "Bearer " + key
        response = request(dict(current))
    return response
