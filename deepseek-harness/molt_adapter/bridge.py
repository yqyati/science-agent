"""Translate Molt's complete raw-token responses into native DSH tool messages."""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import secrets
import threading
import traceback
from urllib.parse import urlsplit


def normalize_completion(payload, tools, request_messages=(), thinking_prefix=False):
    """Molt records raw token text; expose Qwen's text tool syntax to native DSH."""
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list) or len(payload["choices"]) != 1:
        raise ValueError("Expected exactly one Molt completion choice")
    choice = payload["choices"][0]
    if (not isinstance(choice, dict) or not isinstance(choice.get("message"), dict)
            or choice.get("finish_reason") not in {"stop", "length", "tool_calls"}
            or not isinstance(choice["message"].get("content") or "", str)):
        raise ValueError("Malformed Molt completion")
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}).get("properties", {}) for t in tools}
    payload = deepcopy(payload)
    for choice in payload.get("choices", []):
        message = choice.get("message") or {}
        text = message.get("content") or ""
        thinking = thinking_prefix or text.lstrip().startswith("<think>")
        # A partial think block can contain hypothetical tool calls: never execute it.
        if choice.get("finish_reason") == "length" or (thinking and "</think>" not in text):
            message.pop("tool_calls", None)
            if thinking and "</think>" not in text:
                message["reasoning_content"] = text.removeprefix("<think>")
                message["content"] = ""
            continue
        if message.get("tool_calls"):
            continue
        if thinking and "</think>" in text:
            reasoning, text = text.split("</think>", 1)
            message["reasoning_content"] = reasoning.removeprefix("<think>")
        text = re.sub(r"(?:\s*<\|(?:im_end|endoftext)\|>)+\s*$", "", text)
        calls = []
        spans = []
        cursor = 0
        failed = False
        while (start := text.find("<tool_call>", cursor)) >= 0:
            body_start = start + len("<tool_call>")
            body = text[body_start:].lstrip()
            try:
                if body.startswith("{"):
                    # Decode JSON before finding the closing tag; tags can be string data.
                    call, end = json.JSONDecoder().raw_decode(body)
                    suffix = body[end:]
                    if not suffix.lstrip().startswith("</tool_call>"):
                        raise ValueError("Missing tool_call closing tag")
                    stop = len(text) - len(suffix.lstrip()) + len("</tool_call>")
                    name, arguments = call["name"], call["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                else:
                    end = text.find("</tool_call>", body_start)
                    if end < 0:
                        raise ValueError("Incomplete tool_call")
                    stop = end + len("</tool_call>")
                    block = text[body_start:end].strip()
                    function = re.fullmatch(r"<function=([^>]+)>(.*?)</function>", block, re.S)
                    if not function:
                        raise ValueError("Malformed function")
                    name = function.group(1).strip()
                    arguments = {}
                    rest = function.group(2).strip()
                    while rest:
                        parameter = re.match(r"<parameter=([^>]+)>(.*?)</parameter>", rest, re.S)
                        if not parameter:
                            raise ValueError("Malformed or ambiguous parameter tags")
                        key, value = parameter.group(1).strip(), parameter.group(2).strip()
                        if key in arguments:
                            raise ValueError("Duplicate parameter")
                        if schemas.get(name, {}).get(key, {}).get("type") != "string":
                            try:
                                value = json.loads(value)
                            except ValueError:
                                pass
                        arguments[key] = value
                        rest = rest[parameter.end():].strip()
                if not isinstance(name, str) or name not in schemas or not isinstance(arguments, dict):
                    raise ValueError("Unknown tool or invalid arguments")
                encoded = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
                identity = json.dumps([request_messages, text, len(calls)], ensure_ascii=False)
                calls.append({"id": "call_" + hashlib.sha256(identity.encode()).hexdigest()[:24], "type": "function",
                              "function": {"name": name, "arguments": encoded}})
                spans.append((start, stop))
                cursor = stop
            except (KeyError, ValueError, TypeError):
                failed = True
                break
        # Do not silently run a valid subset of a malformed multi-tool completion.
        if calls and not failed:
            for start, stop in reversed(spans):
                text = text[:start] + text[stop:]
            message["tool_calls"] = calls
            choice["finish_reason"] = "tool_calls"
        message["content"] = text.strip()
    return payload


def prepare_messages(messages):
    """Qwen renders adjacent tool results as one user turn; Molt appends one turn."""
    result = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "tool":
            result.append(message)
            index += 1
            continue
        feedback = []
        while index < len(messages) and messages[index].get("role") == "tool":
            content = messages[index].get("content")
            if not isinstance(content, str):
                raise ValueError("DSH adapter requires text-only tool results")
            feedback.append("<tool_response>\n" + content.strip() + "\n</tool_response>")
            index += 1
        result.append({"role": "user", "content": "\n".join(feedback)})
    return result


@contextmanager
def serve(config, verifier, audit):
    target = urlsplit(config["base_url"])
    if target.scheme != "http" or target.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Expected Molt's local session endpoint")
    token = secrets.token_urlsafe(32)
    state = {"requests": 0, "budget_exhausted": False, "error": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_error(403)
                return
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 64 * 1024 * 1024:
                self.send_error(413)
                return
            outgoing = None
            connection = None
            started = False
            try:
                try:
                    body = json.loads(self.rfile.read(size))
                    if not isinstance(body, dict):
                        raise ValueError("Expected a JSON object")
                    if self.path == "/score" and not isinstance(body.get("content"), str):
                        raise ValueError('Expected {"content": "candidate text"}; use score_submission.py')
                except (ValueError, TypeError) as exc:
                    self.send_error(400, str(exc))
                    return
                if self.path == "/score":
                    result = verifier.submit(body["content"])
                    payload = json.dumps(result, allow_nan=False).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    started = True
                    self.wfile.write(payload)
                    return
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                if state["budget_exhausted"] or state["requests"] >= config["max_requests"]:
                    state["budget_exhausted"] = True
                    self.send_error(409, "Episode budget exhausted")
                    return
                state["requests"] += 1
                outgoing = dict(body)
                outgoing["messages"] = prepare_messages(body.get("messages") or [])
                outgoing["stream"] = False
                outgoing.update(model=config["model_name"], temperature=config["temperature"],
                                top_p=config["top_p"], max_tokens=config["max_tokens"])
                outgoing.pop("max_completion_tokens", None)
                outgoing.pop("reasoning_effort", None)
                connection = http.client.HTTPConnection(target.hostname, target.port, timeout=config["request_timeout"])
                connection.request("POST", target.path.rstrip("/") + "/chat/completions",
                                   json.dumps(outgoing).encode(),
                                   {"Content-Type": "application/json", "Authorization": "Bearer " + config["api_key"]})
                response = connection.getresponse()
                # Molt's scoped endpoint is JSON-only. Read and validate before HTTP 200.
                if "text/event-stream" in response.getheader("Content-Type", ""):
                    raise ValueError("Expected a complete JSON response from Molt")
                payload = response.read()
                if response.status == 200:
                    completion = normalize_completion(json.loads(payload), outgoing.get("tools") or [], outgoing["messages"],
                                                      config.get("thinking_prefix", False))
                    for choice in completion.get("choices", []):
                        if choice.get("finish_reason") == "length":
                            state["budget_exhausted"] = True
                self.send_response(response.status)
                self.send_header("Content-Type", "text/event-stream" if body.get("stream") and response.status == 200
                                 else "application/json")
                self.end_headers()
                started = True
                if response.status != 200:
                    self.wfile.write(payload)
                elif body.get("stream"):
                    for choice in completion.get("choices", []):
                        delta = dict(choice["message"])
                        for index, call in enumerate(delta.get("tool_calls") or []):
                            call["index"] = index
                        chunk = {**completion, "object": "chat.completion.chunk", "choices": [
                            {"index": choice.get("index", 0), "delta": delta, "finish_reason": None}]}
                        self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                        chunk["choices"] = [{"index": choice.get("index", 0), "delta": {}, "finish_reason": choice["finish_reason"]}]
                        self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self.wfile.write(json.dumps(completion).encode())
                if response.status != 200:
                    state["error"] = f"Molt policy endpoint returned HTTP {response.status}"
                with audit.open("a") as handle:
                    handle.write(json.dumps({"request": state["requests"], "status": response.status,
                                             "model": outgoing["model"], "temperature": outgoing["temperature"],
                                             "top_p": outgoing["top_p"], "max_tokens": outgoing["max_tokens"]}) + "\n")
            except (BrokenPipeError, ConnectionResetError) as exc:
                # A scoring caller may time out or exit while verification is running.
                # Preserve its verified result; a policy disconnect still invalidates rollout.
                if self.path != "/score":
                    state["error"] = f"DSH connection interrupted: {type(exc).__name__}"
                with audit.open("a") as handle:
                    handle.write(json.dumps({"path": self.path, "error": type(exc).__name__}) + "\n")
            except Exception as exc:
                state["error"] = f"{type(exc).__name__}: {exc}"
                with audit.open("a") as handle:
                    handle.write(json.dumps({"path": self.path, "error": state["error"],
                                             "traceback": traceback.format_exc()}) + "\n")
                if not started:
                    self.send_error(502, "DSH adapter request failed")
            finally:
                if connection:
                    connection.close()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", token, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
