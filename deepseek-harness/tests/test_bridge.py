"""Boundary cases for raw Qwen tool parsing and the JSON-to-SSE boundary."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, build_opener

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "molt_adapter"))
from bridge import normalize_completion, prepare_messages, serve

TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}, "count": {"type": "integer"},
                                        "items": {"type": "array"}}}}}]


def complete(text, finish="stop"):
    return {"id": "test", "object": "chat.completion", "model": "policy", "created": 0,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}]}


def xml(command="echo ok", extra=""):
    return f"<tool_call><function=bash><parameter=command>{command}</parameter>{extra}</function></tool_call>"


def test_json_arguments_containing_tags_unicode_and_newlines():
    arguments = {"command": "print('中文 </tool_call> <tool_call> </parameter>')\nprint(2)"}
    raw = complete("<think>plan</think>" + "<tool_call>" + json.dumps({"name": "bash", "arguments": arguments}) + "</tool_call><|im_end|>\n")
    original = deepcopy(raw)
    first = normalize_completion(raw, TOOLS)
    message = first["choices"][0]["message"]
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == arguments
    assert message["reasoning_content"] == "plan" and message["content"] == ""
    assert raw == original
    assert first == normalize_completion(raw, TOOLS)
    assert first != normalize_completion(raw, TOOLS, [{"role": "user", "content": "another turn"}])


def test_multiple_calls_preserve_order_and_schema_types():
    raw = complete(xml("123", '<parameter=count>2</parameter><parameter=items>[1,2]</parameter>') + xml("false"))
    calls = normalize_completion(raw, TOOLS)["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 2 and calls[0]["id"] != calls[1]["id"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "123", "count": 2, "items": [1, 2]}
    assert json.loads(calls[1]["function"]["arguments"]) == {"command": "false"}


@pytest.mark.parametrize("text", [
    xml() + '<tool_call>{"name":"bash","arguments":',
    xml().replace("</tool_call>", ""),
    xml().replace("bash", "unknown"),
    xml(extra="<parameter=command>oops</parameter>"),
    xml("echo '</parameter>'"),
    '<tool_call>{"name":[],"arguments":{}}</tool_call>',
    xml() + '<tool_call>{"name":"bash","arguments":{"count":NaN}}</tool_call>',
])
def test_malformed_or_ambiguous_calls_never_execute_subset(text):
    message = normalize_completion(complete(text), TOOLS)["choices"][0]["message"]
    assert not message.get("tool_calls")
    assert "<tool_call>" in message["content"]


@pytest.mark.parametrize("text", ["<think>try " + xml(), "try " + xml(), xml()])
def test_length_does_not_execute_tools_or_lose_truncation(text):
    choice = normalize_completion(complete(text, "length"), TOOLS)["choices"][0]
    assert choice["finish_reason"] == "length"
    assert not choice["message"].get("tool_calls")


def test_unclosed_thinking_cannot_execute_example_tools():
    message = normalize_completion(complete("<think>I could use " + xml()), TOOLS)["choices"][0]["message"]
    assert message["content"] == "" and not message.get("tool_calls")


def test_closed_thinking_examples_not_executed():
    message = normalize_completion(complete("<think>example " + xml() + "</think>final answer"), TOOLS)["choices"][0]["message"]
    assert message["content"] == "final answer" and not message.get("tool_calls")


def test_prefilled_thinking_requires_close_and_ignores_examples():
    raw = "example " + xml()
    message = normalize_completion(complete(raw), TOOLS, thinking_prefix=True)["choices"][0]["message"]
    assert message["content"] == "" and not message.get("tool_calls")
    message = normalize_completion(complete(raw + "</think>" + xml("echo actual")), TOOLS,
                                   thinking_prefix=True)["choices"][0]["message"]
    assert len(message["tool_calls"]) == 1
    assert json.loads(message["tool_calls"][0]["function"]["arguments"])["command"] == "echo actual"


def test_think_close_inside_tool_argument_is_literal_data():
    command = "print('</think>')"
    message = normalize_completion(complete(xml(command)), TOOLS)["choices"][0]["message"]
    assert json.loads(message["tool_calls"][0]["function"]["arguments"])["command"] == command


@pytest.mark.parametrize("payload", [{"error": "failed"}, {"choices": []}, {"choices": [None]},
                                    {"choices": [{"message": {"content": []}, "finish_reason": "unknown"}]}])
def test_malformed_success_response_is_infrastructure_error(payload):
    with pytest.raises(ValueError, match="Molt completion"):
        normalize_completion(payload, TOOLS)


def test_qwen_real_template_feedback_group_is_token_equivalent():
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    path = Path('/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/science_model/tool_models/K3_p1_rollout88+nemotron_science_four_tools-0930-063459/checkpoint/hf_iter_0000549/chat_template.jinja')
    template = ImmutableSandboxedEnvironment().from_string(path.read_text())
    messages = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "bash", "arguments": {"command": "first"}}},
        {"function": {"name": "bash", "arguments": {"command": "second"}}}]},
        {"role": "tool", "tool_call_id": "a", "content": "first-result"},
        {"role": "tool", "tool_call_id": "b", "content": "second-result"}]
    original = deepcopy(messages)
    prepared = prepare_messages(messages)
    assert len(prepared) == 3 and prepared[-1]["role"] == "user"
    assert template.render(messages=messages, add_generation_prompt=True) == template.render(messages=prepared, add_generation_prompt=True)
    assert messages == original


@pytest.mark.parametrize("stream,finish,status", [(True, "stop", 200), (False, "stop", 200),
                                                (True, "length", 200), (True, "stop", 503)])
def test_transport_json_only_sampling_and_finish(tmp_path, stream, finish, status):
    observed = []

    class Policy(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            observed.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(complete(xml(), finish) if status == 200 else {"error": "unavailable"}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Policy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = {"base_url": f"http://127.0.0.1:{server.server_port}/s/example/v1", "api_key": "EMPTY",
              "model_name": "policy", "temperature": 0.8, "top_p": 0.9, "max_tokens": 99,
              "max_requests": 10, "request_timeout": 10}
    try:
        with serve(config, None, tmp_path / "audit.jsonl") as (url, token, state):
            request = Request(url + "/v1/chat/completions", data=json.dumps({"stream": stream, "model": "wrong", "tools": TOOLS,
                              "messages": [{"role": "user", "content": "task"}]}).encode(),
                              headers={"Authorization": "Bearer " + token})
            if status != 200:
                with pytest.raises(HTTPError) as error:
                    build_opener(ProxyHandler({})).open(request)
                assert error.value.code == status
                error.value.close()
            else:
                with build_opener(ProxyHandler({})).open(request) as response:
                    payload = response.read()
                if stream:
                    lines = [line[6:] for line in payload.splitlines() if line.startswith(b"data: ")]
                    assert lines[-1] == b"[DONE]"
                    chunks = [json.loads(line) for line in lines[:-1]]
                    assert chunks[-1]["choices"][0]["finish_reason"] == ("length" if finish == "length" else "tool_calls")
                    if finish != "length":
                        assert chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
                else:
                    assert json.loads(payload)["choices"][0]["finish_reason"] == "tool_calls"
        assert bool(state["error"]) == (status != 200)
        assert state["budget_exhausted"] == (finish == "length")
        path, body = observed[0]
        assert path == "/s/example/v1/chat/completions"
        assert body["stream"] is False and body["model"] == "policy"
        assert body["temperature"] == 0.8 and body["top_p"] == 0.9 and body["max_tokens"] == 99
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_bad_score_requests_are_recoverable(tmp_path):
    candidates = []
    verifier = SimpleNamespace(submit=lambda text: candidates.append(text) or {'valid': True, 'objective': 1})
    config = {'base_url': 'http://127.0.0.1:1/v1'}
    with serve(config, verifier, tmp_path / 'audit.jsonl') as (url, token, state):
        for body in [{}, {'answer': {}}, {'content': {}}, {'content': None}, [], 'invalid JSON']:
            payload = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            request = Request(url + '/score', data=payload, headers={'Authorization': 'Bearer ' + token})
            with pytest.raises(HTTPError) as error:
                build_opener(ProxyHandler({})).open(request)
            assert error.value.code == 400
            error.value.close()
            assert state['error'] is None and candidates == []
        request = Request(url + '/score', data=json.dumps({'content': '{"answer": 1}'}).encode(),
                          headers={'Authorization': 'Bearer ' + token})
        with build_opener(ProxyHandler({})).open(request) as response:
            assert json.load(response)['valid'] is True
        assert candidates == ['{"answer": 1}'] and state['error'] is None


def test_score_disconnect_keeps_episode_usable(tmp_path, monkeypatch):
    import http.client
    import socketserver

    write = socketserver._SocketWriter.write
    disconnected = threading.Event()

    def disconnect_once(writer, payload):
        if payload.startswith(b'{"valid": true') and not disconnected.is_set():
            disconnected.set()
            raise BrokenPipeError('scoring caller already exited')
        return write(writer, payload)

    monkeypatch.setattr(socketserver._SocketWriter, 'write', disconnect_once)
    candidates = []
    verifier = SimpleNamespace(submit=lambda text: candidates.append(text) or {'valid': True, 'objective': 1})
    audit = tmp_path / 'audit.jsonl'
    with serve({'base_url': 'http://127.0.0.1:1/v1'}, verifier, audit) as (url, token, state):
        request = Request(url + '/score', data=b'{"content":"candidate"}',
                          headers={'Authorization': 'Bearer ' + token})
        with build_opener(ProxyHandler({})).open(request) as response:
            with pytest.raises(http.client.IncompleteRead):
                response.read()
        assert disconnected.wait(2)
        with build_opener(ProxyHandler({})).open(request) as response:
            assert json.load(response)['valid'] is True
        assert state['error'] is None and len(candidates) == 2
    assert json.loads(audit.read_text()) == {'path': '/score', 'error': 'BrokenPipeError'}
