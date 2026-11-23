from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from urllib.request import Request, ProxyHandler, build_opener

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "molt_adapter"))
from bridge import serve
from verifier import EpisodeVerifier

PIPELINE = Path(os.environ.get("DSH_VERIFIER_ROOT", "/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/sujiaming/DSH_Teacher3536_Handoff_20261113/teacher_rollout/verifier_sources"))


def item():
    return {"task_id": "adapter-synthetic-C4-2-1", "generation_split": "train", "direction": "min",
            "reward_anchor": 3, "reward_scale": 1,
            "task": {"task_id": "adapter-synthetic-C4-2-1", "family": "covering_design",
                     "parameters": {"v": 4, "k": 2, "t": 1}, "verifier": {"kind": "ljcr_exact"}}}


def test_real_verifier_reward_and_duplicate(tmp_path):
    verifier = EpisodeVerifier(item(), PIPELINE, tmp_path)
    candidate = '{"blocks":[[1,2],[3,4]]}'
    assert verifier.submit(candidate)["objective"] == 2
    reward = verifier.outcome()["reward"]
    assert 0.5 < reward < 1
    assert verifier.submit(candidate)["duplicate"]
    assert verifier.outcome()["reward"] == reward
    assert verifier.count == 1
    assert not verifier.submit('{"blocks":[[1,2]]}')["valid"]
    assert verifier.outcome()["reward"] == reward


def test_initial_candidate_requires_improvement(tmp_path):
    row = item()
    row["initial_candidate"] = '{"blocks":[[1,2],[2,3],[3,4]]}'
    verifier = EpisodeVerifier(row, PIPELINE, tmp_path)
    assert verifier.outcome()["reward"] == 0
    verifier.submit('{"blocks":[[1,2],[3,4]]}')
    assert verifier.outcome()["reward"] > 0


def test_eval_rejected(tmp_path):
    row = item()
    row["generation_split"] = "test"
    with pytest.raises(ValueError, match="Only training"):
        EpisodeVerifier(row, PIPELINE, tmp_path)


def test_verifier_failure_is_not_zero_reward(tmp_path):
    verifier = EpisodeVerifier(item(), PIPELINE, tmp_path)
    pipeline = tmp_path / "broken/tools"
    pipeline.mkdir(parents=True)
    (pipeline / "verify_task.py").write_text("raise RuntimeError('infra failure')\n")
    verifier.pipeline = pipeline.parent
    with pytest.raises(RuntimeError):
        verifier.submit('{"blocks":[[1,2],[3,4]]}')
    with pytest.raises(RuntimeError, match="must not become a negative"):
        verifier.outcome()


def test_maximize_reward(tmp_path):
    row = item()
    row.update(direction="max", reward_anchor=0)
    verifier = EpisodeVerifier(row, PIPELINE, tmp_path)
    verifier.submit('{"blocks":[[1,2],[3,4]]}')
    first = verifier.outcome()["reward"]
    verifier.submit('{"blocks":[[1,2],[2,3],[3,4]]}')
    assert verifier.outcome()["reward"] > first




@pytest.mark.skipif(os.environ.get("DSH_RUN_NATIVE_TESTS") != "1", reason="Requires host namespace support")
def test_native_dsh_tool_and_verifier():
    observed = []
    tool_results = []

    class Policy(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observed.append(body)
            tool_results.extend(m for m in body["messages"] if m.get("role") == "tool" or (
                m.get("role") == "user" and (m.get("content") or "").startswith("<tool_response>")))
            if len(observed) == 1:
                tools = {x["function"]["name"]: x["function"] for x in body["tools"]}
                assert not any("subagent" in name for name in tools), tools.keys()
                name = next(name for name in tools if name == "bash" or name.endswith("bash"))
                parameters = tools[name]["parameters"]
                observed[0]["test_tool_schema"] = parameters
                code = "printf '%s' '{\"blocks\":[[1,2],[3,4]]}' > submission.txt\npython3 score_submission.py submission.txt"
                message = {"role": "assistant", "content": "", "tool_calls": [{"index": 0, "id": "call-1", "type": "function", "function": {"name": name, "arguments": json.dumps({"command": code, "description": "Construct and verify covering"})}}]}
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": 'OUTPUT:\n{"blocks":[[1,2],[3,4]]}\nSOLUTION:\nPartition covers all four points.'}
                finish = "stop"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if body.get("stream") else "application/json")
            self.end_headers()
            if body.get("stream"):
                chunk = {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "policy", "choices": [{"index": 0, "delta": message, "finish_reason": None}]}
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish}]
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\ndata: [DONE]\n\n")
            else:
                self.wfile.write(json.dumps({"id": "test", "model": "policy", "choices": [{"index": 0, "message": message, "finish_reason": finish}]}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Policy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runs = Path(os.environ.get("DSH_TEST_RUN_ROOT", ROOT / "runs")) / "native_acceptance"
    runs.mkdir(parents=True, exist_ok=True)
    episode = Path(tempfile.mkdtemp(prefix="episode-", dir=runs))
    config = {"item": item(), "prompt": "Construct a (4,2,1) covering.", "pipeline": str(PIPELINE),
              "base_url": f"http://127.0.0.1:{server.server_port}/s/test/v1", "api_key": "EMPTY", "model_name": "policy",
              "temperature": 1.0, "top_p": 1.0, "max_tokens": 2048, "max_requests": 4,
              "request_timeout": 30, "wall_seconds": 45, "session_id": "test-session"}
    (episode / "config.json").write_text(json.dumps(config))
    try:
        process = subprocess.run([sys.executable, str(ROOT / "molt_adapter/worker.py"), str(episode / "config.json")],
                                 capture_output=True, text=True, timeout=75)
        (episode / "acceptance.log").write_text(process.stdout + process.stderr)
        assert process.returncode == 0, process.stderr
        result = json.loads((episode / "result.json").read_text())
        assert result["best_objective"] == 2 and result["reward"] > 0.5
        assert len(observed) == 2 and tool_results, observed
        assert any('"valid": true' in str(m["content"]) for m in tool_results), tool_results
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
