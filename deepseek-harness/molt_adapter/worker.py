"""One native DSH session per Molt episode; no outer beam search."""
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from bridge import serve
from isolation import ROOT
from verifier import EpisodeVerifier


def main(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if any(config[name] <= 0 for name in ("wall_seconds", "max_requests", "max_tokens", "request_timeout")):
        raise ValueError("Episode limits must be positive")
    episode = config_path.parent
    # Keep untrusted tools unprivileged even when the training container is root.
    if os.geteuid() == 0:
        for path in [episode, *episode.rglob("*")]:
            os.chown(path, 65534, 65534)
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    sys.path.insert(0, str(ROOT / "vendor"))
    from deepseek_harness import DeepSeekHarness
    from deepseek_harness_runtime import resolve_bundled_launch_args

    workspace = episode / "workspace"
    workspace.mkdir()
    home = episode / "dsh-home"
    shutil.copytree(ROOT / "vendor/profile", home / "profiles/sdk")
    verifier = EpisodeVerifier(config["item"], config["pipeline"], episode / "verifier")
    if verifier.initial is not None:
        (workspace / "initial_candidate.txt").write_text(verifier.initial)
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    done = threading.Event()
    result_box = {}
    deadline = time.monotonic() + config["wall_seconds"]
    with serve(config, verifier, episode / "requests.jsonl") as (url, token, state):
        helper = (
            "import json,sys,urllib.request\nfrom pathlib import Path\n"
            "content=Path(sys.argv[1] if len(sys.argv)>1 else 'submission.txt').read_text()\n"
            f"request=urllib.request.Request({(url + '/score')!r},data=json.dumps({{'content':content}}).encode(),"
            f"headers={{'Authorization':{('Bearer ' + token)!r},'Content-Type':'application/json'}})\n"
            f"with urllib.request.build_opener(urllib.request.ProxyHandler({{}})).open(request,timeout={verifier.timeout + 10}) as r: print(r.read().decode())\n"
        )
        (workspace / "score_submission.py").write_text(helper)
        prompt = config["prompt"] + (
            "\n\nUse one DSH research session. Do not launch other model agents or external model APIs. "
            "Submit candidates as OUTPUT:\n<JSON object>\nSOLUTION:\n<explanation>. "
            "Write a candidate to submission.txt, then run python3 score_submission.py submission.txt. "
            "Use verifier feedback to improve the candidate; repeated scores give no extra reward. "
            "The best verified candidate during this episode determines the reward."
        )
        if verifier.initial is not None:
            prompt += f"\nContinue from initial_candidate.txt (verified objective {verifier.anchor}). Seek a strict improvement."
        (workspace / "TASK.md").write_text(prompt)
        patch = home / "molt.patch.yml"
        patch.write_text(
            "- id: llm-deepseek\n  config:\n    streamIdleTimeoutMs: 1200000\n"
            "- id: compaction-basic\n  config:\n    auto: false\n"
            "- id: tool-subagent\n  disabled: true\n"
            "- id: tool-subagent-fork\n  disabled: true\n"
            "- id: tool-subagent-codex\n  disabled: true\n"
            "- id: tool-subagent-claude-code\n  disabled: true\n"
        )
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home), "TMPDIR": "/tmp",
               "DSH_HOME": str(home), "DSH_CWD": str(workspace),
               "DSH_SESSION_ROOT": str(home / "sessions"), "DSH_PERMISSION_MODE": "danger-full-access",
               "DSH_RUNTIME_MODE": "exe", "NO_PROXY": "127.0.0.1,localhost",
               "no_proxy": "127.0.0.1,localhost", "LANG": "C.UTF-8",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        # Proxy configuration contains no model credentials and applies to web tools only.
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            if os.environ.get(key):
                env[key] = os.environ[key]
        launch = {"workspace": str(workspace), "home": str(home), "env": env,
                  "argv": [*resolve_bundled_launch_args("exe"), "--profile", "sdk", "--patch", str(patch)]}
        harness = DeepSeekHarness(
            provider="deepseek-official", model=config["model_name"], base_url=url + "/v1", api_key=token,
            max_tokens=config["max_tokens"], cwd=str(workspace), dsh_home=str(home),
            env=env, request_timeout_seconds=config["wall_seconds"] + 30,
            _launch_args=(sys.executable, str(Path(__file__).with_name("isolation.py")), json.dumps(launch)))

        def infer():
            try:
                result_box["run"] = harness.run(prompt, session_id=config["session_id"])
            except Exception as exc:
                result_box["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                done.set()

        inference = threading.Thread(target=infer, daemon=True)
        inference.start()
        while not done.wait(0.1):
            if stopped.is_set() or time.monotonic() >= deadline or state["budget_exhausted"] or state["error"] or verifier.error:
                break
        truncated = not done.is_set() or state["budget_exhausted"]
        try:
            if "run" in result_box:
                run = result_box["run"]
                (episode / "events.json").write_text(json.dumps(run.events, ensure_ascii=False).replace(token, "<episode-token>"))
                (episode / "response.txt").write_text(run.final_response or "")
                if run.final_response and not stopped.is_set():
                    verifier.submit(run.final_response)
                if run.finish_reason == "error" and not state["budget_exhausted"]:
                    raise RuntimeError("DSH ended with a runtime error; inspect events.json")
            if state["error"] and not state["budget_exhausted"]:
                raise RuntimeError(state["error"])
            if result_box.get("error") and not truncated:
                raise RuntimeError(result_box["error"])
            outcome = {**verifier.outcome(), "truncated": truncated, "model_requests": state["requests"],
                       "termination": "cancelled" if stopped.is_set() else "budget" if truncated else "completed"}
            (episode / "result.json").write_text(json.dumps(outcome, allow_nan=False) + "\n")
        finally:
            harness.close()
            inference.join(timeout=3)


if __name__ == "__main__":
    main(sys.argv[1])
