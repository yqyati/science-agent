"""Trusted, training-only verifier and episode-level objective reward."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading


KINDS = {"ljcr_exact", "ac_official", "endless_official", "v1114_exact"}


class EpisodeVerifier:
    def __init__(self, item, pipeline, archive):
        if item.get("generation_split") != "train":
            raise ValueError("Only training tasks may enter this adapter")
        self.task = item["task"]
        if self.task["task_id"] != item["task_id"]:
            raise ValueError("Task ID mismatch")
        if self.task["verifier"]["kind"] not in KINDS:
            raise ValueError("Unsupported verifier kind")
        self.pipeline = Path(pipeline).resolve()
        self.v1114 = self.task["verifier"]["kind"] == "v1114_exact"
        self.entry = "verify_candidate.py" if self.v1114 else "tools/verify_task.py"
        if not (self.pipeline / self.entry).is_file():
            raise FileNotFoundError(self.pipeline / self.entry)
        self.archive = Path(archive)
        self.archive.mkdir(parents=True, exist_ok=True)
        self.timeout = float(item.get("verifier_timeout", 120))
        self.direction = item["direction"]
        if self.direction not in {"min", "max"}:
            raise ValueError("Invalid optimization direction")
        self.binary = item.get("verification_only", False)
        self.anchor = item.get("reward_anchor")
        self.scale = float(item.get("reward_scale", 1))
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        if self.anchor is not None and (type(self.anchor) not in (int, float) or not math.isfinite(self.anchor)):
            raise ValueError("reward_anchor must be finite")
        self.best = None
        self.count = 0
        self.cache = {}
        self.lock = threading.Lock()
        self.error = None
        self.initial = item.get("initial_candidate")
        if self.initial is not None:
            result = self.submit(self.initial)
            if not result["valid"]:
                raise ValueError("Initial candidate must pass the trusted verifier")
            self.anchor = result["objective"]

    def submit(self, content):
        with self.lock:
            if content in self.cache:
                return {**self.cache[content], "duplicate": True}
            self.count += 1
            prefix = self.archive / f"{self.count:06d}"
            prefix.with_suffix(".submission.txt").write_text(content)
            text = content.strip().removeprefix("OUTPUT:").lstrip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
            try:
                payload, _ = json.JSONDecoder().raw_decode(text)
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                json.dumps(payload, allow_nan=False)
            except (ValueError, TypeError) as exc:
                result = {"valid": False, "objective": None, "error": str(exc)}
            else:
                with tempfile.TemporaryDirectory(prefix="dsh-verifier-") as tmp:
                    tmp = Path(tmp)
                    (tmp / "task.jsonl").write_text(json.dumps(self.task) + "\n")
                    (tmp / "answer.json").write_text(json.dumps(payload, allow_nan=False))
                    command = [sys.executable, "-B", str(self.pipeline / self.entry),
                               "--tasks", str(tmp / "task.jsonl"), "--task-id", self.task["task_id"],
                               "--answer", str(tmp / "answer.json")]
                    if self.v1114:
                        command = [sys.executable, "-B", str(self.pipeline / self.entry),
                                   "--task-id", self.task["task_id"], "--answer-file", str(tmp / "answer.json"),
                                   "--timeout", str(self.timeout)]
                    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
                    process = subprocess.Popen(command, cwd=tmp, env=env, text=True,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                               start_new_session=True)
                    try:
                        stdout, stderr = process.communicate(timeout=self.timeout)
                        if process.returncode:
                            raise RuntimeError(f"Verifier exited {process.returncode}: {stderr[-2000:]}")
                        raw = json.loads(stdout)
                        if raw.get("status") in {"resource_limit", "partial_verified", "verifier_error"}:
                            raise RuntimeError(f"Verifier unavailable: {raw}")
                    except BaseException:
                        self.error = "Trusted verifier failed; episode must not become a negative example"
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                        process.communicate()
                        raise
                ac = self.task["verifier"]["kind"] == "ac_official"
                valid = raw.get("ok" if ac else "valid") is True
                objective = raw.get("score" if self.v1114 else "length" if ac else "objective")
                if self.v1114 and valid and raw.get("direction") != {"min": "minimize", "max": "maximize"}[self.direction]:
                    self.error = "Verifier direction mismatch"
                    raise RuntimeError(self.error)
                valid = valid and type(objective) in (int, float) and math.isfinite(objective)
                result = {"valid": bool(valid), "objective": objective if valid else None, "verifier": raw}
            if result["valid"]:
                score = result["objective"]
                if self.best is None or (score < self.best if self.direction == "min" else score > self.best):
                    self.best = score
            self.cache[content] = result
            prefix.with_suffix(".result.json").write_text(json.dumps(result, allow_nan=False) + "\n")
            return result

    def outcome(self):
        if self.error:
            raise RuntimeError(self.error)
        if self.best is None:
            reward = 0.0
        elif self.binary:
            reward = 1.0
        elif self.anchor is None:
            reward = (1.0 / (1.0 + max(0.0, self.best) / self.scale) if self.direction == "min"
                      else max(0.0, self.best) / (self.scale + max(0.0, self.best)))
        else:
            gain = (self.best - self.anchor) * (1 if self.direction == "max" else -1)
            reward = (math.tanh(max(0.0, gain) / self.scale) if self.initial is not None
                      else 0.5 + 0.5 * math.tanh(gain / self.scale))
        return {"reward": reward, "best_objective": self.best, "valid": self.best is not None,
                "unique_submissions": self.count, "reward_anchor": self.anchor, "reward_scale": self.scale}
