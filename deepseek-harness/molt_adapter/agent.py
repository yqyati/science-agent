"""Molt plugin: policy endpoint -> independent DSH worker -> verified reward."""
import asyncio
from functools import partial
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

from molt.agents import ChatAgent, ChatAgentRunner, ChatContext, Result


class DSHAgent(ChatAgent):
    def __init__(self, thinking_prefix=False):
        self.thinking_prefix = thinking_prefix

    async def run(self, ctx: ChatContext) -> Result:
        item = json.loads(ctx.label) if isinstance(ctx.label, str) else ctx.label
        if not isinstance(item, dict) or item.get("generation_split") != "train":
            raise ValueError("Expected a DSH training row in dsh_item")
        if ctx.images or ctx.tools:
            raise ValueError("The first DSH adapter accepts text tasks and native DSH tools only")
        if len(ctx.messages) != 1 or ctx.messages[0].get("role") != "user":
            raise ValueError("Expected one user task; do not silently discard a conversation prefix")
        root = Path(os.environ["DSH_EPISODE_ROOT"]).resolve()
        root.mkdir(parents=True, exist_ok=True)
        # The worker drops root; ancestors must permit traversal, not expose other episodes.
        if os.geteuid() == 0 and not (root.stat().st_mode & 0o001):
            raise ValueError("DSH_EPISODE_ROOT must allow non-root directory traversal")
        episode = Path(tempfile.mkdtemp(prefix=f"{ctx.session_id}-", dir=root))
        wall_seconds = float(os.environ.get("DSH_WALL_SECONDS", "1800"))
        config = {
            "item": item, "prompt": ctx.prompt, "base_url": ctx.base_url, "api_key": ctx.api_key,
            "model_name": ctx.model_name, "session_id": ctx.session_id,
            "thinking_prefix": self.thinking_prefix,
            "temperature": float(ctx.sampling_params.temperature), "top_p": float(ctx.sampling_params.top_p),
            "max_tokens": min(int(ctx.sampling_params.max_tokens or ctx.max_length),
                              int(os.environ.get("DSH_MAX_COMPLETION_TOKENS", "16384"))),
            "max_requests": int(os.environ.get("DSH_MAX_REQUESTS", "128")),
            "request_timeout": float(os.environ.get("DSH_REQUEST_TIMEOUT", "1200")),
            "wall_seconds": wall_seconds, "pipeline": os.environ["DSH_VERIFIER_ROOT"],
        }
        config_path = episode / "config.json"
        config_path.write_text(json.dumps(config, allow_nan=False))
        config_path.chmod(0o600)
        worker = Path(__file__).with_name("worker.py")
        with (episode / "worker.log").open("w") as log:
            process = await asyncio.create_subprocess_exec(
                os.environ.get("DSH_PYTHON", sys.executable), "-u", str(worker), str(config_path),
                stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
            try:
                await asyncio.wait_for(process.wait(), wall_seconds + float(item.get("verifier_timeout", 120)) + 60)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                if process.returncode is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), 10)
                    except asyncio.TimeoutError:
                        os.killpg(process.pid, signal.SIGKILL)
                        await process.wait()
                raise
        if process.returncode:
            raise RuntimeError(f"DSH worker exited {process.returncode}; inspect {episode / 'worker.log'}")
        result = json.loads((episode / "result.json").read_text())
        return Result(reward=result["reward"], score=result["reward"], truncated=result["truncated"],
                      info={"dsh_episode_dir": str(episode), "dsh_task_id": item["task_id"],
                            "dsh_valid": float(result["valid"]), "dsh_best_objective": result["best_objective"],
                            "dsh_model_requests": result["model_requests"],
                            "dsh_unique_submissions": result["unique_submissions"],
                            "dsh_termination": result["termination"]})


class AgentRunner(ChatAgentRunner):
    def __init__(self):
        super().__init__(DSHAgent)

    async def _ensure_server(self, llm_engine, hf_tokenizer, max_length, sampling_params):
        await super()._ensure_server(llm_engine, hf_tokenizer, max_length, sampling_params)
        prefix = hf_tokenizer.apply_chat_template(
            [{"role": "user", "content": "task"}], tokenize=False, add_generation_prompt=True)
        self.agent_cls = partial(DSHAgent, thinking_prefix=prefix.rstrip().endswith("<think>"))
