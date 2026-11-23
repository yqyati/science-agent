"""Real Molt capture + native DSH + real verifier, with a CPU fake token policy."""
import asyncio
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
from jinja2.sandbox import ImmutableSandboxedEnvironment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "molt_adapter"))
sys.path.insert(0, str(ROOT.parents[1] / "labs-molt"))


def accept():
    import torch
    from agent import AgentRunner
    pipeline = os.environ["DSH_VERIFIER_ROOT"]
    item = {"task_id": "adapter-synthetic-C4-2-1", "generation_split": "train",
            "direction": "min", "reward_anchor": 3, "reward_scale": 1,
            "task": {"task_id": "adapter-synthetic-C4-2-1", "family": "covering_design",
                     "parameters": {"v": 4, "k": 2, "t": 1}, "verifier": {"kind": "ljcr_exact"}}}
    template_path = Path(os.environ.get("DSH_ACCEPTANCE_CHAT_TEMPLATE",
        "/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/science_model/tool_models/"
        "K3_p1_rollout88+nemotron_science_four_tools-0930-063459/checkpoint/hf_iter_0000549/chat_template.jinja"))
    template = ImmutableSandboxedEnvironment().from_string(template_path.read_text())

    class Processor:
        image_token = "<image>"

        def apply_chat_template(self, chat, **kwargs):
            return template.render(messages=chat, **kwargs)

        def __call__(self, text=None, **kwargs):
            return {"input_ids": torch.tensor([[ord(c) for c in text]])}

        def decode(self, ids, **kwargs):
            return "".join(chr(i) for i in ids)

    command = "printf '%s' '{\"blocks\":[[1,2],[3,4]]}' > submission.txt\npython3 score_submission.py submission.txt"
    actions = [
        "Construct then verify.</think><tool_call><function=bash>"
        "<parameter=command>echo FIRST_TOOL_RESULT_SENTINEL</parameter>"
        "<parameter=description>First independent tool</parameter></function></tool_call>"
        "<tool_call><function=bash>"
        f"<parameter=command>{command}</parameter>"
        "<parameter=description>Verify covering</parameter></function></tool_call>",
        'Verified.</think>OUTPUT:\n{"blocks":[[1,2],[3,4]]}\nSOLUTION:\nAll points covered.',
    ]

    class Policy:
        def __init__(self):
            self.prompts = []

        async def generate(self, prompt_token_ids, sampling_params, **kwargs):
            text = actions[len(self.prompts)]
            self.prompts.append(list(prompt_token_ids))
            ids = list(map(ord, text))
            return SimpleNamespace(outputs=[SimpleNamespace(
                token_ids=ids, text="", finish_reason="stop", routed_experts=None,
                logprobs=[{token: SimpleNamespace(logprob=-0.25)} for token in ids])],
                prompt_routed_experts=None), 0

    episodes = Path(os.environ.get("DSH_TEST_RUN_ROOT", ROOT / "runs")) / "molt_native_acceptance"
    episodes.mkdir(parents=True, exist_ok=True)
    episodes.chmod(0o755)

    async def run():
        runner = AgentRunner()
        policy = Policy()
        try:
            trajectories = await runner.execute(
                [{"role": "user", "content": "Construct a (4,2,1) covering."}], item,
                SimpleNamespace(max_tokens=2048, temperature=1.0, top_p=1.0, logprobs=1, min_tokens=0),
                65536, Processor(), policy)
            assert len(policy.prompts) == 2
            assert '"valid": true' in Processor().decode(policy.prompts[1])
            assert "<tool_response>\nFIRST_TOOL_RESULT_SENTINEL\n</tool_response>" in Processor().decode(policy.prompts[1])
            assert Processor().decode(policy.prompts[1]).count("<tool_response>") >= 2
            generated = []
            for trajectory in trajectories:
                assert trajectory.reward > 0.5
                assert len(trajectory.rollout_log_probs) == len(trajectory.observation_tokens)
                for start, end in trajectory.action_ranges:
                    generated.append(Processor().decode(trajectory.observation_tokens[start:end]))
                    assert trajectory.rollout_log_probs[start:end] == [-0.25] * (end - start)
            assert generated == actions
            (episodes / "acceptance.json").write_text(json.dumps({
                "model_requests": len(policy.prompts), "segments": len(trajectories),
                "exact_action_tokens": True, "aligned_logprobs": True, "real_verifier": True,
                "multiple_tool_results_preserved": True, "real_qwen_chat_template": str(template_path),
                "reward": trajectories[0].reward, "gpu_training": False}, indent=2) + "\n")
        finally:
            if runner._server_task:
                runner._server_task.cancel()
                try:
                    await runner._server_task
                except asyncio.CancelledError:
                    pass

    with patch.dict(os.environ, {"DSH_EPISODE_ROOT": str(episodes), "DSH_VERIFIER_ROOT": pipeline,
                                 "DSH_WALL_SECONDS": "45", "DSH_MAX_REQUESTS": "4"}):
        asyncio.run(run())


if __name__ == "__main__":
    accept()
