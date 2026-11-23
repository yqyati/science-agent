#!/usr/bin/env bash
# Independent DSH recipe; run inside an existing eight-GPU Molt training image.
set -euo pipefail
ADAPTER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="$(cd "$ADAPTER/.." && pwd)"
WORKSPACE="$(cd "$HARNESS/../.." && pwd)"
export MOLT_PATH="${MOLT_PATH:-$WORKSPACE/labs-molt}"
export PYTHONPATH="$MOLT_PATH${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${MODEL_PATH:?Set MODEL_PATH to the selected HF training checkpoint}"
: "${DATASET:?Set DATASET to the converted DSH training JSONL}"
export DSH_VERIFIER_ROOT="${DSH_VERIFIER_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/sujiaming/DSH_Teacher3536_Handoff_20261113/teacher_rollout/verifier_sources}"
RUN_NAME="${RUN_NAME:-dsh_single_rl_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/mnt/shared-storage-user/ai4sgi-gpfs1/yangqingyu/workspace/harness/deepseek-harness/runs/$RUN_NAME}"
export MOLT_CKPT_FALLBACK_ROOT="${MOLT_CKPT_FALLBACK_ROOT-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/yangqingyu/molt_checkpoints:/mnt/shared-storage-user/yangqingyu/molt_checkpoints}"
export DSH_EPISODE_ROOT="${DSH_EPISODE_ROOT:-$RUN_ROOT/episodes}"
export DSH_WALL_SECONDS="${DSH_WALL_SECONDS:-1800}"
export DSH_MAX_COMPLETION_TOKENS="${DSH_MAX_COMPLETION_TOKENS:-16384}"
MAX_LEN="${MAX_LEN:-65536}"
MAX_SAMPLES="${MAX_SAMPLES:-2399}"
VERIFIER_ENTRY="$DSH_VERIFIER_ROOT/tools/verify_task.py"
[[ ! -f "$DSH_VERIFIER_ROOT/verify_candidate.py" ]] || VERIFIER_ENTRY="$DSH_VERIFIER_ROOT/verify_candidate.py"
for path in "$MODEL_PATH/config.json" "$DATASET" "$VERIFIER_ENTRY"; do
  [[ -f "$path" ]] || { echo "Missing: $path" >&2; exit 1; }
done
[[ "${PARTIAL_ROLLOUT_ENABLE:-1}" == 1 ]] || { echo 'This recipe requires partial rollout.' >&2; exit 1; }
[[ "${DISABLE_FINAL_SAVE:-0}" != 1 ]] || { echo 'Final model export must remain enabled.' >&2; exit 1; }
mkdir -p "$RUN_ROOT" "$DSH_EPISODE_ROOT"
# Reserve new checkpoints; never overwrite an earlier run.
mkdir "$RUN_ROOT/checkpoint"
chmod o+x "$RUN_ROOT" "$DSH_EPISODE_ROOT"
export WANDB_DISABLED=true RAY_USAGE_STATS_ENABLED=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLT_MOE_RESHARD_AFTER_FWD=0 MOLT_MOE_DISPATCHER=hybridep MOLT_DEFER_GRAD_SYNC=1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PYTHON_BIN" "$ADAPTER/preflight.py"
cd "$MOLT_PATH"
"$PYTHON_BIN" -u "$ADAPTER/train.py" \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$DATASET" --data.input_key prompt --data.label_key dsh_item --data.apply_chat_template \
  --data.max_samples "$MAX_SAMPLES" --data.max_len "$MAX_LEN" \
  --rollout.batch_size "${ROLLOUT_BATCH_SIZE:-8}" --rollout.n_samples_per_prompt "${N_SAMPLES_PER_PROMPT:-8}" \
  --rollout.vllm_generate_batch_size "${ROLLOUT_VLLM_GENERATE_BATCH_SIZE:-6}" --rollout.micro_batch_size 1 \
  --rollout.max_new_tokens "$MAX_LEN" --rollout.temperature 1.0 --rollout.top_p 1.0 \
  --train.batch_size "${TRAIN_BATCH_SIZE:-64}" --train.micro_batch_size 1 \
  --train.max_tokens_per_gpu "${TRAIN_MAX_TOKENS_PER_GPU:-25000}" \
  --train.max_epochs 1 --train.num_episodes 1 --train.async_queue_size 2 --train.partial_rollout_enable \
  --train.colocate_fsdp_models --train.force_on_policy --train.agent_path "$ADAPTER/agent.py" \
  --train.rollout_dump_dir "$RUN_ROOT/rollout_dump" \
  --actor.num_nodes 1 --actor.num_gpus_per_node 4 \
  --vllm.num_engines 1 --vllm.tensor_parallel_size 4 --vllm.sync_backend nccl \
  --vllm.enable_auto_tool_choice --vllm.tool_call_parser qwen3_coder --vllm.enable_prefix_caching \
  --vllm.gpu_memory_utilization 0.8 --vllm.distributed_executor_backend mp \
  --fsdp.param_dtype bf16 --fsdp.attn_implementation te --fsdp.tp_size 1 --fsdp.ep_size 4 --fsdp.cp_size 4 --fsdp.offload optimizer \
  --actor.gradient_checkpoint full --actor.adam.lr 1e-6 --actor.freeze_moe_router --actor.freeze_visual_encoder --actor.log_entropy \
  --actor.loss_mode ppo --actor.loss_agg_mode token-mean --actor.eps_clip_low_high 0.2 0.27 --actor.entropy_coef 0 \
  --algo.advantage.estimator reinforce_baseline --algo.advantage.is_correction_level geo \
  --algo.advantage.is_correction_gating ratio --algo.advantage.is_correction_mode mask \
  --algo.advantage.is_correction_threshold 0.5 5.0 --algo.advantage.score_centering_top_k 0 \
  --algo.kl.estimator k1 --algo.kl.init_coef 0 --reward.clip_range -10 10 \
  --ckpt.output_dir "$RUN_ROOT/checkpoint/hf" --ckpt.path "$RUN_ROOT/checkpoint/state" \
  --ckpt.save_steps 50 --ckpt.save_hf --ckpt.max_num 0 --no-ckpt.save_optimizer \
  --logger.logging_steps 1 --logger.tensorboard_dir "$RUN_ROOT/tensorboard" \
  "$@" 2>&1 | tee "$RUN_ROOT/train.log"
