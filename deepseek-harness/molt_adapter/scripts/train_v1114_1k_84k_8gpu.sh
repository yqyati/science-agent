#!/usr/bin/env bash
# Single DSH episodes; the existing asynchronous 4 actor + 4 rollout recipe.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS="$(cd "$SCRIPT_DIR/../.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/science_model/dsh_model/sft_qwen3.5-35b_p1_flash1017_1.6k/hf_epoch_weights/rollout_489}"
export DATASET="$HARNESS/data/verifiable_1k_v451p9/train.molt.jsonl"
export DSH_VERIFIER_ROOT="$HARNESS/data/verifiable_1k_v451p9/verifier"
export MAX_SAMPLES=1000 MAX_LEN=86016
export N_SAMPLES_PER_PROMPT=8 ROLLOUT_BATCH_SIZE=8 TRAIN_BATCH_SIZE=64
export ROLLOUT_VLLM_GENERATE_BATCH_SIZE=6 PARTIAL_ROLLOUT_ENABLE=1
export RUN_NAME="${RUN_NAME:-dsh_v1114_1k_qwen35b_sft16k_r489_84k_48conc_$(date +%Y%m%d_%H%M%S)}"
"${PYTHON_BIN:-python3}" -B - <<'PY'
import hashlib
import json
import os
from pathlib import Path

data = Path(os.environ['DATASET'])
manifest = json.loads(data.with_suffix('.provenance.json').read_text())
assert manifest['records'] == 1000
assert hashlib.sha256(data.read_bytes()).hexdigest() == manifest['sha256'], 'Dataset changed'
root = Path(os.environ['DSH_VERIFIER_ROOT'])
for name, digest in manifest['verifier_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, f'Verifier changed: {name}'
assert not list(root.rglob('trusted_specs.jsonl')), 'Private QA files must not enter the snapshot'
print('Frozen 1000-question dataset and exact verifier hashes: OK')
PY
# Reward above 0.5 means improvement over the task's baseline objective.
# Length counts generated action tokens, excluding prompts and tool results.
exec bash "$SCRIPT_DIR/train_4plus4.sh" \
  --reward.group_length_penalty.enable \
  --reward.group_length_penalty.success_threshold 0.5 \
  --reward.group_length_penalty.min_pass_rate 0.5 \
  --reward.group_length_penalty.quantile 0.5 \
  --reward.group_length_penalty.max_penalty 0.1 \
  --reward.group_length_penalty.tolerance 0.2 \
  --reward.group_length_penalty.saturation 1.0 \
  "$@"
