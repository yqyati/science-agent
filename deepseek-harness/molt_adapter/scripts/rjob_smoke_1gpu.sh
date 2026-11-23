#!/usr/bin/env bash
# Validate the existing training image, CUDA, and native DSH/Molt integration.
set -euo pipefail
ADAPTER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="$(cd "$ADAPTER/.." && pwd)"
WORKSPACE="$(cd "$HARNESS/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$WORKSPACE/labs-molt:$ADAPTER${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DSH_VERIFIER_ROOT="${DSH_VERIFIER_ROOT:-/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/sujiaming/DSH_Teacher3536_Handoff_20261113/teacher_rollout/verifier_sources}"
export DSH_TEST_RUN_ROOT="${DSH_TEST_RUN_ROOT:-$HARNESS/runs/${RUN_NAME:?Set RUN_NAME}}"
export DSH_RUN_NATIVE_TESTS=1
mkdir -p "$DSH_TEST_RUN_ROOT"
chmod o+x "$DSH_TEST_RUN_ROOT"
exec > >(tee "$DSH_TEST_RUN_ROOT/job.log") 2>&1
trap 'status=$?; echo "DSH_SMOKE_EXIT=$status"; date -Is' EXIT
date -Is
"$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch

assert torch.cuda.is_available(), 'CUDA is not available in the test job'
x = torch.ones((64, 64), device='cuda')
assert (x @ x)[0, 0].item() == 64
torch.cuda.synchronize()
report = {
    'python': sys.executable, 'python_version': sys.version,
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'gpu_count': torch.cuda.device_count(), 'gpu': torch.cuda.get_device_name(0),
    'cuda_matmul': 'passed', 'real_model_inference': False, 'rl_training': False,
    'packages': {name: importlib.util.find_spec(name) is not None for name in
                 ('pytest', 'pydantic', 'openai', 'ray', 'vllm', 'numpy', 'scipy', 'sympy')},
}
Path(os.environ['DSH_TEST_RUN_ROOT'], 'environment.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
"$PYTHON_BIN" "$ADAPTER/preflight.py"
timeout 240 "$PYTHON_BIN" "$ADAPTER/acceptance.py"
echo 'DSH_SMOKE_PASSED: CUDA + filesystem isolation + native tool/verifier + Molt token/logprob capture'
