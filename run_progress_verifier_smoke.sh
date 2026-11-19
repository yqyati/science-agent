#!/usr/bin/env bash
# API-only smoke: no GPU allocation. All defaults below apply to this smoke only.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${PROGRESS_SMOKE_ENV_LOADED:-0}" != 1 ]]; then
  exec bash "$project_dir/run_sandbox_harness.sh" env PROGRESS_SMOKE_ENV_LOADED=1 \
    bash "$project_dir/run_progress_verifier_smoke.sh" "$@"
fi
unset PROGRESS_SMOKE_ENV_LOADED
export PROGRESS_VERIFIER_BASE_URL="${PROGRESS_VERIFIER_BASE_URL:-$API_BASE}"
export PROGRESS_VERIFIER_MODEL="${PROGRESS_VERIFIER_MODEL:-$SUMMARY_MODEL_NAME}"
export PROGRESS_VERIFIER_API_KEY="${PROGRESS_VERIFIER_API_KEY:-$API_KEY}"
export PYTHONPATH="$project_dir:$project_dir/inference${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python3}" "$project_dir/smoke_progress_verifier.py" "$@"
