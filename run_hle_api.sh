#!/usr/bin/env bash
# Load tool credentials first, then apply the independent solver/Judge settings.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${HLE_API_ENV_LOADED:-0}" != 1 ]]; then
  exec bash "$project_dir/run_sandbox_harness.sh" env HLE_API_ENV_LOADED=1 \
    bash "$project_dir/run_hle_api.sh" "$@"
fi
unset HLE_API_ENV_LOADED
# Keep solver credentials in a separate ignored file; never replace tool API_KEY.
if [[ -n "${HLE_API_ENV_FILE:-}" ]]; then
  set -a
  source "$HLE_API_ENV_FILE"
  set +a
fi
export SCIENCE_AGENT_JUDGE_BASE_URL="${HLE_JUDGE_BASE_URL:-http://35.220.164.252:3888/v1}"
export SCIENCE_AGENT_JUDGE_MODEL="${HLE_JUDGE_MODEL:-bailian/deepseek-v4.1-flash}"
export SCIENCE_AGENT_JUDGE_THINKING="${HLE_JUDGE_THINKING:-enabled}"
export SCIENCE_AGENT_JUDGE_MAX_TOKENS="${HLE_JUDGE_MAX_TOKENS:-6000}"
export SCIENCE_AGENT_JUDGE_TIMEOUT="${HLE_JUDGE_TIMEOUT:-180}"
export PYTHONPATH="$project_dir:$project_dir/inference${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python3}" "$project_dir/eval_hle_api.py" "$@"
