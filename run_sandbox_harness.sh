#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  printf 'Usage: bash run_sandbox_harness.sh <command> [args...]\n' >&2
  exit 2
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_file="${SANDBOX_LOCAL_ENV_FILE:-${project_dir}/sandbox.local.env}"

if [[ ! -f "${config_file}" ]]; then
  printf 'Missing local config: %s\nCopy sandbox.local.env.example to sandbox.local.env and fill in API_KEY and SUMMARY_MODEL_NAME.\n' "${config_file}" >&2
  exit 2
fi

set -a
# This file is local-only and ignored by Git.
source "${config_file}"
set +a
export SANDBOX_LOCAL_ENV_FILE="$(cd -- "$(dirname -- "$config_file")" && pwd)/$(basename -- "$config_file")"

export SCIENCE_AGENT_SANDBOX_BASE_URL="${SCIENCE_AGENT_SANDBOX_BASE_URL:-${BASE_URL:-http://10.12.111.133:10018}}"
export SCIENCE_AGENT_READ_PAGE_MODE="${SCIENCE_AGENT_READ_PAGE_MODE:-file}"
export SCIENCE_AGENT_SANDBOX_IMAGE="${SCIENCE_AGENT_SANDBOX_IMAGE:-registry.h.pjlab.org.cn/ailab-ai4sgi-ailab_ai4sgi_pool/yqy:science-agent-cpu-20261010}"
export PYTHON_TOOL_RETURN_MAX_CHARS="${PYTHON_TOOL_RETURN_MAX_CHARS:-50000}"
export SCIENCE_AGENT_TOOLS="${SCIENCE_AGENT_TOOLS:-search,visit,python,google_scholar}"

if [[ -z "${OPEN_SANDBOX_GATEWAY_TOKEN:-}" ]]; then
  printf 'Missing Sandbox Token. Before submitting rjob, run:\npython3 %s/platform_ad_auth.py --save-token %s\n' "${project_dir}" "${config_file}" >&2
  exit 2
fi

exec "$@"
