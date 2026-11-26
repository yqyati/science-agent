#!/usr/bin/env bash
# Share the existing solver, Judge and Sandbox configuration with HLE.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$project_dir/run_hle_api.sh" "$@"
