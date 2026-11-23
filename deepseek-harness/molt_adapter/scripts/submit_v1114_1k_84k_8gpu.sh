#!/usr/bin/env bash
# Use --dry-run to inspect the submission command without creating a job.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="${RUN_NAME:-dsh-v1114-1k-84k-48conc-$(date +%m%d-%H%M%S)}"
RJOB_NAME="${RJOB_NAME:-$RUN_NAME}"
IMAGE="${IMAGE:-registry.h.pjlab.org.cn/ailab-ai4sgi-ailab_ai4sgi_pool/yqy:molt-cu13-runtime}"
dry_run=0
if [[ "${1:-}" == --dry-run ]]; then dry_run=1; shift; fi
command=(rjob submit --name="$RJOB_NAME" --charged-group=ailab_ai4sgi_pool --priority=9
  --namespace=ailab-ai4sgi --gpu=8 --memory=768000 --cpu=48 -P 1
  -e DISTRIBUTED_JOB=true -e "RUN_NAME=$RUN_NAME" --image="$IMAGE"
  --mount=gpfs://gpfs2/ai4sgi-gpfs2:/mnt/shared-storage-gpfs2/ai4sgi-gpfs2)
for name in MODEL_PATH PYTHON_BIN DSH_WALL_SECONDS DSH_MAX_REQUESTS DSH_MAX_COMPLETION_TOKENS DSH_REQUEST_TIMEOUT; do
  if [[ -n "${!name:-}" ]]; then command+=(-e "$name=${!name}"); fi
done
command+=(-- bash "$SCRIPT_DIR/train_v1114_1k_84k_8gpu.sh" "$@")
if [[ "$dry_run" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
source /etc/profile.d/ssh-init.sh
exec "${command[@]}"
