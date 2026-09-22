#!/usr/bin/env bash
# The shared helper snapshots all potential production 27B variants.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ "$(hostname)" == smarty ]] || { echo 'Run on smarty.' >&2; exit 1; }
PROD_SNAPSHOT="$(ktxsvc list)"
printf '%s\n' "$PROD_SNAPSHOT"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.free,power.limit --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
bash "$HOME/src/legolm/scripts/smarty_gpu_block.sh" --until-free 80 \
    python3 "$ROOT/bench/cinference-6000/run.py" "$@"
