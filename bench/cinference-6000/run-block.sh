#!/usr/bin/env bash
# Adapt the existing process-owned restore helper to the observed production LM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ "$(hostname)" == smarty ]] || { echo 'Run on smarty.' >&2; exit 1; }
PROD_SNAPSHOT="$(ktxsvc list)"
printf '%s\n' "$PROD_SNAPSHOT"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.free,power.limit --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
PROD_MODELS="$(printf '%s\n' "$PROD_SNAPSHOT" | awk '$1 ~ /^(qwen-3.8-27b|bonsai-2-27b)$/ && $4 == "yes" {print $1}')"
case "$PROD_MODELS" in
  bonsai-2-27b)
    bash <(sed -e 's@models/qwen-3.8-27b@models/bonsai-2-27b@g' \
      -e 's@2053/health@2062/health@g' \
      "$HOME/src/legolm/scripts/smarty_gpu_block.sh") --until-free 80 \
      python3 "$ROOT/bench/cinference-6000/run.py" "$@"
    ;;
  qwen-3.8-27b|'')
    bash "$HOME/src/legolm/scripts/smarty_gpu_block.sh" --until-free 80 \
      python3 "$ROOT/bench/cinference-6000/run.py" "$@"
    ;;
  *) echo 'Ambiguous production 27B set.' >&2; exit 1 ;;
esac
