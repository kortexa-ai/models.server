#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE=start
if [[ "${1:-}" == "--stop" ]]; then
    MODE=stop
    shift
fi
MODEL_CONFIG="${1:-${ROOT}/qwen-3.8-flash-next-fast/model.json}"
if [[ -d "$MODEL_CONFIG" ]]; then
    MODEL_CONFIG="${MODEL_CONFIG}/model.json"
fi

read_config() {
    python3 - "$MODEL_CONFIG" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value[key]
if isinstance(value, bool):
    print("1" if value else "0")
elif isinstance(value, (dict, list)):
    print(json.dumps(value, separators=(",", ":")))
else:
    print(value)
PY
}

RECIPE_REPO="$(read_config tensorfold.recipe_repo)"
RECIPE_REVISION="$(read_config tensorfold.recipe_revision)"
export MODEL_ID="$(read_config tensorfold.checkpoint)"
export SERVED_NAME="$(read_config id)"
export PORT="$(read_config port)"
export HOST="$(read_config host)"
export PARALLEL="$(read_config tensorfold.parallel)"
export CONTEXT="$(read_config tensorfold.context)"
export KV_DTYPE="$(read_config tensorfold.kv_dtype)"
export PLE_ON_SSD="$(read_config tensorfold.ple_on_ssd)"
export MTP_DRAFTS="$(read_config tensorfold.mtp_drafts)"
export MTP_CONFIDENCE="$(read_config tensorfold.mtp_confidence)"
export THINKING="$(read_config tensorfold.thinking)"
export VISION="$(read_config tensorfold.vision)"
export VISION_URLS="$(read_config tensorfold.vision_urls)"
export CONTAINER_NAME="$(read_config tensorfold.container_name)"

if [[ "$(hostname -s)" != static ]]; then
    echo "TensorFold fast model is configured for static only." >&2
    exit 2
fi
if [[ ! -d "${RECIPE_REPO}/.git" || "$(git -C "$RECIPE_REPO" rev-parse HEAD)" != "$RECIPE_REVISION" ]]; then
    echo "TensorFold recipe checkout is missing or differs from pinned revision ${RECIPE_REVISION}." >&2
    exit 3
fi

cd "$RECIPE_REPO"
if [[ "$MODE" == stop ]]; then
    exec ./stop.sh
fi
export FOREGROUND=1
exec ./start.sh
