#!/bin/bash
set -euo pipefail

MODEL_DIR="$1"; shift
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

eval "$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"

if [[ "${PHOTON_SUPPORTED:-}" == "false" ]]; then
    echo "Not supported: ${MODEL_NAME} has no Photon backend." >&2
    exit 1
fi

PYTHON_BIN="${MODEL_DIR}/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Photon environment not found. Run scripts/setup-photon.sh first." >&2
    exit 1
fi

PORT="${PORT:-$MODEL_PORT}"
HOST="${HOST:-0.0.0.0}"

# Keep the default CPU service independent of shared production GPU capacity.
if [[ "$PHOTON_DEVICE" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi

echo "Starting ${MODEL_NAME} via Photon (${PHOTON_DEVICE}) on port ${PORT}..."
exec "$PYTHON_BIN" "${ROOT}/scripts/photon-server.py" \
    --model "$PHOTON_MODEL" --alias "$MODEL_ID" \
    --device "$PHOTON_DEVICE" --threads "$PHOTON_THREADS" \
    --host "$HOST" --port "$PORT" "$@"
