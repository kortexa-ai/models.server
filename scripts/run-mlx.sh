#!/bin/bash
set -euo pipefail

MODEL_DIR="$1"; shift
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

eval "$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"

if [[ "${MLX_SUPPORTED:-}" == "false" ]]; then
    echo "Not supported: ${MODEL_NAME} is not available for MLX." >&2
    exit 1
fi

if [[ -d "${ROOT}/.venv-mlx" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/.venv-mlx/bin/activate"
fi

PORT="${PORT:-$MODEL_PORT}"
HOST="${HOST:-0.0.0.0}"

BACKEND="${MLX_BACKEND:-mlx_vlm}"

echo "Starting ${MODEL_NAME} via ${BACKEND} on port ${PORT}..."
exec python -m "${BACKEND}.server" \
    --host "$HOST" \
    --port "$PORT" \
    "$@"
