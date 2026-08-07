#!/bin/bash
set -euo pipefail

MODEL_DIR="$1"; shift
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

eval "$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"

if [[ "${MLX_AUDIO_SUPPORTED:-}" == "false" ]]; then
    echo "Not supported: ${MODEL_NAME} has no MLX-Audio backend." >&2
    exit 1
fi

VENV_PATH="${VENV_PATH:-${ROOT}/.venv-mlx}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: MLX environment not found at ${VENV_PATH}." >&2
    echo "Run ./setup.sh or scripts/setup-mlx.sh first." >&2
    exit 1
fi

PORT="${PORT:-$MODEL_PORT}"
HOST="${HOST:-0.0.0.0}"

echo "Starting ${MODEL_NAME} via MLX-Audio on port ${PORT}..."
exec "${PYTHON_BIN}" "${SCRIPTS_DIR}/mlx-audio-server.py" \
    --model "${MLX_AUDIO_MODEL}" \
    --alias "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-batch-size "${MLX_AUDIO_MAX_BATCH_SIZE}" \
    --sample-rate "${MODEL_SAMPLE_RATE}" \
    --default-voice "${MODEL_DEFAULT_VOICE}" \
    --voices "${MODEL_VOICES}" \
    "$@"
