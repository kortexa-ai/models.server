#!/bin/bash
set -euo pipefail

MODEL_DIR="$1"; shift
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

eval "$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"

if [[ "${TRANSFORMERS_SUPPORTED:-}" == "false" ]]; then
    echo "Not supported: ${MODEL_NAME} has no Transformers backend." >&2
    exit 1
fi

if [[ -n "${VENV_PATH:-}" ]]; then
    PYTHON_BIN="${VENV_PATH}/bin/python"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv/bin/python"
elif [[ "$(uname -s)" == "Darwin" && -x "${ROOT}/.venv-mlx/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv-mlx/bin/python"
elif [[ -x "${ROOT}/.venv-vllm/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv-vllm/bin/python"
else
    echo "Transformers environment not found. Run scripts/setup-transformers.sh first." >&2
    exit 1
fi

PORT="${PORT:-$MODEL_PORT}"
HOST="${HOST:-0.0.0.0}"

TRUST_ARGS=()
if [[ "${TRANSFORMERS_TRUST_REMOTE_CODE:-}" == "true" ]]; then
    TRUST_ARGS=(--trust-remote-code)
fi

echo "Starting ${MODEL_NAME} (${TRANSFORMERS_TASK}, CPU) via Transformers on port ${PORT}..."
CUDA_VISIBLE_DEVICES="" exec "${PYTHON_BIN}" "${SCRIPTS_DIR}/transformers-server.py" \
    --model "${TRANSFORMERS_MODEL}" \
    --alias "${MODEL_ID}" \
    --task "${TRANSFORMERS_TASK}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-length "${TRANSFORMERS_MAX_LENGTH}" \
    --threads "${TRANSFORMERS_THREADS}" \
    --top-k "${TRANSFORMERS_TOP_K}" \
    ${TRUST_ARGS[@]+"${TRUST_ARGS[@]}"} \
    "$@"
