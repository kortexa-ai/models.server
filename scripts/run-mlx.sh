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
SERVER_ARGS=(
    --model "$MLX_REPO"
    --host "$HOST"
    --port "$PORT"
)

if [[ -n "${MLX_DRAFT_MODEL:-}" && "${MLX_DISABLE_DRAFT:-}" != "1" ]]; then
    if [[ "${MLX_DRAFT_ENABLED:-true}" == "false" && "${MLX_FORCE_DRAFT:-}" != "1" ]]; then
        echo "Speculative drafter configured but disabled for vanilla ${BACKEND}; set MLX_FORCE_DRAFT=1 to override."
    else
        echo "Using speculative drafter: ${MLX_DRAFT_MODEL}"
        SERVER_ARGS+=(--draft-model "$MLX_DRAFT_MODEL")

        if [[ -n "${MLX_DRAFT_KIND:-}" ]]; then
            SERVER_ARGS+=(--draft-kind "$MLX_DRAFT_KIND")
        fi

        if [[ -n "${MLX_DRAFT_BLOCK_SIZE:-}" ]]; then
            SERVER_ARGS+=(--draft-block-size "$MLX_DRAFT_BLOCK_SIZE")
        fi
    fi
fi

exec python -m "${BACKEND}.server" "${SERVER_ARGS[@]}" "$@"
