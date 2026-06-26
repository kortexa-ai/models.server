#!/bin/bash
set -euo pipefail

MODEL_DIR="$1"; shift
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

eval "$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"

if [[ "${CPU_SUPPORTED:-}" == "false" ]]; then
    echo "Not supported: ${MODEL_NAME} is too large for CPU-only inference." >&2
    exit 1
fi

PORT="${PORT:-$MODEL_PORT}"
HOST="${HOST:-0.0.0.0}"
QUANT="${QUANT:-$CPU_QUANT}"
CONTEXT="${CONTEXT:-$CPU_CONTEXT}"

VISION_ARGS=()
if [[ "${MODEL_MULTIMODAL}" != "true" ]]; then
    VISION_ARGS=(--no-mmproj)
fi

FLASH_ATTN_ARGS=()
if [[ -n "${CPU_FLASH_ATTN:-}" ]]; then
    if [[ "${CPU_FLASH_ATTN}" == "true" ]]; then
        FLASH_ATTN_ARGS=(--flash-attn on)
    else
        FLASH_ATTN_ARGS=(--flash-attn off)
    fi
fi

CHECKPOINT_ARGS=()
if [[ -n "${CPU_CHECKPOINT_MIN_STEP:-}" ]]; then
    CHECKPOINT_ARGS=(--checkpoint-min-step "$CPU_CHECKPOINT_MIN_STEP")
fi

echo "Starting ${MODEL_NAME} (CPU, ${QUANT}) via llama-server on port ${PORT}..."
exec llama-server \
    -hf "${CPU_REPO}:${QUANT}" \
    --alias "$MODEL_ID" \
    --host "$HOST" \
    --port "$PORT" \
    --jinja \
    -c "$CONTEXT" \
    --threads 4 \
    --parallel 1 \
    --temp 1.0 \
    --top-p 0.95 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    "${FLASH_ATTN_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VISION_ARGS[@]}" \
    "$@"
