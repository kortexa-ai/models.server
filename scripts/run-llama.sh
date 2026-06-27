#!/bin/bash
set -euo pipefail

MODEL_DIR="$1"; shift
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

eval "$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"

if [[ "${LLAMA_SUPPORTED:-}" == "false" ]]; then
    echo "Not supported: ${MODEL_NAME} has no GGUF available for llama.cpp." >&2
    exit 1
fi

PORT="${PORT:-$MODEL_PORT}"
HOST="${HOST:-0.0.0.0}"
QUANT="${QUANT:-$LLAMA_QUANT}"
CACHE_TYPE="${CACHE_TYPE:-$MODEL_CACHE_TYPE}"
CONTEXT="${CONTEXT:-$LLAMA_CONTEXT}"
PARALLEL="${PARALLEL:-$LLAMA_PARALLEL}"

VISION_ARGS=()
if [[ "${MODEL_MULTIMODAL}" != "true" ]]; then
    VISION_ARGS=(--no-mmproj)
fi

# MTP speculative decoding: enabled when the GGUF has MTP heads (e.g. unsloth's *-MTP-GGUF).
# https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
MTP_ARGS=()
if [[ "${LLAMA_MTP:-}" == "true" ]]; then
    MTP_ARGS=(--spec-type draft-mtp)
    if [[ -n "${LLAMA_MTP_N_MAX:-}" ]]; then
        MTP_ARGS+=(--spec-draft-n-max "$LLAMA_MTP_N_MAX")
    fi
fi

# Embedding mode: enable --embedding and drop chat/sampling-only flags
# (--jinja, --temp, --top-p are inert here; skipped for clarity).
EMBEDDING_ARGS=()
if [[ "${MODEL_EMBEDDING:-false}" == "true" ]]; then
    EMBEDDING_ARGS=(--embedding)
    echo "Starting ${MODEL_NAME} (GGUF ${QUANT}) via llama-server in EMBEDDING mode on port ${PORT}..."
    exec llama-server \
        -hf "${LLAMA_REPO}:${QUANT}" \
        --alias "$MODEL_ID" \
        --host "$HOST" \
        --port "$PORT" \
        -c "$CONTEXT" \
        -ngl 99 \
        --threads -1 \
        --parallel "$PARALLEL" \
        --no-mmap \
        --flash-attn on \
        --cache-type-k "$CACHE_TYPE" \
        --cache-type-v "$CACHE_TYPE" \
        "${EMBEDDING_ARGS[@]}" \
        "${VISION_ARGS[@]}" \
        "$@"
fi

echo "Starting ${MODEL_NAME} (GGUF ${QUANT}) via llama-server on port ${PORT}..."
exec llama-server \
    -hf "${LLAMA_REPO}:${QUANT}" \
    --alias "$MODEL_ID" \
    --host "$HOST" \
    --port "$PORT" \
    --jinja \
    -c "$CONTEXT" \
    -ngl 99 \
    --threads -1 \
    --parallel "$PARALLEL" \
    --no-context-shift \
    --temp 0.6 \
    --top-k 20 \
    --top-p 0.95 \
    --no-mmap \
    --flash-attn on \
    --cache-type-k "$CACHE_TYPE" \
    --cache-type-v "$CACHE_TYPE" \
    "${MTP_ARGS[@]}" \
    "${VISION_ARGS[@]}" \
    "$@"
