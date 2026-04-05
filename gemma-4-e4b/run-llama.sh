#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Gemma 4 E4B (effective 4B, dense with PLE) via llama-server
# Architecture: Dense with Per-Layer Embeddings, up to 128K native context
# GGUF Q8_0 is ~5GB — trivially fits in VRAM

QUANT="${QUANT:-Q8_0}"
PORT="${PORT:-2038}"
HOST="${HOST:-0.0.0.0}"
CONTEXT="${CONTEXT:-131072}"
PARALLEL="${PARALLEL:-8}"
CACHE_TYPE="${CACHE_TYPE:-f16}"

HF_REPO="unsloth/gemma-4-E4B-it-GGUF"

echo "Starting Gemma 4 E4B (GGUF ${QUANT}) via llama-server"
echo "Port: ${PORT}"
echo "Context: ${CONTEXT}"
echo "Cache type: ${CACHE_TYPE}"
echo "HF repo: ${HF_REPO}:${QUANT}"

exec llama-server \
    -hf "${HF_REPO}:${QUANT}" \
    --alias gemma-4-e4b \
    --host "$HOST" \
    --port "$PORT" \
    --jinja \
    -c "$CONTEXT" \
    -ngl 99 \
    --threads -1 \
    --parallel "$PARALLEL" \
    --temp 1.0 \
    --top-p 0.95 \
    --no-mmap \
    --flash-attn on \
    --cache-type-k "$CACHE_TYPE" \
    --cache-type-v "$CACHE_TYPE" \
    "$@"
