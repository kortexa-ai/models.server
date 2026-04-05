#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Gemma 4 31B (dense) via llama-server (llama.cpp)
# Architecture: Dense, 31B params all active, up to 256K native context
# GGUF Q8_0 is ~33GB — fits in 96GB with plenty for KV cache

QUANT="${QUANT:-Q8_0}"
PORT="${PORT:-2037}"
HOST="${HOST:-0.0.0.0}"
CONTEXT="${CONTEXT:-131072}"
PARALLEL="${PARALLEL:-4}"
CACHE_TYPE="${CACHE_TYPE:-q8_0}"

HF_REPO="unsloth/gemma-4-31B-it-GGUF"

echo "Starting Gemma 4 31B (GGUF ${QUANT}) via llama-server"
echo "Port: ${PORT}"
echo "Context: ${CONTEXT}"
echo "Cache type: ${CACHE_TYPE}"
echo "HF repo: ${HF_REPO}:${QUANT}"

exec llama-server \
    -hf "${HF_REPO}:${QUANT}" \
    --alias gemma-4-31b \
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
