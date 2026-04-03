#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Gemma 4 26B-A4B via llama-server (llama.cpp)
# Architecture: MoE, 26B total / 4B active, up to 256K native context
# GGUF Q8_0 is ~28GB — fits easily in 96GB with tons left for KV cache
# With f16 KV cache and 128K context: ~28GB model + ~40GB KV = ~68GB total

QUANT="${QUANT:-Q8_0}"
PORT="${PORT:-2036}"
HOST="${HOST:-0.0.0.0}"
CONTEXT="${CONTEXT:-131072}"
PARALLEL="${PARALLEL:-8}"
CACHE_TYPE="${CACHE_TYPE:-f16}"

HF_REPO="unsloth/gemma-4-26B-A4B-it-GGUF"

echo "Starting Gemma 4 26B-A4B (GGUF ${QUANT}) via llama-server"
echo "Port: ${PORT}"
echo "Context: ${CONTEXT}"
echo "Cache type: ${CACHE_TYPE}"
echo "HF repo: ${HF_REPO}:${QUANT}"

exec llama-server \
    -hf "${HF_REPO}:${QUANT}" \
    --alias gemma-4-26b-a4b \
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
