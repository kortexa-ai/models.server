#!/bin/bash
# CPU-only llama-server for Raspberry Pi (8GB)
set -euo pipefail

cd "$(dirname "$0")"

QUANT="${QUANT:-Q4_K_M}"
CACHE_TYPE="${CACHE_TYPE:-q4_0}"
CONTEXT="${CONTEXT:-4096}"
PORT="${PORT:-2029}"
HOST="${HOST:-0.0.0.0}"

echo "Starting llama-server (CPU) on port $PORT..."
llama-server -hf "unsloth/Qwen3.5-4B-GGUF:${QUANT}" --alias qwen-3.5-4b \
    --host "$HOST" --port "$PORT" \
    --jinja -c "$CONTEXT" -ngl 0 --threads 4 --parallel 1 \
    --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40 \
    --cache-type-k "$CACHE_TYPE" --cache-type-v "$CACHE_TYPE" \
    "$@"
