#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# venv only required on macOS (mlx-vlm/mlx-lm); Linux uses llama-server
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi


MODEL_MLX="mlx-community/Qwen3.5-4B-MLX-4bit"
PORT=2029
HOST="0.0.0.0"

OS="$(uname -s)"

if [[ "$OS" == "Darwin" ]]; then
    # macOS - use mlx-vlm
    # Model is loaded on first request via the "model" field in the request body.
    echo "Starting mlx-vlm server on port $PORT..."
    python -m mlx_vlm.server \
        --host "$HOST" \
        --port "$PORT" \
        "$@"
else
    # Linux - use llama-server with GGUF
    QUANT="Q4_K_M"
    CACHE_TYPE="q4_0"
    echo "Starting llama-server on port $PORT..."
    llama-server -hf unsloth/Qwen3.5-4B-GGUF:$QUANT --alias qwen-3.5-4b --host "$HOST" --port $PORT \
        --jinja -ngl 99 --threads -1 \
        --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40 \
        --no-mmap --flash-attn on \
        --cache-type-k $CACHE_TYPE --cache-type-v $CACHE_TYPE \
        "$@"
fi
