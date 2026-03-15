#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

MODELS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$(uname -s)" == "Darwin" && -d "${MODELS_ROOT}/.venv-mlx" ]]; then
    # shellcheck disable=SC1091
    source "${MODELS_ROOT}/.venv-mlx/bin/activate"
fi


MODEL_MLX="mlx-community/Qwen3.5-35B-A3B-4bit"
PORT="${PORT:-2027}"
HOST="${HOST:-0.0.0.0}"

OS="$(uname -s)"

if [[ "$OS" == "Darwin" ]]; then
    # macOS - use mlx-vlm
    # Model is loaded on first request via the "model" field in the request body.
    # Clients should send model="mlx-community/Qwen3.5-35B-A3B-4bit"
    echo "Starting mlx-vlm server on port $PORT..."
    python -m mlx_vlm.server \
        --host "$HOST" \
        --port "$PORT" \
        "$@"
else
    # Linux - use llama-server with GGUF
    QUANT="${QUANT:-UD-Q4_K_XL}"
    CACHE_TYPE="${CACHE_TYPE:-q4_0}"
    CONTEXT="${CONTEXT:-65536}"
    PARALLEL="${PARALLEL:-1}"
    echo "Starting llama-server on port $PORT..."
    llama-server -hf "unsloth/Qwen3.5-35B-A3B-GGUF:${QUANT}" --alias qwen-3.5-35b-a3b --host "$HOST" --port "$PORT" \
        --jinja -c "$CONTEXT" -ngl 99 --threads -1 --parallel "$PARALLEL" \
        --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40 \
        --no-mmap --flash-attn on \
        --cache-type-k "$CACHE_TYPE" --cache-type-v "$CACHE_TYPE" \
        "$@"
fi
