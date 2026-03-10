#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

MODELS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -d ".venv-mlx" ]]; then
    source .venv-mlx/bin/activate
elif [[ -d "${MODELS_ROOT}/.venv-mlx" ]]; then
    # shellcheck disable=SC1091
    source "${MODELS_ROOT}/.venv-mlx/bin/activate"
fi


MODEL="mlx-community/Nanbeige4.1-3B-8bit"
PORT=2025
HOST="0.0.0.0"

OS="$(uname -s)"

if [[ "$OS" == "Darwin" ]]; then
    # macOS - use MLX-LM
    echo "Starting MLX-LM server on port $PORT..."
    python -m mlx_lm.server \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --max-tokens 4096 \
        --trust-remote-code \
        "$@"
else
    # Linux - use vLLM
    # Detect if CUDA is available
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        MAX_LEN=131072
        DTYPE="bfloat16"
        echo "Starting vLLM server (CUDA) on port $PORT..."
    else
        MAX_LEN=2048
        DTYPE="float32"
        echo "Starting vLLM server (CPU) on port $PORT..."
    fi

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization 0.12 \
        --trust-remote-code \
        "$@"
fi
