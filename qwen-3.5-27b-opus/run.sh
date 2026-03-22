#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

MODELS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$(uname -s)" == "Darwin" && -d "${MODELS_ROOT}/.venv-mlx" ]]; then
    # shellcheck disable=SC1091
    source "${MODELS_ROOT}/.venv-mlx/bin/activate"
fi

MODEL_MLX="mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
PORT="${PORT:-2032}"
HOST="${HOST:-0.0.0.0}"

OS="$(uname -s)"
HOSTNAME="$(hostname)"

IS_SPARK=false
if [[ "$HOSTNAME" == *"spark"* ]]; then
    IS_SPARK=true
fi

if [[ "$OS" == "Darwin" ]]; then
    echo "Starting mlx-vlm server on port $PORT..."
    python -m mlx_vlm.server \
        --host "$HOST" \
        --port "$PORT" \
        "$@"
elif [[ "$IS_SPARK" == true ]]; then
    # DGX Spark - use vLLM Docker with AutoRound int4
    # NOTE: No AutoRound int4 available for Opus distill, using base 27B int4
    # Memory fraction: 0.4 = ~51GB
    GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.4}"
    # Fall back to standard 27B int4 - Opus distill not available in int4
    MODEL_HF="Qwen/Qwen3.5-27B-Instruct-AutoRound-int4-sym"

    echo "Starting vLLM server on DGX Spark (port $PORT, GPU mem: ${GPU_MEM_FRAC})..."
    echo "NOTE: Using base 27B int4 (Opus distill not available in int4)"
    docker run --rm -it --network host --gpus all \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        vllm/vllm-openai:vllm-node \
        --model "$MODEL_HF" \
        --host "$HOST" \
        --port "$PORT" \
        --max-model-len 8192 \
        --gpu-memory-utilization "$GPU_MEM_FRAC" \
        --enforce-eager \
        "$@"
else
    QUANT="${QUANT:-UD-Q4_K_XL}"
    CACHE_TYPE="${CACHE_TYPE:-q4_0}"
    CONTEXT="${CONTEXT:-262144}"
    PARALLEL="${PARALLEL:-1}"
    echo "Starting llama-server on port $PORT..."
    llama-server -hf "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF:${QUANT}" --alias qwen-3.5-27b-opus --host "$HOST" --port "$PORT" \
        --jinja -c "$CONTEXT" -ngl 99 --threads -1 --parallel "$PARALLEL" \
        --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40 \
        --no-mmap --flash-attn on \
        --cache-type-k "$CACHE_TYPE" --cache-type-v "$CACHE_TYPE" \
        "$@"
fi
