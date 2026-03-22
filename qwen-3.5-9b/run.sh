#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

MODELS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$(uname -s)" == "Darwin" && -d "${MODELS_ROOT}/.venv-mlx" ]]; then
    # shellcheck disable=SC1091
    source "${MODELS_ROOT}/.venv-mlx/bin/activate"
fi

MODEL_MLX="mlx-community/Qwen3.5-9B-MLX-4bit"
PORT="${PORT:-2025}"
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
    # Explicit KV budget: 4 GiB, sized for a single 64k request
    KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-4294967296}"
    GPU_MEM_UTILIZATION="${GPU_MEM_UTILIZATION:-0.18}"
    CONTEXT="${CONTEXT:-65536}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
    MODEL_HF="Intel/Qwen3.5-9B-int4-AutoRound"

    echo "Starting vLLM server on DGX Spark (port $PORT, KV cache bytes: ${KV_CACHE_MEMORY_BYTES}, gpu cap: ${GPU_MEM_UTILIZATION}, ctx: ${CONTEXT})..."
    docker run --rm --network host --gpus all --ipc host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        vllm-node:latest \
        vllm serve "$MODEL_HF" \
        --host "$HOST" \
        --port "$PORT" \
        --max-model-len "$CONTEXT" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEM_UTILIZATION" \
        --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
        --load-format fastsafetensors \
        --reasoning-parser qwen3 \
        --kv-cache-dtype fp8 \
        --enable-force-include-usage \
        "$@"
else
    QUANT="${QUANT:-UD-Q4_K_XL}"
    CACHE_TYPE="${CACHE_TYPE:-q4_0}"
    CONTEXT="${CONTEXT:-65536}"
    PARALLEL="${PARALLEL:-1}"
    echo "Starting llama-server on port $PORT..."
    llama-server -hf "unsloth/Qwen3.5-9B-GGUF:${QUANT}" --alias qwen-3.5-9b --host "$HOST" --port "$PORT" \
        --jinja -c "$CONTEXT" -ngl 99 --threads -1 --parallel "$PARALLEL" \
        --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40 \
        --no-mmap --flash-attn on \
        --cache-type-k "$CACHE_TYPE" --cache-type-v "$CACHE_TYPE" \
        "$@"
fi
