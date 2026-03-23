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
    # Explicit KV budget: 16 GiB, sized for two 64k requests
    # KV math: 2(K+V) * 64 layers * 4 kv_heads * 256 head_dim * 1 byte(fp8) * 65536 ctx * 2 reqs = 16 GiB
    KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-17179869184}"
    GPU_MEM_UTILIZATION="${GPU_MEM_UTILIZATION:-0.40}"
    CONTEXT="${CONTEXT:-65536}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
    # Fall back to standard 27B int4 - Opus distill not available in int4
    MODEL_HF="Intel/Qwen3.5-27B-int4-AutoRound"

    CONTAINER_NAME="qwen-3.5-27b-opus"
    trap 'docker stop "$CONTAINER_NAME" 2>/dev/null' EXIT
    echo "Starting vLLM server on DGX Spark (port $PORT, KV cache bytes: ${KV_CACHE_MEMORY_BYTES}, gpu cap: ${GPU_MEM_UTILIZATION}, ctx: ${CONTEXT})..."
    echo "NOTE: Using base 27B int4 (Opus distill not available in int4)"
    if docker container inspect "$CONTAINER_NAME" &>/dev/null; then
        echo "Reusing existing container $CONTAINER_NAME..."
        docker start -a "$CONTAINER_NAME"
    else
        docker run --name "$CONTAINER_NAME" --network host --gpus all --ipc host --privileged \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -e HF_TOKEN="${HF_TOKEN:-}" \
            -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
            -v ~/.cache/huggingface:/root/.cache/huggingface \
            -v ~/.cache/vllm-docker:/root/.cache/vllm \
            vllm-node:latest \
            vllm serve "$MODEL_HF" \
            --served-model-name qwen-3.5-27b-opus \
            --host "$HOST" \
            --port "$PORT" \
            --max-model-len "$CONTEXT" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --gpu-memory-utilization "$GPU_MEM_UTILIZATION" \
            --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
            --load-format fastsafetensors \
            --reasoning-parser qwen3 \
            --enable-auto-tool-choice \
            --tool-call-parser qwen3_xml \
            --kv-cache-dtype fp8 \
            --enable-force-include-usage \
            "$@"
    fi
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
