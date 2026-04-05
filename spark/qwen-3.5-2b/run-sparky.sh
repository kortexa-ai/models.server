#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# DGX Spark - use vLLM Docker with AutoRound int4
# Explicit KV budget: 0.75 GiB, sized for a single 32k request
# KV math: 2(K+V) * 24 layers * 2 kv_heads * 256 head_dim * 1 byte(fp8) * 32768 ctx * 1 req = 0.75 GiB
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-805306368}"
CONTEXT="${CONTEXT:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MODEL_HF="Intel/Qwen3.5-2B-int4-AutoRound"
PORT="${PORT:-2030}"
HOST="${HOST:-0.0.0.0}"

CONTAINER_NAME="qwen-3.5-2b"
trap 'docker stop "$CONTAINER_NAME" 2>/dev/null' EXIT
echo "Starting vLLM server on DGX Spark (port $PORT, KV cache bytes: ${KV_CACHE_MEMORY_BYTES}, ctx: ${CONTEXT})..."
if docker container inspect "$CONTAINER_NAME" &>/dev/null; then
    echo "Reusing existing container $CONTAINER_NAME..."
    docker start -a "$CONTAINER_NAME"
else
    exec docker run --name "$CONTAINER_NAME" --network host --gpus all --ipc host --privileged \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v ~/.cache/vllm-docker:/root/.cache/vllm \
        vllm-node:latest \
        vllm serve "$MODEL_HF" \
        --served-model-name qwen-3.5-2b \
        --host "$HOST" \
        --port "$PORT" \
        --max-model-len "$CONTEXT" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
        --load-format fastsafetensors \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --kv-cache-dtype fp8 \
        --enable-force-include-usage \
        "$@"
fi
