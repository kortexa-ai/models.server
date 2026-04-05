#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Nemotron-3-Nano 30B-A3B on DGX Spark via Docker vLLM with Marlin NVFP4
# Architecture: Mamba-2 hybrid + MoE, 30B total / 3.5B active

MODEL_NVFP4="${MODEL_NVFP4:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
PORT="${PORT:-2034}"
HOST="${HOST:-0.0.0.0}"
IMAGE="${IMAGE:-vllm-node}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-17179869184}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

CONTAINER_NAME="vllm-nemotron-3-nano"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Starting Nemotron-3-Nano 30B-A3B (NVFP4) via Docker vLLM on port $PORT..."
exec docker run \
    --rm \
    --name "${CONTAINER_NAME}" \
    --privileged \
    --gpus all \
    --network host \
    --ipc host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -v "${SCRIPT_DIR}/nano_v3_reasoning_parser.py:/workspace/vllm/nano_v3_reasoning_parser.py:ro" \
    -e VLLM_NVFP4_GEMM_BACKEND=marlin \
    -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -e HF_TOKEN \
    -w /workspace/vllm \
    "${IMAGE}" \
    vllm serve "${MODEL_NVFP4}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --served-model-name nemotron-3-nano-30b-a3b \
        --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser-plugin nano_v3_reasoning_parser.py \
        --reasoning-parser nano_v3 \
        --kv-cache-dtype "${KV_CACHE_DTYPE}" \
        --enable-prefix-caching \
        "$@"
