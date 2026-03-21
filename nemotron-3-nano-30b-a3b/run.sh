#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Nemotron-3-Nano 30B-A3B via eugr's spark-vllm-docker (Marlin NVFP4 backend)
# Architecture: NemotronHForCausalLM (Mamba-2 hybrid + MoE, 30B total / 3.5B active)
# ~45 tok/s sustained on single DGX Spark

IMAGE="${IMAGE:-vllm-node}"
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
PORT="${PORT:-2034}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
SCRIPT_DIR="$(pwd)"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker is not installed." >&2
    exit 1
fi

CONTAINER_NAME="vllm-nemotron-3-nano"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Starting Nemotron-3-Nano 30B-A3B (NVFP4) via eugr/spark-vllm-docker"
echo "Image: ${IMAGE}"
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Max model len: ${MAX_MODEL_LEN}"
echo "KV cache dtype: ${KV_CACHE_DTYPE}"

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
    vllm serve "${MODEL}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --served-model-name nemotron-3-nano-30b-a3b \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser-plugin nano_v3_reasoning_parser.py \
        --reasoning-parser nano_v3 \
        --kv-cache-dtype "${KV_CACHE_DTYPE}" \
        --enable-prefix-caching \
        "$@"
