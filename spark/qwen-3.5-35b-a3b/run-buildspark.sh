#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Qwen 3.5 35B-A3B via buildspark/vllm-gb10-mtp Docker
# BF16 weights + FP8 KV cache + MTP speculative decoding
# ~20 tok/s sustained on single DGX Spark
# Requires --privileged for Triton SM121 kernel loading
# First boot: ~10 min (torch.compile + FlashInfer autotuning), cached after that
#
# NOTE: FP8 weight quantization is broken on SM121 — BF16 weights only.
# NOTE: Needs ~100 GiB free GPU memory (stop other models first).

IMAGE="${IMAGE:-buildspark/vllm-gb10-mtp}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
PORT="${PORT:-2027}"
HOST="${HOST:-0.0.0.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-12}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker is not installed." >&2
    exit 1
fi

CONTAINER_NAME="vllm-qwen35-35b-a3b"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Starting Qwen 3.5 35B-A3B (BF16+MTP) via buildspark/vllm-gb10-mtp"
echo "Image: ${IMAGE}"
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"

exec docker run \
    --rm \
    --name "${CONTAINER_NAME}" \
    --privileged \
    --gpus all \
    --ipc host \
    --network host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -e VLLM_USE_FLASHINFER_MOE_FP8=0 \
    -e VLLM_FLASHINFER_MOE_BACKEND=latency \
    -e HF_HUB_DISABLE_XET=1 \
    -e HF_TOKEN \
    "${IMAGE}" \
    vllm serve "${MODEL}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --served-model-name qwen-3.5-35b-a3b \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --kv-cache-dtype fp8_e4m3 \
        --enable-prefix-caching \
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        "$@"
