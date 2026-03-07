#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

IMAGE_TAG="${IMAGE_TAG:-kortexa/qwen-3.5-27b-vllm:ngc-26.02}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen-3.5-27b-vllm}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.5-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-unsloth/Qwen3.5-27B-GGUF:Q4_K_M}"
PORT="${PORT:-2026}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$HOME/.cache/vllm}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-}"

if ! command -v docker &>/dev/null; then
    echo "Error: docker is not installed."
    exit 1
fi

mkdir -p "${HF_CACHE_DIR}"
mkdir -p "${VLLM_CACHE_DIR}"

if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "Error: docker image ${IMAGE_TAG} is missing."
    echo "Build it first with ./docker-build.sh"
    exit 1
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

DOCKER_ARGS=(
    --rm
    --name "${CONTAINER_NAME}"
    --gpus all
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -p "${PORT}:${CONTAINER_PORT}"
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
    -v "${VLLM_CACHE_DIR}:/root/.cache/vllm"
    -e HF_TOKEN
    -e HUGGING_FACE_HUB_TOKEN
)

if [[ -n "${VLLM_ATTENTION_BACKEND}" ]]; then
    DOCKER_ARGS+=(-e "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}")
fi

SERVER_ARGS=(
    serve
    "${MODEL_REPO}"
    --host "${HOST}"
    --port "${CONTAINER_PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --reasoning-parser "${REASONING_PARSER}"
)

if [[ "${LANGUAGE_MODEL_ONLY}" == "1" ]]; then
    SERVER_ARGS+=(--language-model-only)
fi

echo "Starting ${MODEL_REPO} on port ${PORT} with image ${IMAGE_TAG}"

exec docker run "${DOCKER_ARGS[@]}" "${IMAGE_TAG}" "${SERVER_ARGS[@]}" "$@"
