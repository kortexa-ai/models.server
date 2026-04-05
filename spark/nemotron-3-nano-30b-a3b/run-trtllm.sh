#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Nemotron-3-Nano 30B-A3B on TRT-LLM (sourcebuild Docker)
# Architecture: NemotronHForCausalLM (Mamba-2 hybrid + MoE, 30B total / 3.5B active)

IMAGE="${IMAGE:-local/trtllm-main:main-transformers--5.3.0-sourcebuild-120-real}"
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
PORT="${PORT:-2034}"
HOST="${HOST:-0.0.0.0}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"
REASONING_PARSER="${REASONING_PARSER:-nano-v3}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
LOG_LEVEL="${LOG_LEVEL:-info}"

HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
TRTLLM_CACHE_DIR="${TRTLLM_CACHE_DIR:-$HOME/.cache/tensorrt_llm}"
SCRIPT_DIR="$(pwd)"
CONFIG_FILE="${SCRIPT_DIR}/trtllm-config.yaml"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker is not installed." >&2
    exit 1
fi

mkdir -p "${HF_CACHE_DIR}" "${TRTLLM_CACHE_DIR}"

CONTAINER_NAME="trtllm-nemotron-3-nano"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Starting Nemotron-3-Nano 30B-A3B (NVFP4) via TRT-LLM"
echo "Image: ${IMAGE}"
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "Max batch size: ${MAX_BATCH_SIZE}"
echo "Max seq len: ${MAX_SEQ_LEN}"
echo "Reasoning parser: ${REASONING_PARSER}"

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
    -v "${TRTLLM_CACHE_DIR}:/root/.cache/tensorrt_llm" \
    -v "${SCRIPT_DIR}:/workspace/model-config:ro" \
    -e HF_TOKEN \
    -e HUGGING_FACE_HUB_TOKEN \
    "${IMAGE}" \
    python3 -m tensorrt_llm.commands.serve serve "${MODEL}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --backend pytorch \
        --tp_size 1 \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --max_batch_size "${MAX_BATCH_SIZE}" \
        --trust_remote_code \
        --reasoning_parser "${REASONING_PARSER}" \
        --tool_parser "${TOOL_PARSER}" \
        --extra_llm_api_options /workspace/model-config/trtllm-config.yaml \
        --log_level "${LOG_LEVEL}" \
        "$@"
