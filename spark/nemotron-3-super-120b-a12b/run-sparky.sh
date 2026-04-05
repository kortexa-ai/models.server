#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Nemotron-3-Super 120B-A12B on DGX Spark via TRT-LLM Docker
# Architecture: NemotronHForCausalLM (Mamba-2 hybrid + LatentMoE, 120B total / 12B active)

MODEL_NVFP4="${MODEL_NVFP4:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
PORT="${PORT:-2033}"
HOST="${HOST:-0.0.0.0}"
IMAGE="${IMAGE:-local/trtllm-main:main-transformers--5.3.0-sourcebuild-120-real}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-65536}"
REASONING_PARSER="${REASONING_PARSER:-nano-v3}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
LOG_LEVEL="${LOG_LEVEL:-info}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
TRTLLM_CACHE_DIR="${TRTLLM_CACHE_DIR:-$HOME/.cache/tensorrt_llm}"

mkdir -p "${HF_CACHE_DIR}" "${TRTLLM_CACHE_DIR}"

CONTAINER_NAME="trtllm-nemotron-3-super"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "Starting Nemotron-3-Super 120B-A12B (NVFP4) via TRT-LLM on port $PORT..."
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
    python3 -m tensorrt_llm.commands.serve serve "${MODEL_NVFP4}" \
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
