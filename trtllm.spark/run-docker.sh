#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESOLVER="${SCRIPT_DIR}/../bare.spark/resolve_model.py"
CONFIG_FILE_HOST="${SCRIPT_DIR}/extra-llm-api-config.yml"
CONFIG_FILE_CONTAINER="/workspace/trtllm.spark/extra-llm-api-config.yml"

usage() {
    cat <<'EOF'
Usage:
  ./trtllm.spark/run-docker.sh --list
  ./trtllm.spark/run-docker.sh <model-key-or-hf-repo> [trtllm args...]

Examples:
  ./trtllm.spark/run-docker.sh 0.8b
  PORT=2251 ./trtllm.spark/run-docker.sh 4b
  REASONING_PARSER=qwen3 ./trtllm.spark/run-docker.sh Qwen/Qwen3.5-4B
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--list" ]]; then
    exec python3 "${RESOLVER}" --list
fi

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

MODEL_INPUT="$1"
shift

eval "$(python3 "${RESOLVER}" "${MODEL_INPUT}" --format shell)"

IMAGE="${IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6}"
CONTAINER_NAME="${CONTAINER_NAME:-trtllm-${MODEL_KEY}}"
MODEL_PATH="${MODEL_PATH:-${VLLM_MODEL_PATH}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-2250}"
BACKEND="${BACKEND:-pytorch}"
TP="${TP:-${TP_DEFAULT}}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-${CONTEXT_LENGTH_DEFAULT}}"
KV_CACHE_FREE_GPU_MEMORY_FRACTION="${KV_CACHE_FREE_GPU_MEMORY_FRACTION:-}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-}"
REASONING_PARSER="${REASONING_PARSER:-}"
LOG_LEVEL="${LOG_LEVEL:-info}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
TRTLLM_CACHE_DIR="${TRTLLM_CACHE_DIR:-$HOME/.cache/tensorrt_llm}"
USE_DEFAULT_CONFIG="${USE_DEFAULT_CONFIG:-0}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker is not installed." >&2
    exit 1
fi

mkdir -p "${HF_CACHE_DIR}" "${TRTLLM_CACHE_DIR}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

DOCKER_ARGS=(
    --rm
    --name "${CONTAINER_NAME}"
    --privileged
    --gpus all
    --network host
    --ipc host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
    -v "${TRTLLM_CACHE_DIR}:/root/.cache/tensorrt_llm"
    -e HF_TOKEN
    -e HUGGING_FACE_HUB_TOKEN
)

if [[ "${USE_DEFAULT_CONFIG}" == "1" && -f "${CONFIG_FILE_HOST}" ]]; then
    DOCKER_ARGS+=(-v "${SCRIPT_DIR}:/workspace/trtllm.spark:ro")
fi

SERVER_ARGS=(
    trtllm-serve serve "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --backend "${BACKEND}"
    --log_level "${LOG_LEVEL}"
    --tp_size "${TP}"
    --max_seq_len "${CONTEXT_LENGTH}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    SERVER_ARGS+=(--trust_remote_code)
fi

if [[ -n "${KV_CACHE_FREE_GPU_MEMORY_FRACTION}" ]]; then
    SERVER_ARGS+=(--kv_cache_free_gpu_memory_fraction "${KV_CACHE_FREE_GPU_MEMORY_FRACTION}")
fi

if [[ -n "${MAX_BATCH_SIZE}" ]]; then
    SERVER_ARGS+=(--max_batch_size "${MAX_BATCH_SIZE}")
fi

if [[ -n "${MAX_NUM_TOKENS}" ]]; then
    SERVER_ARGS+=(--max_num_tokens "${MAX_NUM_TOKENS}")
fi

if [[ -n "${REASONING_PARSER}" ]]; then
    SERVER_ARGS+=(--reasoning_parser "${REASONING_PARSER}")
fi

if [[ "${USE_DEFAULT_CONFIG}" == "1" && -f "${CONFIG_FILE_HOST}" ]]; then
    SERVER_ARGS+=(--extra_llm_api_options "${CONFIG_FILE_CONTAINER}")
fi

echo "Starting ${DISPLAY_NAME} in TensorRT-LLM Docker with ${MODEL_PATH}"
echo "Image: ${IMAGE}"
echo "Container: ${CONTAINER_NAME}"
echo "Port: ${PORT}"
echo "Backend: ${BACKEND}"
echo "Context length: ${CONTEXT_LENGTH}"
echo "Served model name target: ${SERVED_MODEL_NAME}"
if [[ -n "${KV_CACHE_FREE_GPU_MEMORY_FRACTION}" ]]; then
    echo "KV cache free GPU memory fraction: ${KV_CACHE_FREE_GPU_MEMORY_FRACTION}"
fi
if [[ -n "${REASONING_PARSER}" ]]; then
    echo "Reasoning parser: ${REASONING_PARSER}"
fi

exec docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${SERVER_ARGS[@]}" "$@"
