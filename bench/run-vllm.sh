#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESOLVER="${SCRIPT_DIR}/resolve_model.py"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage:
  ./run-vllm.sh --list
  ./run-vllm.sh <model-key-or-hf-repo> [vllm args...]

Examples:
  ./run-vllm.sh 0.8b
  PORT=2241 ./run-vllm.sh 9b
  GPU_MEMORY_UTILIZATION=0.25 ./run-vllm.sh Qwen/Qwen3.5-0.8B
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

VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/.venv-vllm}"
PYTHON_BIN="${PYTHON_BIN:-$(venv_python "${VENV_PATH}")}"
VLLM_BIN="${VLLM_BIN:-$(venv_bin "${VENV_PATH}")/vllm}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-${VLLM_PORT}}"
TP="${TP:-${TP_DEFAULT}}"
MODEL_PATH="${MODEL_PATH:-${VLLM_MODEL_PATH}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-${CONTEXT_LENGTH_DEFAULT}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.25}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
REASONING_PARSER="${REASONING_PARSER:-${REASONING_PARSER_DEFAULT}}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${SCRIPT_DIR}/.cache/vllm}"
VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${SCRIPT_DIR}/.config/vllm}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${VLLM_CACHE_ROOT}/torch_compile_cache}"
HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${VLLM_BIN}" ]]; then
    echo "Error: vLLM is not installed in ${VENV_PATH}." >&2
    echo "Run bench/setup.sh first." >&2
    exit 1
fi

export PATH="$(venv_bin "${VENV_PATH}"):${PATH}"

mkdir -p \
    "${VLLM_CACHE_ROOT}" \
    "${VLLM_CONFIG_ROOT}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}"
export VLLM_CACHE_ROOT
export VLLM_CONFIG_ROOT
export TORCHINDUCTOR_CACHE_DIR
export HF_HOME
export HF_HUB_CACHE
export TRANSFORMERS_CACHE

configure_cuda_env "${PYTHON_BIN}"

CMD=(
    "${VLLM_BIN}" serve "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tensor-parallel-size "${TP}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${CONTEXT_LENGTH}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --reasoning-parser "${REASONING_PARSER}"
)

echo "Starting ${DISPLAY_NAME} in bare-metal vLLM with ${MODEL_PATH} on ${HOST}:${PORT}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Context length: ${CONTEXT_LENGTH}"
echo "vLLM cache root: ${VLLM_CACHE_ROOT}"
echo "HF cache root: ${HF_HOME}"
if [[ -n "${TRITON_PTXAS_PATH:-}" ]]; then
    echo "Triton ptxas: ${TRITON_PTXAS_PATH}"
fi

exec "${CMD[@]}" "$@"
