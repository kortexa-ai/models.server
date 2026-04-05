#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESOLVER="${SCRIPT_DIR}/resolve_model.py"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage:
  ./run-sglang.sh --list
  ./run-sglang.sh [--profile standard|nvfp4] <model-key-or-hf-repo> [sglang args...]

Examples:
  ./run-sglang.sh 0.8b
  ./run-sglang.sh --profile standard 9b
  ATTENTION_BACKEND=trtllm_mha ./run-sglang.sh 0.8b
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--list" ]]; then
    exec python3 "${RESOLVER}" --list
fi

PROFILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="${2:-}"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

MODEL_INPUT="$1"
shift

RESOLVER_ARGS=("${MODEL_INPUT}" --format shell)
if [[ -n "${PROFILE}" ]]; then
    RESOLVER_ARGS+=(--profile "${PROFILE}")
fi
eval "$(python3 "${RESOLVER}" "${RESOLVER_ARGS[@]}")"

VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/.venv-sglang}"
PYTHON_BIN="${PYTHON_BIN:-$(venv_python "${VENV_PATH}")}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-${SGLANG_PORT}}"
TP="${TP:-${TP_DEFAULT}}"
MODEL_PATH="${MODEL_PATH:-${SGLANG_MODEL_PATH}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-${CONTEXT_LENGTH_DEFAULT}}"
QUANTIZATION="${QUANTIZATION:-${QUANTIZATION_DEFAULT}}"
REASONING_PARSER="${REASONING_PARSER:-${REASONING_PARSER_DEFAULT}}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"
ENABLE_CACHE_REPORT="${ENABLE_CACHE_REPORT:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: SGLang is not installed in ${VENV_PATH}." >&2
    echo "Run bench/setup.sh first." >&2
    exit 1
fi

configure_cuda_env "${PYTHON_BIN}"

CMD=(
    "${PYTHON_BIN}" -m sglang.launch_server
    --model-path "${MODEL_PATH}"
    --tp "${TP}"
    --reasoning-parser "${REASONING_PARSER}"
    --attention-backend "${ATTENTION_BACKEND}"
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
)

if [[ -n "${QUANTIZATION}" ]]; then
    CMD+=(--quantization "${QUANTIZATION}")
fi

if [[ "${ENABLE_CACHE_REPORT}" == "1" ]]; then
    CMD+=(--enable-cache-report)
fi

if [[ -n "${CONTEXT_LENGTH}" ]]; then
    CMD+=(--context-length "${CONTEXT_LENGTH}")
fi

if [[ -n "${MEM_FRACTION_STATIC}" ]]; then
    CMD+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
fi

echo "Starting ${DISPLAY_NAME} (${SGLANG_PROFILE}) in bare-metal SGLang with ${MODEL_PATH} on ${HOST}:${PORT}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Attention backend: ${ATTENTION_BACKEND}"
echo "Context length: ${CONTEXT_LENGTH}"
if [[ -n "${TRITON_PTXAS_PATH:-}" ]]; then
    echo "Triton ptxas: ${TRITON_PTXAS_PATH}"
fi

exec "${CMD[@]}" "$@"
