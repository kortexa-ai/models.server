#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESOLVER="${SCRIPT_DIR}/resolve_model.py"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage:
  ./run-llama.sh --list
  ./run-llama.sh [--quant Q4_K_M] <model-key-or-hf-repo> [llama-server args...]

Examples:
  ./run-llama.sh 0.8b
  ./run-llama.sh --quant Q8_0 2b
  PORT=2026 ./run-llama.sh 27b
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--list" ]]; then
    exec python3 "${RESOLVER}" --list
fi

require_command llama-server

QUANT="Q4_K_M"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quant)
            QUANT="${2:-Q4_K_M}"
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

eval "$(python3 "${RESOLVER}" "${MODEL_INPUT}" --format shell)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-${LLAMA_PORT}}"
CONTEXT="${CONTEXT:-${CONTEXT_LENGTH_DEFAULT}}"
CACHE_TYPE="${CACHE_TYPE:-q4_0}"
PARALLEL="${PARALLEL:-1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_ID}}"

# Build the HF repo spec with quantization tag
HF_SPEC="${LLAMA_GGUF_REPO}:${QUANT}"

CMD=(
    llama-server
    -hf "${HF_SPEC}"
    --alias "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --jinja
    -c "${CONTEXT}"
    -ngl 99
    --threads -1
    --parallel "${PARALLEL}"
    --temp 1.0
    --top-p 0.95
    --min-p 0.01
    --top-k 40
    --no-mmap
    --flash-attn on
    --cache-type-k "${CACHE_TYPE}"
    --cache-type-v "${CACHE_TYPE}"
)

echo "Starting ${DISPLAY_NAME} via llama-server on ${HOST}:${PORT}"
echo "Model: ${HF_SPEC}"
echo "Quantization: ${QUANT}"
echo "Context length: ${CONTEXT}"
echo "Cache type: ${CACHE_TYPE}"

exec "${CMD[@]}" "$@"
