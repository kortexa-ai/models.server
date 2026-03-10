#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESOLVER="${SCRIPT_DIR}/resolve_model.py"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage:
  ./run-mlx.sh --list
  ./run-mlx.sh <model-key-or-hf-repo> [mlx_vlm.server args...]

Examples:
  ./run-mlx.sh 0.8b
  PORT=2031 ./run-mlx.sh 9b
  ./run-mlx.sh mlx-community/Qwen3.5-9B-MLX-4bit
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

VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/.venv-mlx}"
PYTHON_BIN="${PYTHON_BIN:-$(venv_python "${VENV_PATH}")}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-${MLX_PORT}}"

# Use the mlx model path from the catalog, fall back to the raw input if it looks like a HF repo
if [[ -n "${MLX_MODEL_PATH}" ]]; then
    MODEL_PATH="${MODEL_PATH:-${MLX_MODEL_PATH}}"
else
    MODEL_PATH="${MODEL_PATH:-${MODEL_INPUT}}"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: MLX environment is not installed in ${VENV_PATH}." >&2
    echo "Run bench/setup.sh first." >&2
    exit 1
fi

echo "Starting ${DISPLAY_NAME} via mlx_vlm.server on ${HOST}:${PORT}"
echo "Model: ${MODEL_PATH}"

exec "${PYTHON_BIN}" -m mlx_vlm.server \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "$@"
