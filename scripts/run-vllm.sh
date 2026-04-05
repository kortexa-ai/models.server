#!/bin/bash
set -euo pipefail

MODEL_DIR="$(cd "$1" && pwd)"; shift
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

# shellcheck source=./setup-common.sh
source "${SCRIPTS_DIR}/setup-common.sh"

eval "$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"

if [[ "${VLLM_SUPPORTED:-}" == "false" ]]; then
    echo "Not supported: ${MODEL_NAME} is not configured for vLLM." >&2
    exit 1
fi

VENV_PATH="${VENV_PATH:-${ROOT}/.venv-vllm}"
PYTHON_BIN="${PYTHON_BIN:-$(venv_python "${VENV_PATH}")}"
VLLM_BIN="${VLLM_BIN:-$(venv_bin "${VENV_PATH}")/vllm}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${VLLM_BIN}" ]]; then
    echo "Error: vLLM is not installed in ${VENV_PATH}." >&2
    echo "Run ./setup.sh or scripts/setup-vllm.sh first." >&2
    exit 1
fi

export PATH="$(venv_bin "${VENV_PATH}"):${PATH}"

# Pass through HF token if set, or copy from default location
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

# Marlin NVFP4 environment
if [[ "${VLLM_MARLIN:-}" == "true" ]]; then
    export VLLM_NVFP4_GEMM_BACKEND=marlin
    export VLLM_TEST_FORCE_FP8_MARLIN=1
    export VLLM_MARLIN_USE_ATOMIC_ADD=1
fi

configure_cuda_env "${PYTHON_BIN}"

PORT="${PORT:-$MODEL_PORT}"
HOST="${HOST:-0.0.0.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS:-$MODEL_PARALLEL}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$MODEL_CONTEXT}"

echo "Starting ${MODEL_NAME} via bare-metal vLLM on port ${PORT}..."

CMD=(
    "${VLLM_BIN}" serve "${VLLM_MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${MODEL_ID}"
    --tensor-parallel-size 1
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --kv-cache-dtype "${VLLM_KV_CACHE_DTYPE}"
    --attention-backend "${VLLM_ATTENTION_BACKEND}"
)

# Optional flags from model config
if [[ -n "${VLLM_KV_CACHE_BYTES:-}" ]]; then
    CMD+=(--kv-cache-memory-bytes "${VLLM_KV_CACHE_BYTES}")
fi
if [[ "${VLLM_QUANTIZATION}" != "auto" ]]; then
    CMD+=(--quantization "${VLLM_QUANTIZATION}")
fi
if [[ "${VLLM_TRUST_REMOTE_CODE:-}" == "true" ]]; then
    CMD+=(--trust-remote-code)
fi
if [[ -n "${VLLM_TOOL_CALL_PARSER:-}" ]]; then
    CMD+=(--enable-auto-tool-choice --tool-call-parser "${VLLM_TOOL_CALL_PARSER}")
fi
if [[ -n "${VLLM_REASONING_PARSER:-}" ]]; then
    CMD+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
fi
if [[ -n "${VLLM_REASONING_PARSER_PLUGIN:-}" ]]; then
    CMD+=(--reasoning-parser-plugin "${MODEL_DIR}/${VLLM_REASONING_PARSER_PLUGIN}")
fi
CMD+=(--enable-prefix-caching)

exec "${CMD[@]}" "$@"
