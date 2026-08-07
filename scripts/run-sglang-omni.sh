#!/bin/bash
set -euo pipefail

MODEL_DIR="$1"; shift
SCRIPTS_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
# shellcheck source=./setup-common.sh
source "${SCRIPTS_DIR}/setup-common.sh"

eval "$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"

if [[ "${SGLANG_OMNI_SUPPORTED:-}" == "false" ]]; then
    echo "Not supported: ${MODEL_NAME} has no SGLang-Omni backend." >&2
    exit 1
fi

VENV_PATH="${VENV_PATH:-${ROOT}/.venv-sglang-omni}"
PYTHON_BIN="${PYTHON_BIN:-$(venv_python "${VENV_PATH}")}"
SGLANG_ROOT="${SGLANG_ROOT:-${ROOT}/.engines/sglang-omni}"
AUDIO8_ROOT="${AUDIO8_ROOT:-${ROOT}/.engines/audio8-tts}"
if [[ ! -x "${PYTHON_BIN}" || ! -d "${SGLANG_ROOT}/sglang_omni" ]]; then
    echo "Error: Audio8 SGLang-Omni environment is not installed." >&2
    echo "Run ./setup.sh or scripts/setup-sglang-omni.sh first." >&2
    exit 1
fi

CONFIG_PATH="${SGLANG_OMNI_CONFIG}"
if [[ "${CONFIG_PATH}" != /* ]]; then
    CONFIG_PATH="${AUDIO8_ROOT}/${CONFIG_PATH}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: SGLang-Omni config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

export PATH="$(venv_bin "${VENV_PATH}"):${PATH}"
export PYTHONPATH="${SGLANG_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
configure_cuda_env "${PYTHON_BIN}"

MODEL_PATH="$(
    SGLANG_MODEL_REPO="${SGLANG_OMNI_MODEL}" "${PYTHON_BIN}" - <<'PY'
import os

from huggingface_hub import snapshot_download

print(snapshot_download(os.environ["SGLANG_MODEL_REPO"]))
PY
)"

if [[ -n "${SGLANG_OMNI_ATTENTION_BACKEND:-}" ]]; then
    export AUDIO8_TTS_ATTENTION_BACKEND="${SGLANG_OMNI_ATTENTION_BACKEND}"
fi
if [[ -n "${SGLANG_OMNI_MEMORY_FRACTION_STATIC:-}" ]]; then
    export AUDIO8_TTS_MEM_FRACTION_STATIC="${SGLANG_OMNI_MEMORY_FRACTION_STATIC}"
fi
if [[ -n "${SGLANG_OMNI_MAX_RUNNING_REQUESTS:-}" ]]; then
    export AUDIO8_TTS_MAX_RUNNING_REQUESTS="${SGLANG_OMNI_MAX_RUNNING_REQUESTS}"
fi
if [[ "${SGLANG_OMNI_TORCH_COMPILE:-false}" == "true" ]]; then
    export AUDIO8_TTS_ENABLE_TORCH_COMPILE=1
fi
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/audio8-flashinfer}"

PORT="${PORT:-$MODEL_PORT}"
HOST="${HOST:-0.0.0.0}"

echo "Starting ${MODEL_NAME} via SGLang-Omni on port ${PORT}..."
exec "${PYTHON_BIN}" -m sglang_omni.cli.cli serve \
    --model-path "${MODEL_PATH}" \
    --config "${CONFIG_PATH}" \
    --model-name "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    "$@"
