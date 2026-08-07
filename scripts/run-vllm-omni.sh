#!/bin/bash
set -euo pipefail

MODEL_DIR="$1"; shift
SCRIPTS_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
# shellcheck source=./setup-common.sh
source "${SCRIPTS_DIR}/setup-common.sh"

eval "$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"

if [[ "${VLLM_OMNI_SUPPORTED:-}" == "false" ]]; then
    echo "Not supported: ${MODEL_NAME} has no vLLM-Omni backend." >&2
    exit 1
fi

VENV_PATH="${VENV_PATH:-${ROOT}/.venv-vllm-omni}"
PYTHON_BIN="${PYTHON_BIN:-$(venv_python "${VENV_PATH}")}"
VLLM_BIN="${VLLM_BIN:-$(venv_bin "${VENV_PATH}")/vllm}"
if [[ ! -x "${PYTHON_BIN}" || ! -x "${VLLM_BIN}" ]]; then
    echo "Error: vLLM-Omni is not installed in ${VENV_PATH}." >&2
    echo "Run ./setup.sh or scripts/setup-vllm-omni.sh first." >&2
    exit 1
fi

export PATH="$(venv_bin "${VENV_PATH}"):${PATH}"
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
configure_cuda_env "${PYTHON_BIN}"

PORT="${PORT:-$MODEL_PORT}"
HOST="${HOST:-0.0.0.0}"

CMD=(
    "${VLLM_BIN}" serve "${VLLM_OMNI_MODEL}"
    --omni
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${MODEL_ID}"
)

if [[ -n "${VLLM_OMNI_DEPLOY_CONFIG:-}" ]]; then
    DEPLOY_CONFIG="$(
        VLLM_OMNI_CONFIG_NAME="${VLLM_OMNI_DEPLOY_CONFIG}" \
        VLLM_OMNI_MODEL_DIR="${MODEL_DIR}" \
        "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import vllm_omni

name = os.environ["VLLM_OMNI_CONFIG_NAME"]
path = Path(name)
if not path.is_absolute():
    model_path = Path(os.environ["VLLM_OMNI_MODEL_DIR"]) / path
    package_path = Path(vllm_omni.__file__).resolve().parent / "deploy" / path
    path = model_path if model_path.is_file() else package_path
if not path.is_file():
    raise SystemExit(f"vLLM-Omni deploy config not found: {path}")
print(path)
PY
    )"
    CMD+=(--deploy-config "${DEPLOY_CONFIG}")
fi
if [[ "${VLLM_OMNI_TRUST_REMOTE_CODE:-}" == "true" ]]; then
    CMD+=(--trust-remote-code)
fi

echo "Starting ${MODEL_NAME} via vLLM-Omni on port ${PORT}..."
exec "${CMD[@]}" "$@"
