#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=./setup-common.sh
source "${SCRIPT_DIR}/setup-common.sh"

require_command uv

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
VENV_PATH="${VENV_PATH:-${ROOT}/.venv-vllm-omni}"
VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
VLLM_OMNI_VERSION="${VLLM_OMNI_VERSION:-0.26.0}"

if [[ -x "${VENV_PATH}/bin/python" ]]; then
    echo "Updating existing vLLM-Omni venv at ${VENV_PATH}"
else
    echo "Creating vLLM-Omni venv at ${VENV_PATH} with ${PYTHON_BIN}"
    uv venv --python "${PYTHON_BIN}" "${VENV_PATH}"
fi

echo "Installing vLLM ${VLLM_VERSION} for CUDA 13..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    "vllm==${VLLM_VERSION}" \
    --torch-backend cu130

echo "Installing vLLM-Omni ${VLLM_OMNI_VERSION}..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    "vllm-omni==${VLLM_OMNI_VERSION}" \
    ninja

configure_cuda_env "${VENV_PATH}/bin/python"

echo "Verifying vLLM-Omni environment..."
"${VENV_PATH}/bin/python" - <<'PY'
from importlib.metadata import version
import torch

print("python =", __import__("sys").version.split()[0])
print("torch =", torch.__version__)
print("torch.cuda =", torch.version.cuda)
print("cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device =", torch.cuda.get_device_name(0))
    print("capability =", torch.cuda.get_device_capability(0))
print("vllm =", version("vllm"))
print("vllm-omni =", version("vllm-omni"))
PY

echo ""
echo "vLLM-Omni environment ready at ${VENV_PATH}"
