#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=./setup-common.sh
source "${SCRIPT_DIR}/setup-common.sh"

require_command uv

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
VENV_PATH="${VENV_PATH:-${ROOT}/.venv-vllm}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"
VLLM_WHEEL_INDEX="${VLLM_WHEEL_INDEX:-}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=5.14.1}"
NUMPY_SPEC="${NUMPY_SPEC:-numpy<2.4}"

if [[ -x "${VENV_PATH}/bin/python" ]]; then
    echo "Updating existing vLLM venv at ${VENV_PATH}"
else
    echo "Creating vLLM venv at ${VENV_PATH} with ${PYTHON_BIN}"
    uv venv --python "${PYTHON_BIN}" "${VENV_PATH}"
fi

echo "Installing latest stable vLLM for CUDA 13..."
VLLM_INSTALL_ARGS=(
    --python "${VENV_PATH}/bin/python"
    --upgrade
    "${VLLM_SPEC}"
    --torch-backend cu130
)
if [[ -n "${VLLM_WHEEL_INDEX}" ]]; then
    VLLM_INSTALL_ARGS+=(--extra-index-url "${VLLM_WHEEL_INDEX}" --prerelease allow)
fi
uv pip install "${VLLM_INSTALL_ARGS[@]}"

# Old nightlies installed a separately versioned cubin wheel.  Current
# FlashInfer downloads matching cubins on demand; a stale wheel prevents import.
uv pip uninstall --python "${VENV_PATH}/bin/python" flashinfer-cubin 2>/dev/null || true

echo "Upgrading Transformers (>=5.14.1 required for the current model stack)..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    "${TRANSFORMERS_SPEC}" "${NUMPY_SPEC}"

echo "Installing runtime build helpers used by FP4 kernels..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade ninja

configure_cuda_env "${VENV_PATH}/bin/python"

echo "Verifying vLLM environment..."
"${VENV_PATH}/bin/python" - <<'PY'
from importlib.metadata import version
import torch

print("python =", __import__("sys").version.split()[0])
print("torch =", torch.__version__)
print("torch.cuda =", torch.version.cuda)
print("cudnn =", torch.backends.cudnn.version())
print("cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device =", torch.cuda.get_device_name(0))
    print("capability =", torch.cuda.get_device_capability(0))
print("vllm =", version("vllm"))
print("transformers =", version("transformers"))
print("triton =", version("triton"))
print("flashinfer-python =", version("flashinfer-python"))
PY
