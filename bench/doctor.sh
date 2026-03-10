#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage:
  ./doctor.sh vllm
  ./doctor.sh sglang
  ./doctor.sh mlx
EOF
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 1
fi

case "$1" in
    vllm)
        VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/.venv-vllm}"
        ;;
    sglang)
        VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/.venv-sglang}"
        ;;
    mlx)
        VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/.venv-mlx}"
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac

PYTHON_BIN="$(venv_python "${VENV_PATH}")"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: ${PYTHON_BIN} is missing." >&2
    exit 1
fi

if [[ "$1" == "mlx" ]]; then
    echo "python: ${PYTHON_BIN}"
    echo
    "${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in ("torch", "mlx-lm", "mlx-vlm", "torchvision"):
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError:
        pass

import platform
print(f"platform={platform.platform()}")
print(f"machine={platform.machine()}")
PY
else
    configure_cuda_env "${PYTHON_BIN}"

    echo "python: ${PYTHON_BIN}"
    echo "cuda_home: ${CUDA_HOME}"
    echo "triton_ptxas: ${TRITON_PTXAS_PATH:-<unset>}"
    echo "torch_cuda_arch_list: ${TORCH_CUDA_ARCH_LIST}"
    echo

    if command -v ptxas >/dev/null 2>&1; then
        ptxas --version
    fi

    echo
    "${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import torch

for package in ("torch", "vllm", "sglang", "sgl-kernel", "transformers", "triton", "flashinfer-python"):
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError:
        pass

print("torch.version.cuda=", torch.version.cuda)
print("cudnn.version=", torch.backends.cudnn.version())
print("cuda.available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device=", torch.cuda.get_device_name(0))
    print("capability=", torch.cuda.get_device_capability(0))
PY
fi
