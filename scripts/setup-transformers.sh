#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=./setup-common.sh
source "${SCRIPT_DIR}/setup-common.sh"

require_command uv

PYTHON_BIN="${PYTHON_BIN:-3.12}"
VENV_PATH="${VENV_PATH:-${ROOT}/.venv}"

echo "Creating Transformers venv at ${VENV_PATH}..."
uv venv --python "${PYTHON_BIN}" --allow-existing "${VENV_PATH}"

echo "Installing Transformers and torch..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    'transformers>=5.12.0' \
    torch

echo "Verifying Transformers environment..."
"${VENV_PATH}/bin/python" - <<'PY'
from importlib.metadata import version
import torch

print("python =", __import__("sys").version.split()[0])
print("transformers =", version("transformers"))
print("torch =", torch.__version__)
print("cpu_threads =", torch.get_num_threads())
print("mps_available =", torch.backends.mps.is_available())
PY

echo ""
echo "Transformers environment ready at ${VENV_PATH}"
