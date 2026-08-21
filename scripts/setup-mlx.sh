#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=./setup-common.sh
source "${SCRIPT_DIR}/setup-common.sh"

require_command uv

VENV_PATH="${VENV_PATH:-${ROOT}/.venv-mlx}"

echo "Creating MLX venv at ${VENV_PATH}..."
uv venv --allow-existing "${VENV_PATH}"

echo "Installing mlx-vlm, mlx-lm, mlx-audio, Transformers, torch, torchvision..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    'mlx-vlm>=0.6.14' \
    mlx-lm \
    'mlx-audio[server]>=0.4.7' \
    'transformers>=5.14.1' \
    soundfile \
    torch \
    torchvision

echo "Verifying MLX environment..."
"${VENV_PATH}/bin/python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import platform

for package in ("mlx-vlm", "mlx-lm", "mlx-audio", "transformers", "soundfile", "torch", "torchvision"):
    try:
        print(f"{package} = {version(package)}")
    except PackageNotFoundError:
        print(f"{package} = NOT INSTALLED")

print(f"platform = {platform.platform()}")
print(f"machine = {platform.machine()}")
PY

echo ""
echo "MLX environment ready at ${VENV_PATH}"
