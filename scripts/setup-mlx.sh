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

echo "Installing mlx-vlm, mlx-lm, torch, torchvision..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    'mlx-vlm>=0.6.0' \
    mlx-lm \
    torch \
    torchvision

echo "Patching mlx-vlm for Gemma 4 MTP speculative decoding (mlx-vlm#1260)..."
# gemma4's rollback_speculative_cache only guards for int, but mtp.py passes a
# plain Python list -> AttributeError: 'list' object has no attribute 'max'.
# Add the list/tuple branch (mirrors qwen3_5). Idempotent.
"${VENV_PATH}/bin/python" - <<'PY'
from importlib.util import find_spec
from pathlib import Path

path = Path(find_spec("mlx_vlm").origin).parent / "models" / "gemma4" / "language.py"
src = path.read_text()
if "elif isinstance(accepted, (list, tuple)):" in src:
    print("  already patched")
else:
    anchor = "        if isinstance(accepted, int):\n            accepted = mx.array([accepted])\n"
    if anchor not in src:
        raise SystemExit("ERROR: rollback_speculative_cache anchor not found; mlx-vlm layout changed?")
    patch = anchor + (
        "        elif isinstance(accepted, (list, tuple)):\n"
        "            # mtp.py passes accepted as a plain Python list; convert so the\n"
        "            # mx-array ops below (.max(), .size, +1, .shape) work. Mirrors the\n"
        "            # qwen3_5 rollback hook. Works around mlx-vlm#1260.\n"
        "            accepted = mx.array(accepted)\n"
    )
    path.write_text(src.replace(anchor, patch, 1))
    print(f"  patched {path}")
PY

echo "Verifying MLX environment..."
"${VENV_PATH}/bin/python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import platform

for package in ("mlx-vlm", "mlx-lm", "torch", "torchvision"):
    try:
        print(f"{package} = {version(package)}")
    except PackageNotFoundError:
        print(f"{package} = NOT INSTALLED")

print(f"platform = {platform.platform()}")
print(f"machine = {platform.machine()}")
PY

echo ""
echo "MLX environment ready at ${VENV_PATH}"
