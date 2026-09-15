#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=./setup-common.sh
source "${SCRIPT_DIR}/setup-common.sh"

require_command git
require_command uv

PYTHON_BIN="${PYTHON_BIN:-3.13}"
VENV_PATH="${VENV_PATH:-${ROOT}/.venv-omlx}"
ENGINES_ROOT="${ENGINES_ROOT:-${ROOT}/.engines}"
OMLX_ROOT="${OMLX_ROOT:-${ENGINES_ROOT}/omlx-k2-horizon}"
OMLX_REPO="${OMLX_REPO:-https://github.com/hermitdave/omlx.git}"
OMLX_COMMIT="${OMLX_COMMIT:-9b6126e9108887f2be0a3295f06286167f06350a}"

if [[ ! -d "${OMLX_ROOT}/.git" ]]; then
    git clone "${OMLX_REPO}" "${OMLX_ROOT}"
fi
git -C "${OMLX_ROOT}" fetch origin
git -C "${OMLX_ROOT}" checkout --detach "${OMLX_COMMIT}"

if [[ -x "${VENV_PATH}/bin/python" ]]; then
    echo "Updating existing oMLX venv at ${VENV_PATH}"
else
    echo "Creating oMLX venv at ${VENV_PATH} with Python ${PYTHON_BIN}"
    uv venv --python "${PYTHON_BIN}" "${VENV_PATH}"
fi

echo "Installing oMLX with K2 Horizon support..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    --editable "${OMLX_ROOT}[grammar]"

echo "Verifying oMLX environment..."
PYTHONPATH="${OMLX_ROOT}" "${VENV_PATH}/bin/python" - <<'PY'
from importlib.metadata import version
from omlx.patches.k2_horizon import apply_k2_horizon_patch

print(f"omlx = {version('omlx')}")
print(f"mlx = {version('mlx')}")
print(f"mlx-lm = {version('mlx-lm')}")
if not apply_k2_horizon_patch():
    raise SystemExit("K2 Horizon patch did not register")
print("k2_horizon patch = registered")
PY

echo ""
echo "Patched oMLX environment ready at ${VENV_PATH}"
