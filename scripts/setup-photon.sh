#!/bin/bash
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
source "${SCRIPTS_DIR}/setup-common.sh"
require_command uv
require_command ffmpeg

# Keep Photon's torch and native kernels separate from the shared model engines.
VENV_PATH="${ROOT}/parakeet-redux/.venv"
uv venv --python 3.12 --allow-existing "$VENV_PATH"
if [[ "$(uname -s)" == "Linux" ]]; then
    uv pip install --python "$VENV_PATH/bin/python" \
        --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
fi
uv pip install --python "$VENV_PATH/bin/python" \
    'moondream==2.4.0' 'fastapi>=0.115,<1' 'uvicorn>=0.30,<1' 'python-multipart>=0.0.20,<1'

"$VENV_PATH/bin/python" - <<'PY'
import moondream
assert "moondream/parakeet-redux" in moondream.photon_models()
print("Photon ready; moondream", moondream.__version__)
PY
echo "Start on demand: ./run.sh parakeet-redux"
