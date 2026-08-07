#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=./setup-common.sh
source "${SCRIPT_DIR}/setup-common.sh"

require_command git
require_command uv

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
VENV_PATH="${VENV_PATH:-${ROOT}/.venv-sglang-omni}"
ENGINES_ROOT="${ENGINES_ROOT:-${ROOT}/.engines}"
SGLANG_ROOT="${SGLANG_ROOT:-${ENGINES_ROOT}/sglang-omni}"
AUDIO8_ROOT="${AUDIO8_ROOT:-${ENGINES_ROOT}/audio8-tts}"
SGLANG_COMMIT="${SGLANG_COMMIT:-68a572348837f7b004857b4b07993c20ade4c017}"
AUDIO8_COMMIT="${AUDIO8_COMMIT:-7b6a5a15e4529da4e37463736de1d70753ceb2fb}"

checkout_repo() {
    local repo_url="$1"
    local checkout_path="$2"
    local commit="$3"

    if [[ ! -d "${checkout_path}/.git" ]]; then
        git clone "${repo_url}" "${checkout_path}"
    fi
    git -C "${checkout_path}" fetch origin
    git -C "${checkout_path}" checkout --detach "${commit}"
}

mkdir -p "${ENGINES_ROOT}"

echo "Checking out SGLang-Omni ${SGLANG_COMMIT}..."
checkout_repo \
    "https://github.com/sgl-project/sglang-omni.git" \
    "${SGLANG_ROOT}" \
    "${SGLANG_COMMIT}"

echo "Checking out Audio8 adapter ${AUDIO8_COMMIT}..."
checkout_repo \
    "https://github.com/Audio8-AI/Audio8_TTS.git" \
    "${AUDIO8_ROOT}" \
    "${AUDIO8_COMMIT}"

if [[ -x "${VENV_PATH}/bin/python" ]]; then
    echo "Updating existing SGLang-Omni venv at ${VENV_PATH}"
else
    echo "Creating SGLang-Omni venv at ${VENV_PATH} with ${PYTHON_BIN}"
    uv venv --python "${PYTHON_BIN}" "${VENV_PATH}"
fi

echo "Installing Audio8's validated CUDA runtime..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    "torch==2.9.1" \
    --torch-backend cu128
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    --editable "${SGLANG_ROOT}"
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    "transformers==4.57.1" \
    huggingface_hub \
    ninja

"${AUDIO8_ROOT}/sglang_omni/scripts/install_adapter.sh" "${SGLANG_ROOT}"

configure_cuda_env "${VENV_PATH}/bin/python"

echo "Verifying SGLang-Omni environment..."
PYTHONPATH="${SGLANG_ROOT}" "${VENV_PATH}/bin/python" - <<'PY'
from importlib.metadata import version
import torch

from sglang_omni.models.audio8_tts import factory

print("python =", __import__("sys").version.split()[0])
print("torch =", torch.__version__)
print("torch.cuda =", torch.version.cuda)
print("cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device =", torch.cuda.get_device_name(0))
    print("capability =", torch.cuda.get_device_capability(0))
print("transformers =", version("transformers"))
print("audio8_adapter =", factory.__name__)
PY

echo ""
echo "Audio8 SGLang-Omni environment ready at ${VENV_PATH}"
