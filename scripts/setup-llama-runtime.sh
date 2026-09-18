#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: scripts/setup-llama-runtime.sh <model-dir>" >&2
    exit 1
fi
MODEL_DIR="$(CDPATH= cd "$1" && pwd)"
SCRIPTS_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
source "${SCRIPTS_DIR}/setup-common.sh"

CONFIG="$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"
eval "$CONFIG"
if [[ -z "${LLAMA_RUNTIME_REPO:-}" || -z "${LLAMA_RUNTIME_REVISION:-}" || -z "${LLAMA_RUNTIME_DIR:-}" ]]; then
    echo "Error: ${MODEL_NAME} has no isolated llama.cpp runtime configured." >&2
    exit 1
fi

require_command git
require_command cmake
ENGINE_DIR="${ROOT}/${LLAMA_RUNTIME_DIR}"

if [[ ! -d "${ENGINE_DIR}/.git" ]]; then
    git clone --depth 1 --no-checkout "$LLAMA_RUNTIME_REPO" "$ENGINE_DIR"
else
    if [[ "$(git -C "$ENGINE_DIR" remote get-url origin)" != "$LLAMA_RUNTIME_REPO" ]]; then
        echo "Error: unexpected origin in ${ENGINE_DIR}; refusing to modify it." >&2
        exit 1
    fi
    if [[ -n "$(git -C "$ENGINE_DIR" status --porcelain)" ]]; then
        echo "Error: ${ENGINE_DIR} has local changes; refusing to modify it." >&2
        exit 1
    fi
fi
git -C "$ENGINE_DIR" fetch --depth 1 origin "$LLAMA_RUNTIME_REVISION"
git -C "$ENGINE_DIR" checkout --detach "$LLAMA_RUNTIME_REVISION"

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=ON)
case "$(uname -s)" in
    Darwin)
        CMAKE_ARGS+=(-DGGML_METAL=ON)
        ;;
    Linux)
        if [[ -x /usr/local/cuda/bin/nvcc ]] || command -v nvcc >/dev/null 2>&1; then
            export PATH="/usr/local/cuda/bin:${PATH}"
            CMAKE_ARGS+=(-DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=89;120" -DGGML_CUDA_NCCL=OFF)
        else
            CMAKE_ARGS+=(-DGGML_CUDA=OFF)
        fi
        ;;
    *)
        echo "Error: unsupported platform $(uname -s)." >&2
        exit 1
        ;;
esac
cmake -S "$ENGINE_DIR" -B "${ENGINE_DIR}/build" "${CMAKE_ARGS[@]}"
cmake --build "${ENGINE_DIR}/build" --config Release --target llama-server --parallel 4
"${ENGINE_DIR}/build/bin/llama-server" --version
echo "Isolated runtime ready at ${ENGINE_DIR}/build/bin/llama-server; shared binaries are unchanged."
