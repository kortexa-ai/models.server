#!/usr/bin/env bash
# One-time, idempotent setup for a Shingi System One model (Linux + CUDA only).
# - Checks out the pinned shingi-27b package source into .engines/shingi/src
#   and installs it into .engines/shingi/venv (skipped when the stamp matches).
# - Packages with device-state caching build their corrected Prism runtime in
#   .engines/shingi/runtime. Older packages reuse the model's runtime directory when it is at the
#   pinned revision with shared libraries; otherwise builds its own copy under
#   .engines/shingi/prism. It never modifies the shared runtime directory.
# - Builds .engines/shingi/bin/readout against llama and the multimodal mtmd
#   library (skipped when the stamp matches).
# - Downloads the pinned weights (and vision projector, when configured) into
#   the Hugging Face cache if needed.
# run-shingi.sh only reads these files, so it also works under systemd.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: scripts/setup-shingi.sh <model-dir>" >&2
    exit 1
fi
MODEL_DIR="$(CDPATH= cd "$1" && pwd)"
SCRIPTS_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
source "${SCRIPTS_DIR}/setup-common.sh"

CONFIG="$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"
eval "$CONFIG"
if [[ "${SHINGI_SUPPORTED:-}" == "false" ]]; then
    echo "Error: ${MODEL_NAME} has no shingi configuration." >&2
    exit 1
fi
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Error: ${MODEL_NAME} currently runs only on Linux with CUDA." >&2
    exit 1
fi

require_command git
require_command uv
require_command c++

ENGINE_DIR="${ROOT}/.engines/shingi"
SRC="${ENGINE_DIR}/src"
VENV="${ENGINE_DIR}/venv"
READOUT="${ENGINE_DIR}/bin/readout"
mkdir -p "${ENGINE_DIR}/bin"

# Check out one pinned revision of a repository into an owned directory.
checkout_pinned() {
    local repo="$1" revision="$2" dir="$3"
    if [[ ! -d "${dir}/.git" ]]; then
        git init --quiet "$dir"
        git -C "$dir" remote add origin "$repo"
    elif [[ "$(git -C "$dir" remote get-url origin)" != "$repo" ]]; then
        echo "Error: unexpected origin in ${dir}; refusing to modify it." >&2
        exit 1
    elif [[ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]]; then
        echo "Error: ${dir} has local changes; refusing to modify it." >&2
        exit 1
    fi
    if [[ "$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)" != "$revision" ]]; then
        git -C "$dir" fetch --quiet --depth 1 origin "$revision"
        git -C "$dir" checkout --quiet --detach "$revision"
    fi
}

# 1. Package source and Python environment.
checkout_pinned "$SHINGI_PACKAGE_REPO" "$SHINGI_PACKAGE_REVISION" "$SRC"
if [[ -x "${VENV}/bin/python" && "$(cat "${ENGINE_DIR}/package.stamp" 2>/dev/null || true)" == "$SHINGI_PACKAGE_REVISION" ]]; then
    echo "shingi: package already installed at ${SHINGI_PACKAGE_REVISION}"
else
    [[ -x "${VENV}/bin/python" ]] || uv venv --quiet --python ">=3.11" "$VENV"
    uv pip install --quiet --python "${VENV}/bin/python" --reinstall-package shingi-27b "$SRC"
    printf '%s\n' "$SHINGI_PACKAGE_REVISION" > "${ENGINE_DIR}/package.stamp"
    echo "shingi: installed package ${SHINGI_PACKAGE_REVISION} into ${VENV}"
fi

# 2. Prism runtime headers and shared libraries.
runtime_usable() {
    local dir="$1"
    [[ "$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)" == "$SHINGI_RUNTIME_REVISION" ]] \
        && [[ -z "$(git -C "$dir" status --porcelain --untracked-files=no 2>/dev/null)" ]] \
        && [[ -f "${dir}/include/llama.h" && -f "${dir}/ggml/include/ggml-backend.h" ]] \
        && [[ -f "${dir}/vendor/nlohmann/json.hpp" ]] \
        && [[ -f "${dir}/build/bin/libllama.so" && -f "${dir}/build/bin/libggml.so" ]] \
        && [[ -f "${dir}/build/bin/libggml-base.so" ]] \
        && [[ -f "${dir}/tools/mtmd/mtmd.h" && -f "${dir}/build/bin/libmtmd.so" ]]
}

PRISM="${ROOT}/${SHINGI_RUNTIME_DIR}"
if [[ -f "${SRC}/scripts/patch-prism.py" ]]; then
    # Quantized on-device state needs the package's exact correction. Keep the
    # shared Bonsai runtime untouched, and let the package verify its managed patch.
    echo "shingi: building the isolated corrected Prism runtime"
    SHINGI_HOME="${ENGINE_DIR}/runtime" bash "${SRC}/scripts/build.sh"
    PRISM="${ENGINE_DIR}/runtime/prism"
    [[ "$(git -C "$PRISM" rev-parse HEAD)" == "$SHINGI_RUNTIME_REVISION" ]] || {
        echo "Error: package and model.json Prism revisions differ." >&2
        exit 1
    }
    python3 "${SRC}/scripts/patch-prism.py" "$PRISM" --check
elif runtime_usable "$PRISM"; then
    echo "shingi: reusing the shared Prism runtime in ${PRISM}"
else
    PRISM="${ENGINE_DIR}/prism"
    echo "shingi: the shared Prism runtime is missing, stale, or static; using ${PRISM}"
    if ! runtime_usable "$PRISM"; then
        require_command cmake
        if [[ -x /usr/local/cuda/bin/nvcc ]]; then
            export PATH="/usr/local/cuda/bin:${PATH}"
        fi
        require_command nvcc
        checkout_pinned "$SHINGI_RUNTIME_REPO" "$SHINGI_RUNTIME_REVISION" "$PRISM"
        jobs="$(nproc)"
        [[ "$jobs" -le 8 ]] || jobs=8  # CUDA kernel compilation needs several GB of RAM per job.
        cmake -S "$PRISM" -B "${PRISM}/build" \
            -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
            "-DCMAKE_CUDA_ARCHITECTURES=86;89;120;121" -DBUILD_SHARED_LIBS=ON \
            -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
            -DLLAMA_BUILD_MTMD=ON -DMTMD_VIDEO=OFF
        # mtmd is the image encoder library the readout links for the vision projector.
        cmake --build "${PRISM}/build" --target llama mtmd --parallel "$jobs"
    fi
fi

# 3. Native readout, linked to the runtime and mtmd with an rpath.
# run-shingi.sh checks only the leading package and runtime revisions.
stamp="${SHINGI_PACKAGE_REVISION} ${SHINGI_RUNTIME_REVISION} mtmd ${PRISM}"
if [[ -x "$READOUT" && "$(cat "${READOUT}.stamp" 2>/dev/null || true)" == "$stamp" ]]; then
    echo "shingi: readout already built for ${stamp}"
else
    c++ -std=c++17 -O2 -Wall -Wextra "${SRC}/src/native/readout.cpp" \
        -I"${PRISM}/include" -I"${PRISM}/ggml/include" -I"${PRISM}/vendor" -I"${PRISM}/tools/mtmd" \
        -L"${PRISM}/build/bin" -Wl,-rpath,"${PRISM}/build/bin" \
        -lmtmd -lllama -lggml -lggml-base -o "${READOUT}.tmp"
    mv "${READOUT}.tmp" "$READOUT"
    printf '%s\n' "$stamp" > "${READOUT}.stamp"
    echo "shingi: built ${READOUT}"
fi

# 4. Pinned weights (and the optional vision projector) in the standard Hugging Face cache.
for filename in "$SHINGI_MODEL_FILE" "$SHINGI_CALIBRATION_FILE" ${SHINGI_PROJECTOR_FILE:+"$SHINGI_PROJECTOR_FILE"}; do
    "${VENV}/bin/python" -c 'import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(sys.argv[1], sys.argv[2], revision=sys.argv[3]))' \
        "$SHINGI_WEIGHTS_REPO" "$filename" "$SHINGI_WEIGHTS_REVISION"
done

echo "shingi: ${MODEL_NAME} is ready. Start it with ./run.sh ${MODEL_ID}"
