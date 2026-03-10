#!/usr/bin/env bash
set -euo pipefail

bench_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

require_command() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "Error: ${cmd} is required but was not found in PATH." >&2
        exit 1
    fi
}

python_site_packages() {
    local python_bin="$1"
    "${python_bin}" - <<'PY'
import site
import sys

for path in site.getsitepackages():
    if "site-packages" in path or "dist-packages" in path:
        print(path)
        break
else:
    for path in sys.path:
        if "site-packages" in path or "dist-packages" in path:
            print(path)
            break
PY
}

configure_cuda_env() {
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

    if [[ -x "${CUDA_HOME}/bin/ptxas" && -z "${TRITON_PTXAS_PATH:-}" ]]; then
        export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"
    fi

    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"

    if [[ -d "${CUDA_HOME}/bin" ]]; then
        export PATH="${CUDA_HOME}/bin:${PATH}"
    fi

    local lib_dirs=()
    if [[ -d /usr/local/lib ]]; then
        lib_dirs+=(/usr/local/lib)
    fi
    if [[ -d "${CUDA_HOME}/lib64" ]]; then
        lib_dirs+=("${CUDA_HOME}/lib64")
    fi

    if [[ -n "${1:-}" && -x "$1" ]]; then
        local site_packages
        site_packages="$(python_site_packages "$1")"
        if [[ -n "${site_packages}" && -d "${site_packages}/torch/lib" ]]; then
            lib_dirs+=("${site_packages}/torch/lib")
        fi
        if [[ -n "${site_packages}" && -d "${site_packages}/nvidia" ]]; then
            while IFS= read -r lib_dir; do
                lib_dirs+=("${lib_dir}")
            done < <(find "${site_packages}/nvidia" -maxdepth 2 -type d -name lib | sort)
        fi
    fi

    if (( ${#lib_dirs[@]} > 0 )); then
        export LD_LIBRARY_PATH="$(IFS=:; printf '%s' "${lib_dirs[*]}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}

venv_python() {
    local venv_path="$1"
    echo "${venv_path}/bin/python"
}

venv_bin() {
    local venv_path="$1"
    echo "${venv_path}/bin"
}
