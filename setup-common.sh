#!/bin/bash

models_server_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: $cmd is not installed."
        exit 1
    fi
}

setup_shared_mlx_env() {
    local root
    root="$(models_server_root)"

    require_command uv

    echo "Creating/updating shared virtual environment at ${root}/.venv-mlx..."
    uv venv "${root}/.venv-mlx"

    # shellcheck disable=SC1091
    source "${root}/.venv-mlx/bin/activate"
    uv pip install --upgrade \
        'mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git' \
        mlx-lm \
        torch \
        torchvision

    echo "Shared MLX environment ready at ${root}/.venv-mlx"
}

check_llama_server() {
    require_command llama-server
    echo "Found llama-server at $(command -v llama-server)"
}

setup_local_vllm_env() {
    local model_dir="$1"

    require_command uv

    echo "Creating/updating local virtual environment at ${model_dir}/.venv..."
    uv venv "${model_dir}/.venv"

    # shellcheck disable=SC1091
    source "${model_dir}/.venv/bin/activate"
    uv pip install --upgrade vllm

    echo "Local vLLM environment ready at ${model_dir}/.venv"
}
