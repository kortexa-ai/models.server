#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OS="$(uname -s)"
ARCH="$(uname -m)"

# shellcheck disable=SC1091
source "${ROOT}/setup-common.sh"

case "$OS" in
    Darwin)
        echo "Detected macOS (${ARCH}) - preparing shared MLX environment..."
        setup_shared_mlx_env
        echo ""
        echo "Setup complete! Shared environment is ready for MLX-backed models."
        ;;
    Linux)
        echo "Detected Linux (${ARCH}) - checking llama-server..."
        check_llama_server
        if [[ -d "${ROOT}/vllm-spark" ]]; then
            echo "Experimental Spark vLLM tooling lives in ${ROOT}/vllm-spark"
        fi
        echo ""
        echo "Setup complete! Llama-backed models can use the system llama-server."
        ;;
    *)
        echo "Unsupported platform: ${OS}"
        exit 1
        ;;
esac
