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
        echo "For bench engines (vLLM, SGLang, MLX bench venv): run bench/setup.sh"
        ;;
    Linux)
        echo "Detected Linux (${ARCH})"
        echo ""
        echo "--- Checking llama-server ---"
        check_llama_server
        echo ""
        echo "--- Setting up vLLM environment ---"
        "${ROOT}/bench/setup-vllm.sh"
        echo ""
        echo "Setup complete!"
        ;;
    *)
        echo "Unsupported platform: ${OS}"
        exit 1
        ;;
esac
