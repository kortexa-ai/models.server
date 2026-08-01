#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# shellcheck disable=SC1091
source "${ROOT}/scripts/setup-common.sh"

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Darwin)
        echo "Detected macOS (${ARCH}) — setting up MLX environment..."
        "${ROOT}/scripts/setup-mlx.sh"
        ;;
    Linux)
        echo "Detected Linux (${ARCH})"
        echo ""
        echo "--- Checking llama-server ---"
        require_command llama-server
        echo "Found llama-server at $(command -v llama-server)"
        echo ""
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "--- Setting up vLLM environment ---"
            "${ROOT}/scripts/setup-vllm.sh"
        else
            echo "No NVIDIA GPU detected, skipping vLLM setup."
            echo ""
            echo "--- Setting up Transformers CPU environment ---"
            "${ROOT}/scripts/setup-transformers.sh"
        fi
        ;;
    *)
        echo "Unsupported platform: ${OS}"
        exit 1
        ;;
esac

echo ""
echo "Setup complete!"
