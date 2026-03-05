#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed."
    exit 1
fi

echo "Creating virtual environment..."
uv venv

OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" == "Darwin" ]]; then
    echo "Detected macOS ($ARCH) - installing MLX-LM backend..."
    uv pip install mlx-lm
    echo "Backend: MLX-LM"
elif [[ "$OS" == "Linux" ]]; then
    # Check for NVIDIA GPU
    if command -v nvidia-smi &>/dev/null; then
        echo "Detected Linux with NVIDIA GPU - installing vLLM backend..."
        uv pip install vllm
        echo "Backend: vLLM (CUDA)"
    else
        echo "Detected Linux without NVIDIA GPU - installing vLLM (CPU)..."
        uv pip install vllm
        echo "Backend: vLLM (CPU)"
    fi
else
    echo "Unsupported platform: $OS"
    exit 1
fi

echo ""
echo "Setup complete! Run ./run.sh to start the server."
