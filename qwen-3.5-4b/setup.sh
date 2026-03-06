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
    echo "Detected macOS ($ARCH) - installing mlx-vlm backend..."
    uv pip install 'mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git' torch torchvision

    echo "Backend: mlx-vlm"
elif [[ "$OS" == "Linux" ]]; then
    echo "Detected Linux - installing vLLM backend..."
    uv pip install vllm
    echo "Backend: vLLM"
else
    echo "Unsupported platform: $OS"
    exit 1
fi

echo ""
echo "Setup complete! Run ./run.sh to start the server."
