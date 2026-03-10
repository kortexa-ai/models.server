#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

OS="$(uname -s)"
ARCH="$(uname -m)"

echo "=== Bench Setup ==="
echo "Platform: ${OS} (${ARCH})"
echo ""

SETUP_MLX=0
SETUP_VLLM=0
SETUP_SGLANG=0
HAS_LLAMA=0

case "${OS}" in
    Darwin)
        echo "--- Setting up MLX environment (macOS native) ---"
        "${SCRIPT_DIR}/setup-mlx.sh"
        SETUP_MLX=1
        echo ""
        ;;
    Linux)
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "--- Setting up vLLM environment (Linux + CUDA) ---"
            "${SCRIPT_DIR}/setup-vllm.sh"
            SETUP_VLLM=1
            echo ""

            echo "--- Setting up SGLang environment (Linux + CUDA) ---"
            "${SCRIPT_DIR}/setup-sglang.sh"
            SETUP_SGLANG=1
            echo ""
        else
            echo "No NVIDIA GPU detected, skipping vLLM and SGLang setup."
            echo ""
        fi
        ;;
    *)
        echo "Unsupported platform: ${OS}"
        exit 1
        ;;
esac

# Check for llama-server on all platforms
echo "--- Checking llama-server ---"
if command -v llama-server >/dev/null 2>&1; then
    echo "Found llama-server at $(command -v llama-server)"
    HAS_LLAMA=1
else
    echo "llama-server not found in PATH (optional — build from api.server/llama.cpp/build-llama.sh)"
fi
echo ""

echo "=== Setup Summary ==="
if (( SETUP_MLX )); then
    echo "  MLX (mlx-vlm/mlx-lm): OK — ${SCRIPT_DIR}/.venv-mlx"
fi
if (( SETUP_VLLM )); then
    echo "  vLLM:                  OK — ${SCRIPT_DIR}/.venv-vllm"
fi
if (( SETUP_SGLANG )); then
    echo "  SGLang:                OK — ${SCRIPT_DIR}/.venv-sglang"
fi
if (( HAS_LLAMA )); then
    echo "  llama-server:          OK — $(command -v llama-server)"
else
    echo "  llama-server:          not found"
fi
echo ""
echo "Done."
