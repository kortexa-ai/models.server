#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

OS="$(uname -s)"
ARCH="$(uname -m)"

# shellcheck disable=SC1091
source ../setup-common.sh

if [[ "$OS" == "Darwin" ]]; then
    echo "Detected macOS ($ARCH) - reusing the shared MLX environment."
    exec ../setup.sh
elif [[ "$OS" == "Linux" ]]; then
    echo "Detected Linux ($ARCH) - installing a local vLLM environment."
    setup_local_vllm_env "$(pwd)"
else
    echo "Unsupported platform: $OS"
    exit 1
fi

echo ""
echo "Setup complete! Run ./run.sh to start the server."
