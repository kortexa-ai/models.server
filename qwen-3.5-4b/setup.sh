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

    # Patch mlx-vlm to forward extra_body params (e.g. enable_thinking) to the chat template
    SITE=$(.venv/bin/python -c "import mlx_vlm; print(mlx_vlm.__path__[0])")
    # 1) server.py: forward extra body params to apply_chat_template
    sed -i '' 's/num_audios=len(audio),$/num_audios=len(audio),\n            **({k: v for k, v in request.__pydantic_extra__.items()} if request.__pydantic_extra__ else {}),/' "$SITE/server.py"
    # 2) prompt_utils.py: forward kwargs to get_chat_template
    sed -i '' 's/return get_chat_template(processor, messages, add_generation_prompt)$/return get_chat_template(processor, messages, add_generation_prompt, **kwargs)/' "$SITE/prompt_utils.py"
    echo "Applied mlx-vlm patches"

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
