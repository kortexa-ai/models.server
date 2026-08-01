#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Determine model directory
if [[ $# -ge 1 && -d "${ROOT}/$1" && -f "${ROOT}/$1/model.json" ]]; then
    # ./run.sh qwen-3.5-4b
    MODEL_DIR="${ROOT}/$1"; shift
elif [[ $# -ge 1 && -d "$1" && -f "$1/model.json" ]]; then
    # ./run.sh /absolute/path/to/model
    MODEL_DIR="$(cd "$1" && pwd)"; shift
elif [[ -f "./model.json" ]]; then
    # cd qwen-3.5-4b && ../run.sh
    MODEL_DIR="$(pwd)"
else
    echo "Usage: run.sh [model-dir] [--engine llama|vllm|mlx|cpu|transformers] [extra args...]" >&2
    echo "" >&2
    echo "Run from a model directory:  cd qwen-3.5-4b && ../run.sh" >&2
    echo "Or specify the model:        ./run.sh qwen-3.5-4b" >&2
    exit 1
fi

# Parse --engine override
ENGINE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine) ENGINE="$2"; shift 2 ;;
        *) break ;;
    esac
done

# Honor a model-specific default before platform auto-detection. This keeps
# tiny CPU-first models off GPU backends unless explicitly overridden.
if [[ -z "$ENGINE" ]]; then
    ENGINE=$(python3 -c "import json; m=json.load(open('${MODEL_DIR}/model.json')); print(m.get('default_engine', ''))")
fi

# Auto-detect engine from platform and model config
if [[ -z "$ENGINE" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
        # Prefer MLX, but fall back to llama.cpp (Metal) for models with no MLX config
        HAS_MLX=$(python3 -c "import json; m=json.load(open('${MODEL_DIR}/model.json')); print('yes' if m.get('mlx') else 'no')")
        if [[ "$HAS_MLX" == "yes" ]]; then
            ENGINE=mlx
        else
            ENGINE=llama
        fi
    elif [[ "$(uname -m)" == "aarch64" && ! -d /usr/local/cuda ]]; then
        ENGINE=cpu
    else
        # Default to llama.cpp, unless model has no GGUF (NVFP4 models → vllm)
        HAS_LLAMA=$(python3 -c "import json; m=json.load(open('${MODEL_DIR}/model.json')); print('yes' if m.get('llama') else 'no')")
        if [[ "$HAS_LLAMA" == "yes" ]]; then
            ENGINE=llama
        else
            ENGINE=vllm
        fi
    fi
fi

exec "${ROOT}/scripts/run-${ENGINE}.sh" "${MODEL_DIR}" "$@"
