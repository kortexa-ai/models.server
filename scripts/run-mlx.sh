#!/bin/bash
set -euo pipefail

MODEL_DIR="$1"; shift
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"

eval "$(python3 "${SCRIPTS_DIR}/parse-config.py" "${MODEL_DIR}/model.json")"

if [[ "${MLX_SUPPORTED:-}" == "false" ]]; then
    echo "Not supported: ${MODEL_NAME} is not available for MLX." >&2
    exit 1
fi

if [[ -d "${ROOT}/.venv-mlx" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/.venv-mlx/bin/activate"
fi

PORT="${PORT:-$MODEL_PORT}"
HOST="${HOST:-0.0.0.0}"

BACKEND="${MLX_BACKEND:-mlx_vlm}"

echo "Starting ${MODEL_NAME} via ${BACKEND} on port ${PORT}..."
MODEL_PATH="$MLX_REPO"
if [[ -n "${MLX_SUBDIR:-}" ]]; then
    echo "Resolving ${MLX_REPO}/${MLX_SUBDIR} from its multi-precision repository..."
    MODEL_PATH="$(
        MLX_MODEL_REPO="$MLX_REPO" MLX_MODEL_SUBDIR="$MLX_SUBDIR" python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo = os.environ["MLX_MODEL_REPO"]
subdir = os.environ["MLX_MODEL_SUBDIR"].strip("/")
snapshot = snapshot_download(repo, allow_patterns=[f"{subdir}/*"])
model_path = Path(snapshot) / subdir
if not (model_path / "config.json").is_file():
    raise SystemExit(f"MLX model subdirectory has no config.json: {model_path}")
print(model_path)
PY
    )"
fi

SERVER_ARGS=(
    --model "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
)

LM_ARGS=()
if [[ -n "${MLX_PROMPT_CONCURRENCY:-}" ]]; then
    LM_ARGS+=(--prompt-concurrency "$MLX_PROMPT_CONCURRENCY")
fi
if [[ -n "${MLX_DECODE_CONCURRENCY:-}" ]]; then
    LM_ARGS+=(--decode-concurrency "$MLX_DECODE_CONCURRENCY")
fi
if [[ -n "${MLX_PROMPT_CACHE_SIZE:-}" ]]; then
    LM_ARGS+=(--prompt-cache-size "$MLX_PROMPT_CACHE_SIZE")
fi
if [[ -n "${MLX_PROMPT_CACHE_BYTES:-}" ]]; then
    LM_ARGS+=(--prompt-cache-bytes "$MLX_PROMPT_CACHE_BYTES")
fi
if [[ -n "${MLX_CHAT_TEMPLATE_ARGS:-}" ]]; then
    LM_ARGS+=(--chat-template-args "$MLX_CHAT_TEMPLATE_ARGS")
fi

VLM_ARGS=()
if [[ -n "${MLX_VISION_CACHE_SIZE:-}" ]]; then
    VLM_ARGS+=(--vision-cache-size "$MLX_VISION_CACHE_SIZE")
fi
if [[ -n "${MLX_MAX_KV_SIZE:-}" ]]; then
    VLM_ARGS+=(--max-kv-size "$MLX_MAX_KV_SIZE")
fi
if [[ -n "${MLX_MAX_NUM_SEQS:-}" ]]; then
    VLM_ARGS+=(--max-num-seqs "$MLX_MAX_NUM_SEQS")
fi
if [[ -n "${MLX_KV_BITS:-}" ]]; then
    VLM_ARGS+=(--kv-bits "$MLX_KV_BITS")
fi
if [[ -n "${MLX_KV_QUANT_SCHEME:-}" ]]; then
    VLM_ARGS+=(--kv-quant-scheme "$MLX_KV_QUANT_SCHEME")
fi
if [[ -n "${MLX_KV_GROUP_SIZE:-}" ]]; then
    VLM_ARGS+=(--kv-group-size "$MLX_KV_GROUP_SIZE")
fi
if [[ -n "${MLX_QUANTIZED_KV_START:-}" ]]; then
    VLM_ARGS+=(--quantized-kv-start "$MLX_QUANTIZED_KV_START")
fi

COMMON_MLX_ARGS=()
if [[ -n "${MLX_PREFILL_STEP_SIZE:-}" ]]; then
    COMMON_MLX_ARGS+=(--prefill-step-size "$MLX_PREFILL_STEP_SIZE")
fi
if [[ -n "${MLX_MAX_TOKENS:-}" ]]; then
    COMMON_MLX_ARGS+=(--max-tokens "$MLX_MAX_TOKENS")
fi

DRAFT_ARGS=()
if [[ -n "${MLX_DRAFT_MODEL:-}" && "${MLX_DISABLE_DRAFT:-}" != "1" ]]; then
    if [[ "${MLX_DRAFT_ENABLED:-true}" == "false" && "${MLX_FORCE_DRAFT:-}" != "1" ]]; then
        echo "Speculative drafter configured but disabled for vanilla ${BACKEND}; set MLX_FORCE_DRAFT=1 to override."
    else
        echo "Using speculative drafter: ${MLX_DRAFT_MODEL}"
        DRAFT_ARGS+=(--draft-model "$MLX_DRAFT_MODEL")

        if [[ -n "${MLX_DRAFT_KIND:-}" ]]; then
            DRAFT_ARGS+=(--draft-kind "$MLX_DRAFT_KIND")
        fi

        if [[ -n "${MLX_DRAFT_BLOCK_SIZE:-}" ]]; then
            DRAFT_ARGS+=(--draft-block-size "$MLX_DRAFT_BLOCK_SIZE")
        fi
    fi
fi

case "$BACKEND" in
    mlx_lm)
        exec python -m mlx_lm server \
            "${SERVER_ARGS[@]}" \
            ${COMMON_MLX_ARGS[@]+"${COMMON_MLX_ARGS[@]}"} \
            ${LM_ARGS[@]+"${LM_ARGS[@]}"} \
            "$@"
        ;;
    *)
        exec python -m "${BACKEND}.server" \
            "${SERVER_ARGS[@]}" \
            ${COMMON_MLX_ARGS[@]+"${COMMON_MLX_ARGS[@]}"} \
            ${VLM_ARGS[@]+"${VLM_ARGS[@]}"} \
            ${DRAFT_ARGS[@]+"${DRAFT_ARGS[@]}"} \
            "$@"
        ;;
esac
