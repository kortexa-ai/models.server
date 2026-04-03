#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Gemma 4 26B-A4B on bare-metal vLLM with online FP8 quantization
# Architecture: MoE, 26B total / 4B active, up to 256K native context
# 30 layers: 25 sliding (1024 window, 8 kv_heads, 256 head_dim) + 5 full (2 kv_heads, 512 head_dim)
# KV math (fp8, 4 reqs @ 131K):
#   sliding: 2 * 25 * 8 * 256 * 1 * 1024  * 4 ≈ 0.4 GiB
#   full:    2 *  5 * 2 * 512 * 1 * 131072 * 4 ≈ 5.4 GiB  → ~6 GiB total, budget 8 GiB

MODELS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_DIR="${MODELS_ROOT}/bench"

# shellcheck source=../bench/common.sh
source "${BENCH_DIR}/common.sh"

MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
PORT="${PORT:-2036}"
HOST="${HOST:-0.0.0.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-8589934592}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
QUANTIZATION="${QUANTIZATION:-fp8}"

VENV_PATH="${VENV_PATH:-${BENCH_DIR}/.venv-vllm}"
PYTHON_BIN="${PYTHON_BIN:-$(venv_python "${VENV_PATH}")}"
VLLM_BIN="${VLLM_BIN:-$(venv_bin "${VENV_PATH}")/vllm}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${VLLM_BIN}" ]]; then
    echo "Error: vLLM is not installed in ${VENV_PATH}." >&2
    echo "Run ./setup.sh or bench/setup-vllm.sh first." >&2
    exit 1
fi

export PATH="$(venv_bin "${VENV_PATH}"):${PATH}"

# Cache paths
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${BENCH_DIR}/.cache/vllm}"
VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${BENCH_DIR}/.config/vllm}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${VLLM_CACHE_ROOT}/torch_compile_cache}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

mkdir -p \
    "${VLLM_CACHE_ROOT}" \
    "${VLLM_CONFIG_ROOT}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}"
export VLLM_CACHE_ROOT VLLM_CONFIG_ROOT TORCHINDUCTOR_CACHE_DIR
export HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE

# Pass through HF token if set, or copy from default location
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

configure_cuda_env "${PYTHON_BIN}"

echo "Starting Gemma 4 26B-A4B (FP8) via bare-metal vLLM"
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "KV cache bytes: ${KV_CACHE_MEMORY_BYTES}"
echo "Max model len: ${MAX_MODEL_LEN}"
echo "Quantization: ${QUANTIZATION}"
echo "Attention backend: ${ATTENTION_BACKEND}"
echo "KV cache dtype: ${KV_CACHE_DTYPE}"

CMD=(
    "${VLLM_BIN}" serve "${MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name gemma-4-26b-a4b
    --tensor-parallel-size 1
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --attention-backend "${ATTENTION_BACKEND}"
    --enable-auto-tool-choice
    --tool-call-parser pythonic
)

# Skip --quantization for auto (lets vLLM detect from checkpoint)
if [[ "${QUANTIZATION}" != "auto" ]]; then
    CMD+=(--quantization "${QUANTIZATION}")
fi

exec "${CMD[@]}" "$@"
