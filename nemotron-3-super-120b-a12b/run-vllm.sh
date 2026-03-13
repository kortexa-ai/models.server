#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Nemotron-3-Super 120B-A12B on bare-metal vLLM
# From NVIDIA forum: uses Marlin NVFP4 backend for FP4→BF16 dequant
# Architecture: NemotronHForCausalLM (Mamba-2 hybrid + LatentMoE, 120B total / 12B active)

MODELS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_DIR="${MODELS_ROOT}/bench"

# shellcheck source=../bench/common.sh
source "${BENCH_DIR}/common.sh"

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
PORT="${PORT:-2033}"
HOST="${HOST:-0.0.0.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
REASONING_PARSER="${REASONING_PARSER:-super_v3}"
REASONING_PARSER_PLUGIN="${REASONING_PARSER_PLUGIN:-${SCRIPT_DIR}/super_v3_reasoning_parser.py}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

VENV_PATH="${VENV_PATH:-${BENCH_DIR}/.venv-vllm}"
PYTHON_BIN="${PYTHON_BIN:-$(venv_python "${VENV_PATH}")}"
VLLM_BIN="${VLLM_BIN:-$(venv_bin "${VENV_PATH}")/vllm}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${VLLM_BIN}" ]]; then
    echo "Error: vLLM is not installed in ${VENV_PATH}." >&2
    echo "Run ./setup.sh or bench/setup-vllm.sh first." >&2
    exit 1
fi

export PATH="$(venv_bin "${VENV_PATH}"):${PATH}"

# Cache paths — avoid root-owned Docker leftovers
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

# Marlin NVFP4 backend env vars (from NVIDIA forum)
export VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND:-marlin}"
export VLLM_TEST_FORCE_FP8_MARLIN="${VLLM_TEST_FORCE_FP8_MARLIN:-1}"
export VLLM_MARLIN_USE_ATOMIC_ADD="${VLLM_MARLIN_USE_ATOMIC_ADD:-1}"

configure_cuda_env "${PYTHON_BIN}"

echo "Starting Nemotron-3-Super 120B-A12B (NVFP4) via bare-metal vLLM"
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Max model len: ${MAX_MODEL_LEN}"
echo "KV cache dtype: ${KV_CACHE_DTYPE}"
echo "Reasoning parser: ${REASONING_PARSER}"
echo "NVFP4 GEMM backend: ${VLLM_NVFP4_GEMM_BACKEND}"

exec "${VLLM_BIN}" serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name nemotron-3-super-120b-a12b \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --attention-backend TRITON_ATTN \
    --enable-chunked-prefill \
    --mamba-ssm-cache-dtype float16 \
    --reasoning-parser-plugin "${REASONING_PARSER_PLUGIN}" \
    --reasoning-parser "${REASONING_PARSER}" \
    "$@"
