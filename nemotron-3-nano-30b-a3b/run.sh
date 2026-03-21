#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Nemotron-3-Nano 30B-A3B on bare-metal vLLM
# Architecture: NemotronHForCausalLM (Mamba-2 hybrid + MoE, 30B total / 3.5B active)

MODELS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_DIR="${MODELS_ROOT}/bench"

# shellcheck source=../bench/common.sh
source "${BENCH_DIR}/common.sh"

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"
PORT="${PORT:-2034}"
HOST="${HOST:-0.0.0.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.2}"
REASONING_PARSER="${REASONING_PARSER:-nano_v3}"
REASONING_PARSER_PLUGIN="${REASONING_PARSER_PLUGIN:-${SCRIPT_DIR}/nano_v3_reasoning_parser.py}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

VENV_PATH="${VENV_PATH:-${BENCH_DIR}/.venv-vllm}"
PYTHON_BIN="${PYTHON_BIN:-$(venv_python "${VENV_PATH}")}"
VLLM_BIN="${VLLM_BIN:-$(venv_bin "${VENV_PATH}")/vllm}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${VLLM_BIN}" ]]; then
    echo "Error: vLLM is not installed in ${VENV_PATH}." >&2
    echo "Run bench/setup-vllm.sh first." >&2
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

# FlashInfer MOE FP4 backend (kernels pre-compiled at ~/.cache/flashinfer/)
export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-1}"
export VLLM_FLASHINFER_MOE_BACKEND="${VLLM_FLASHINFER_MOE_BACKEND:-throughput}"

configure_cuda_env "${PYTHON_BIN}"

echo "Starting Nemotron-3-Nano 30B-A3B (NVFP4) via bare-metal vLLM"
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Max model len: ${MAX_MODEL_LEN}"
echo "KV cache dtype: ${KV_CACHE_DTYPE}"
echo "Reasoning parser: ${REASONING_PARSER}"

exec "${VLLM_BIN}" serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name nemotron-3-nano-30b-a3b \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin "${REASONING_PARSER_PLUGIN}" \
    --reasoning-parser "${REASONING_PARSER}" \
    "$@"
