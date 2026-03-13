#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Nemotron-3-Super 120B-A12B on llama-server (llama.cpp)
# Architecture: NemotronHForCausalLM (Mamba-2 hybrid + LatentMoE, 120B total / 12B active)
# GGUF Q4_K_M is ~82.5GB — tight on 128GB, so context capped at 16k

QUANT="${QUANT:-Q4_K_M}"
PORT="${PORT:-2033}"
HOST="${HOST:-0.0.0.0}"
CONTEXT="${CONTEXT:-16384}"
PARALLEL="${PARALLEL:-1}"
CACHE_TYPE="${CACHE_TYPE:-q4_0}"

# Model card recommends temp=1.0, top_p=0.95 for all tasks
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"

HF_REPO="unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF"

echo "Starting Nemotron-3-Super 120B-A12B (GGUF ${QUANT}) via llama-server"
echo "Port: ${PORT}"
echo "Context: ${CONTEXT}"
echo "Cache type: ${CACHE_TYPE}"
echo "HF repo: ${HF_REPO}:UD-${QUANT}"

exec llama-server \
    -hf "${HF_REPO}:UD-${QUANT}" \
    --alias nemotron-3-super-120b-a12b \
    --host "$HOST" \
    --port "$PORT" \
    --jinja \
    -c "$CONTEXT" \
    -ngl 99 \
    --threads -1 \
    --parallel "$PARALLEL" \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --no-mmap \
    --flash-attn on \
    --cache-type-k "$CACHE_TYPE" \
    --cache-type-v "$CACHE_TYPE" \
    "$@"
