#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Nemotron-Cascade-2 30B-A3B via llama-server (llama.cpp)
# Architecture: Hybrid Mamba-2 + MoE, 30B total / 3B active, 262K native context
# GGUF Q8_0 is ~33.5GB — fits in 60GB with ~27GB left for fp16 KV cache
# Model card recommends temp=1.0, top_p=0.95
# For more context length at the cost of KV precision, use CACHE_TYPE=q4_0

QUANT="${QUANT:-Q8_0}"
PORT="${PORT:-2035}"
HOST="${HOST:-0.0.0.0}"
CONTEXT="${CONTEXT:-65536}"
PARALLEL="${PARALLEL:-8}"
CACHE_TYPE="${CACHE_TYPE:-f16}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"

HF_REPO="bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF"

echo "Starting Nemotron-Cascade-2 30B-A3B (GGUF ${QUANT}) via llama-server"
echo "Port: ${PORT}"
echo "Context: ${CONTEXT}"
echo "Cache type: ${CACHE_TYPE}"
echo "HF repo: ${HF_REPO}:${QUANT}"

exec llama-server \
    -hf "${HF_REPO}:${QUANT}" \
    --alias nemotron-cascade-2-30b-a3b \
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
