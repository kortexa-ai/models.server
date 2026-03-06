#!/bin/bash

if ! command -v llama-server &>/dev/null; then
    echo "llama-server could not be found"
    exit 1
fi

if [[ "$(uname)" == "Darwin" ]]; then
    # Mac Mini M4 Pro — smaller quant, smaller KV cache
    QUANT="Q4_K_M"
    CACHE_TYPE="q4_0"
else
    # Ubuntu + RTX 6000 Blackwell
    QUANT="Q8_0"
    CACHE_TYPE="q8_0"
fi

llama-server -hf LiquidAI/LFM2-24B-A2B-GGUF:$QUANT --alias lfm-2-24b-a2b --host 0.0.0.0 --port 2028 \
    --jinja -ngl 99 --threads -1 \
    --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40 \
    --no-mmap --flash-attn on \
    --cache-type-k $CACHE_TYPE --cache-type-v $CACHE_TYPE \
    $*
