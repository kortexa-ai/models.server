#!/bin/bash

if ! command -v llama-server &>/dev/null; then
    echo "llama-server could not be found"
    exit 1
fi

if [[ "$(uname)" == "Darwin" ]]; then
    # Mac Mini M4 Pro — smaller quant, smaller KV cache
    QUANT="UD-Q4_K_XL"
    CACHE_TYPE="q4_0"
else
    # Ubuntu + RTX 6000 Blackwell
    QUANT="UD-Q8_K_XL"
    CACHE_TYPE="q8_0"
fi

llama-server -hf unsloth/gpt-oss-120b-GGUF:$QUANT --alias gpt-oss-120b --host 0.0.0.0 --port 2023 \
    --jinja -ngl 99 --threads -1 \
    --ctx-size 32768 --temp 1.0 --top-p 1.0 --top-k 0 \
    --flash-attn on \
    --cache-type-k $CACHE_TYPE --cache-type-v $CACHE_TYPE \
    $*
