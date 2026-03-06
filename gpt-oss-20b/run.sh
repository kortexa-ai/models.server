#!/bin/bash

if ! command -v llama-server &>/dev/null; then
    echo "llama-server could not be found"
    exit 1
fi

llama-server -hf unsloth/gpt-oss-20b-GGUF:Q8_K_XL --alias gpt-oss-20b --host 0.0.0.0 --port 2020 \
    --jinja -ngl 99 --threads -1 \
    --ctx-size 32768 --temp 1.0 --top-p 1.0 --top-k 0 \
    --no-mmap --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    $*
