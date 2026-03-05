#!/bin/bash
# Quick benchmark: compare llama-server vs mlx-vlm on the same prompt.
# Assumes one server is running on port 2027 (default) and the other on the port you pass as $1.
# Usage: ./benchmark.sh [alt_port]
#   No args  → just benchmarks whatever is on port 2027
#   alt_port → benchmarks both 2027 and alt_port

set -euo pipefail

PROMPT='Explain quantum entanglement in three sentences.'
MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"

bench() {
    local label="$1" port="$2"
    local url="http://localhost:${port}/chat/completions"

    echo "=== $label (port $port) ==="

    # Non-streaming request — grab timing + usage from the response
    local start end elapsed
    start=$(python3 -c 'import time; print(time.time())')

    local resp
    resp=$(curl -s "$url" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
            \"max_tokens\": 256,
            \"stream\": false
        }")

    end=$(python3 -c 'import time; print(time.time())')
    elapsed=$(python3 -c "print(f'{$end - $start:.2f}')")

    # Extract usage stats if present
    local prompt_tokens completion_tokens
    # Try both OpenAI (prompt_tokens/completion_tokens) and mlx-vlm (input_tokens/output_tokens) field names
    prompt_tokens=$(echo "$resp" | python3 -c "import sys,json; u=json.load(sys.stdin).get('usage',{}); print(u.get('prompt_tokens', u.get('input_tokens', '?')))" 2>/dev/null || echo "?")
    completion_tokens=$(echo "$resp" | python3 -c "import sys,json; u=json.load(sys.stdin).get('usage',{}); print(u.get('completion_tokens', u.get('output_tokens', '?')))" 2>/dev/null || echo "?")

    local content
    content=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "(failed to parse)")

    echo "Time:              ${elapsed}s"
    echo "Prompt tokens:     $prompt_tokens"
    echo "Completion tokens: $completion_tokens"
    if [[ "$completion_tokens" != "?" && "$completion_tokens" != "0" ]]; then
        local tps
        tps=$(python3 -c "print(f'{int($completion_tokens) / $elapsed:.1f}')")
        echo "Tokens/sec:        $tps"
    fi
    echo "Response:          ${content:0:200}"
    echo ""
}

bench "Primary" 2027

if [[ "${1:-}" != "" ]]; then
    bench "Alternate" "$1"
fi
