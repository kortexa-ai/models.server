#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_ROOT="$(cd "${PROJECT_ROOT}/../llama.cpp" && pwd)"
BENCH_PYTHON="${PROJECT_ROOT}/.venv-bench/bin/python"
REQUIREMENTS="${LLAMA_ROOT}/tools/server/bench/speed-bench/requirements.txt"

command -v uv >/dev/null 2>&1 || {
    echo "Missing required command: uv" >&2
    exit 1
}
[[ -f "$REQUIREMENTS" ]] || {
    echo "Missing upstream SPEED-Bench requirements: $REQUIREMENTS" >&2
    exit 1
}

if [[ ! -x "$BENCH_PYTHON" ]]; then
    uv venv --python python3 "${PROJECT_ROOT}/.venv-bench"
fi

uv pip install --python "$BENCH_PYTHON" -r "$REQUIREMENTS"

echo "Benchmark environment is ready: $BENCH_PYTHON"
echo "No model was started and no benchmark was run."
