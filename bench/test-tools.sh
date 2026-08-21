#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

bash -n \
    bench/setup.sh \
    bench/capture-metadata.sh \
    bench/run-suite.sh
python3 -m py_compile \
    bench/capability-probe.py \
    bench/run-speed-bench.py

./bench/run-suite.sh lfm2.5-vl-3b >/dev/null
./bench/run-suite.sh qwen-3.8-27b --suite smoke >/dev/null
./bench/run-suite.sh lfm2.5-embedding-350m >/dev/null

echo "Benchmark tool syntax and dry runs passed; no inference requests were sent."
