#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: bench/run-suite.sh <model-id> [--suite smoke|standard] [--output DIR] [--execute]

Without --execute, print the plan and send no inference requests.
The script only benchmarks an already-running endpoint; it never manages services.
EOF
}

[[ $# -ge 1 ]] || {
    usage
    exit 2
}

MODEL_ID="$1"
shift
SUITE="standard"
OUTPUT_DIR=""
EXECUTE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            SUITE="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --execute)
            EXECUTE=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done
[[ "$SUITE" == "smoke" || "$SUITE" == "standard" ]] || {
    echo "Unknown suite: $SUITE" >&2
    exit 2
}

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_FILE="${PROJECT_ROOT}/${MODEL_ID}/model.json"
LLAMA_ROOT="$(cd "${PROJECT_ROOT}/../llama.cpp" && pwd)"
SPEED_BENCH="${LLAMA_ROOT}/tools/server/bench/speed-bench/speed_bench.py"
SPEED_BENCH_WRAPPER="${PROJECT_ROOT}/bench/run-speed-bench.py"
BENCH_PYTHON="${PROJECT_ROOT}/.venv-bench/bin/python"
[[ -f "$MODEL_FILE" ]] || {
    echo "Unknown model or missing model.json: $MODEL_ID" >&2
    exit 2
}

MODEL_PORT="$(jq -er '.port' "$MODEL_FILE")"
MODEL_CONTEXT="$(jq -er '.llama.context // .context' "$MODEL_FILE")"
MODEL_PARALLEL="$(jq -er '.llama.parallel // .parallel // 1' "$MODEL_FILE")"
MODEL_EMBEDDING="$(jq -r '.embedding // false' "$MODEL_FILE")"
CONTEXT_PER_SLOT="$((MODEL_CONTEXT / MODEL_PARALLEL))"
BASE_URL="http://127.0.0.1:${MODEL_PORT}"

echo "Benchmark plan"
printf '  model: %s\n  endpoint: %s\n  suite: %s\n' \
    "$MODEL_ID" "$BASE_URL" "$SUITE"
printf '  configured context: %s total, %s per slot, %s slots\n' \
    "$MODEL_CONTEXT" "$CONTEXT_PER_SLOT" "$MODEL_PARALLEL"
echo "  capability probes: enabled"
if [[ "$MODEL_EMBEDDING" == "true" ]]; then
    echo "  SPEED-Bench: skipped (embedding suite is separate)"
elif [[ "$SUITE" == "smoke" ]]; then
    echo "  SPEED-Bench: qualitative coding,math,writing; OSL 128; limit 1"
else
    echo "  SPEED-Bench: qualitative all; OSL 512; limit 5"
    echo "  SPEED-Bench: throughput_1k and throughput_8k; OSL 512; limit 5"
    if ((CONTEXT_PER_SLOT >= 33280)); then
        echo "  SPEED-Bench: throughput_32k; OSL 512; limit 5"
    else
        echo "  SPEED-Bench: throughput_32k skipped (does not fit one slot)"
    fi
    if ((MODEL_PARALLEL > 1)); then
        echo "  concurrency pass: ${MODEL_PARALLEL} clients"
    fi
fi

if [[ "$EXECUTE" -ne 1 ]]; then
    echo "Dry run only. Add --execute after the live service baseline is approved."
    exit 0
fi

for command_name in curl jq nvidia-smi; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "Missing required command: $command_name" >&2
        exit 1
    }
done
[[ -x "$BENCH_PYTHON" ]] || {
    echo "Missing benchmark environment. Run ./bench/setup.sh first." >&2
    exit 1
}
[[ -f "$SPEED_BENCH" ]] || {
    echo "Missing upstream SPEED-Bench client: $SPEED_BENCH" >&2
    exit 1
}

curl --max-time 5 -fsS "${BASE_URL}/health" >/dev/null
if [[ -z "$OUTPUT_DIR" ]]; then
    RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
    OUTPUT_DIR="${PROJECT_ROOT}/bench-results/${RUN_STAMP}-${MODEL_ID}-${SUITE}"
elif [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}"
fi
[[ ! -e "$OUTPUT_DIR" ]] || {
    echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
    exit 1
}
mkdir -p "$OUTPUT_DIR"

"${PROJECT_ROOT}/bench/capture-metadata.sh" "$MODEL_ID" "$OUTPUT_DIR"
LIVE_PARALLEL="$(jq -er '.total_slots' "${OUTPUT_DIR}/props.json")"
LIVE_CONTEXT_PER_SLOT="$(jq -er '.default_generation_settings.n_ctx' \
    "${OUTPUT_DIR}/props.json")"
if [[ "$LIVE_PARALLEL" != "$MODEL_PARALLEL" \
    || "$LIVE_CONTEXT_PER_SLOT" != "$CONTEXT_PER_SLOT" ]]; then
    printf 'Live runtime overrides configured dimensions: context_per_slot=%s slots=%s\n' \
        "$LIVE_CONTEXT_PER_SLOT" "$LIVE_PARALLEL"
fi
MODEL_PARALLEL="$LIVE_PARALLEL"
CONTEXT_PER_SLOT="$LIVE_CONTEXT_PER_SLOT"
DATASET_REVISION="$("$BENCH_PYTHON" - <<'PY'
from huggingface_hub import HfApi

print(HfApi().dataset_info("nvidia/SPEED-Bench").sha)
PY
)"
printf '%s\n' "$DATASET_REVISION" \
    >"${OUTPUT_DIR}/speed-bench-dataset-revision.txt"
INITIAL_POWER_LIMIT="$(jq -er '.system.gpu_power_limit_w' "${OUTPUT_DIR}/manifest.json")"
INITIAL_SERVER_PID="$(jq -er '.process.pid' "${OUTPUT_DIR}/manifest.json")"
BENCH_GPU_UUID="$(jq -er '.system.gpu_uuid' "${OUTPUT_DIR}/manifest.json")"
nvidia-smi --id="$BENCH_GPU_UUID" \
    --query-gpu=name,uuid,driver_version,pstate,power.limit,power.draw,clocks.current.graphics,clocks.current.memory,temperature.gpu,memory.total,memory.used,memory.free \
    --format=csv >"${OUTPUT_DIR}/gpu-before-suite.csv"

TELEMETRY_PID=""
cleanup() {
    local status=$?
    trap - EXIT HUP INT TERM
    if [[ -n "$TELEMETRY_PID" ]] && kill -0 "$TELEMETRY_PID" 2>/dev/null; then
        kill -TERM "${TELEMETRY_PID:?}" 2>/dev/null || true
        wait "$TELEMETRY_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT HUP INT TERM

nvidia-smi --id="$BENCH_GPU_UUID" \
    --query-gpu=timestamp,power.draw,power.limit,utilization.gpu,clocks.current.graphics,clocks.current.memory,temperature.gpu,memory.used,memory.free \
    --format=csv -l 1 >"${OUTPUT_DIR}/gpu-telemetry.csv" &
TELEMETRY_PID=$!

"$BENCH_PYTHON" "${PROJECT_ROOT}/bench/capability-probe.py" \
    "$MODEL_FILE" "${OUTPUT_DIR}/canaries.json" --execute \
    | tee "${OUTPUT_DIR}/canaries.log"

run_speed_bench() {
    local label="$1"
    local started_ns ended_ns wall_seconds result_file temp_file
    shift
    result_file="${OUTPUT_DIR}/speed-${label}.json"
    temp_file="${result_file}.tmp"
    started_ns="$(date +%s%N)"
    "$BENCH_PYTHON" "$SPEED_BENCH_WRAPPER" \
        --source "$SPEED_BENCH" \
        --dataset-revision "$DATASET_REVISION" \
        --url "$BASE_URL" \
        --model "$MODEL_ID" \
        --extra-inputs '{"temperature":0,"ignore_eos":true}' \
        --timeout 900 \
        --output "$result_file" \
        "$@" | tee "${OUTPUT_DIR}/speed-${label}.log"
    ended_ns="$(date +%s%N)"
    wall_seconds="$(awk -v start="$started_ns" -v end="$ended_ns" \
        'BEGIN {printf "%.6f", (end-start)/1000000000}')"
    jq --argjson wall_seconds "$wall_seconds" '
        ([.results[] | select(.ok) | .completion_tokens] | add // 0) as $tokens
        | . + {
            suite_metrics: {
                wall_seconds: $wall_seconds,
                completion_tokens: $tokens,
                aggregate_completion_tokens_per_wall_second:
                    (if $wall_seconds > 0 then $tokens / $wall_seconds else null end)
            }
        }
    ' "$result_file" >"$temp_file"
    mv "$temp_file" "$result_file"
}

if [[ "$MODEL_EMBEDDING" != "true" ]]; then
    if [[ "$SUITE" == "smoke" ]]; then
        run_speed_bench smoke \
            --bench qualitative --category coding,math,writing \
            --osl 128 --limit 1 --concurrency 1
    else
        run_speed_bench qualitative \
            --bench qualitative --category all \
            --osl 512 --limit 5 --concurrency 1
        run_speed_bench throughput-1k \
            --bench throughput_1k --category all \
            --osl 512 --limit 5 --concurrency 1
        run_speed_bench throughput-8k \
            --bench throughput_8k --category all \
            --osl 512 --limit 5 --concurrency 1
        if ((CONTEXT_PER_SLOT >= 33280)); then
            run_speed_bench throughput-32k \
                --bench throughput_32k --category all \
                --osl 512 --limit 5 --concurrency 1
        fi
        if ((MODEL_PARALLEL > 1)); then
            run_speed_bench concurrency \
                --bench qualitative --category coding,writing \
                --osl 512 --limit "$MODEL_PARALLEL" \
                --concurrency "$MODEL_PARALLEL"
        fi
    fi
fi

if [[ -n "$TELEMETRY_PID" ]] && kill -0 "$TELEMETRY_PID" 2>/dev/null; then
    kill -TERM "${TELEMETRY_PID:?}"
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=""
fi
nvidia-smi --id="$BENCH_GPU_UUID" \
    --query-gpu=name,uuid,driver_version,pstate,power.limit,power.draw,clocks.current.graphics,clocks.current.memory,temperature.gpu,memory.total,memory.used,memory.free \
    --format=csv >"${OUTPUT_DIR}/gpu-after.csv"
FINAL_POWER_LIMIT="$(nvidia-smi --id="$BENCH_GPU_UUID" --query-gpu=power.limit --format=csv,noheader,nounits | awk '{printf "%.0f", $1}')"
if [[ "$FINAL_POWER_LIMIT" != "$INITIAL_POWER_LIMIT" ]]; then
    echo "Power limit changed during run: ${INITIAL_POWER_LIMIT} W -> ${FINAL_POWER_LIMIT} W" >&2
    exit 1
fi
if ! awk -F, -v expected="$INITIAL_POWER_LIMIT" '
    NR > 1 {
        value=$3
        gsub(/[^0-9.]/, "", value)
        if ((value + 0) != (expected + 0)) exit 1
    }
' "${OUTPUT_DIR}/gpu-telemetry.csv"; then
    echo "Power limit changed during telemetry sampling." >&2
    exit 1
fi
FINAL_SERVER_PID="$(ss -H -ltnp "sport = :${MODEL_PORT}" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n 1)"
if [[ "$FINAL_SERVER_PID" != "$INITIAL_SERVER_PID" ]]; then
    echo "Server PID changed during run: $INITIAL_SERVER_PID -> ${FINAL_SERVER_PID:-none}" >&2
    exit 1
fi

echo "Benchmark run complete: $OUTPUT_DIR"
