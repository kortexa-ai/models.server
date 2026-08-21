#!/usr/bin/env bash
# Legacy direct-launch harness archived with the pre-reset benchmark log.
set -euo pipefail

export PATH=/home/francip/bin:/home/francip/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin

if [[ "$(hostname)" != "smarty" ]]; then
    echo "This benchmark must run on smarty." >&2
    exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

server_pid=""
server_log=""
qwen36_stopped=0
log_paths=()

health_code() {
    curl --max-time 3 -s -o /dev/null -w "%{http_code}" \
        "http://127.0.0.1:$1/health" || true
}

stop_qwen38() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "${server_pid:?}"
        for _ in $(seq 1 30); do
            if ! kill -0 "$server_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            echo "Qwen 3.8 PID $server_pid did not stop after SIGTERM; sending SIGKILL." >&2
            kill -KILL "${server_pid:?}"
        fi
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=""

    for _ in $(seq 1 20); do
        if [[ "$(health_code 2053)" != "200" ]]; then
            return 0
        fi
        sleep 1
    done
    echo "Qwen 3.8 port 2053 did not close." >&2
    return 1
}

restore_baseline() {
    restore_rc=$?
    trap - EXIT HUP INT TERM
    set +e

    stop_qwen38

    if [[ "$qwen36_stopped" -eq 1 ]]; then
        echo "RESTORE starting Qwen 3.6"
        ktxsvc start models/qwen-3.6-27b
        qwen36_ready=0
        for _ in $(seq 1 120); do
            if [[ "$(health_code 2032)" == "200" ]]; then
                qwen36_ready=1
                qwen36_stopped=0
                break
            fi
            sleep 5
        done
        if [[ "$qwen36_ready" -ne 1 ]]; then
            echo "ERROR Qwen 3.6 did not become healthy during restoration." >&2
            restore_rc=97
        fi
    fi

    echo "FINAL_HEALTH"
    for port in 4004 4001 2032 2039 4002 4003 2040; do
        code="$(health_code "$port")"
        printf "HEALTH port=%s code=%s\n" "$port" "$code"
        if [[ "$code" != "200" ]]; then
            restore_rc=97
        fi
    done
    if [[ "$(health_code 2053)" == "200" ]]; then
        echo "ERROR Qwen 3.8 still answers on port 2053." >&2
        restore_rc=97
    fi
    nvidia-smi \
        --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu \
        --format=csv,noheader
    git status --short
    exit "$restore_rc"
}
trap restore_baseline EXIT HUP INT TERM

for port in 4004 4001 2032 2039 4002 4003 2040; do
    if [[ "$(health_code "$port")" != "200" ]]; then
        echo "ERROR baseline endpoint $port is not healthy." >&2
        exit 1
    fi
done

start_qwen38() {
    local depth="$1"
    stop_qwen38
    server_log="$(mktemp --suffix=.log "/tmp/qwen38-depth${depth}.XXXXXX")"
    log_paths+=("$server_log")

    local -a command=(
        /home/francip/bin/llama-server
        -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
        --alias qwen-3.8-27b
        --host 127.0.0.1
        --port 2053
        --jinja
        -c 262144
        -ngl 99
        --threads -1
        --parallel 1
        --no-context-shift
        --temp 0.6
        --top-k 20
        --top-p 0.95
        --no-mmap
        --flash-attn on
        --cache-type-k q8_0
        --cache-type-v q8_0
    )
    if ((depth > 0)); then
        command+=(--spec-type draft-mtp --spec-draft-n-max "$depth")
    fi

    local start_ms
    start_ms="$(date +%s%3N)"
    "${command[@]}" >"$server_log" 2>&1 &
    server_pid=$!

    for _ in $(seq 1 120); do
        if [[ "$(health_code 2053)" == "200" ]]; then
            local end_ms free_mib process_mib props
            end_ms="$(date +%s%3N)"
            free_mib="$(nvidia-smi \
                --query-gpu=memory.free \
                --format=csv,noheader,nounits | tr -d ' ')"
            if ((free_mib < 10240)); then
                echo "ERROR Qwen 3.8 crossed the 10 GiB free-VRAM floor." >&2
                return 1
            fi
            process_mib="$(nvidia-smi \
                --query-compute-apps=pid,used_memory \
                --format=csv,noheader,nounits | \
                awk -F, -v wanted="$server_pid" \
                    '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1 == wanted) print $2}')"
            props="$(curl --max-time 10 -fsS http://127.0.0.1:2053/props)"
            echo "$props" | jq -e \
                '.total_slots == 1 and
                 .default_generation_settings.n_ctx == 262144 and
                 .modalities.vision == true' >/dev/null
            printf "SERVER depth=%s load_ms=%s pid=%s process_mib=%s free_mib=%s log=%s\n" \
                "$depth" "$((end_ms - start_ms))" "$server_pid" \
                "${process_mib:-unknown}" "$free_mib" "$server_log"
            return 0
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "ERROR Qwen 3.8 depth $depth exited during startup." >&2
            tail -n 100 "$server_log" >&2
            return 1
        fi
        local free_mib
        free_mib="$(nvidia-smi \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits | tr -d ' ')"
        if ((free_mib < 10240)); then
            echo "ERROR Qwen 3.8 crossed the 10 GiB free-VRAM floor while loading." >&2
            return 1
        fi
        sleep 2
    done

    echo "ERROR Qwen 3.8 depth $depth did not become healthy." >&2
    tail -n 100 "$server_log" >&2
    return 1
}

bench_tps=""
bench_draft_n=0
bench_accepted=0
bench_prompt_n=0
bench_prompt_tps=0
bench_cached=0

run_generation() {
    local port="$1"
    local model="$2"
    local label="$3"
    local workload="$4"
    local run_number="$5"
    local max_tokens="$6"
    local prompt="$7"
    local payload response error completion finish accept_pct

    payload="$(printf '%s' "$prompt" | jq -Rsc \
        --arg model "$model" \
        --argjson max_tokens "$max_tokens" \
        '. as $prompt | {
            model: $model,
            messages: [{role: "user", content: $prompt}],
            temperature: 0,
            max_tokens: $max_tokens,
            ignore_eos: true,
            chat_template_kwargs: {enable_thinking: false}
        }')"
    response="$(printf '%s' "$payload" | curl --max-time 900 -fsS \
        "http://127.0.0.1:${port}/v1/chat/completions" \
        -H 'Content-Type: application/json' --data-binary @-)"
    error="$(echo "$response" | jq -r '.error.message // empty')"
    if [[ -n "$error" ]]; then
        echo "ERROR generation $label $workload $run_number: $error" >&2
        return 1
    fi

    completion="$(echo "$response" | jq -r '.usage.completion_tokens // 0')"
    finish="$(echo "$response" | jq -r '.choices[0].finish_reason // "unknown"')"
    if [[ "$completion" -ne "$max_tokens" || "$finish" != "length" ]]; then
        echo "ERROR forced generation returned completion=$completion finish=$finish" >&2
        return 1
    fi

    bench_tps="$(echo "$response" | jq -r '.timings.predicted_per_second')"
    bench_draft_n="$(echo "$response" | jq -r '.timings.draft_n // 0')"
    bench_accepted="$(echo "$response" | jq -r '.timings.draft_n_accepted // 0')"
    bench_prompt_n="$(echo "$response" | jq -r '.timings.prompt_n // 0')"
    bench_prompt_tps="$(echo "$response" | jq -r '.timings.prompt_per_second // 0')"
    bench_cached="$(echo "$response" | jq -r \
        '.usage.prompt_tokens_details.cached_tokens // .timings.cache_n // 0')"
    accept_pct="$(awk \
        -v drafted="$bench_draft_n" -v accepted="$bench_accepted" \
        'BEGIN {
            if (drafted > 0) printf "%.2f", 100*accepted/drafted;
            else printf "0.00"
        }')"

    printf "RESULT label=%s workload=%s run=%s completion=%s decode_tps=%s prompt_n=%s prompt_tps=%s cached=%s draft_n=%s accepted=%s accept_pct=%s\n" \
        "$label" "$workload" "$run_number" "$completion" "$bench_tps" \
        "$bench_prompt_n" "$bench_prompt_tps" "$bench_cached" \
        "$bench_draft_n" "$bench_accepted" "$accept_pct"
}

summarize_values() {
    local label="$1"
    shift
    printf "%s\n" "$@" | awk -v label="$label" '
        NR == 1 {min=$1; max=$1}
        {sum+=$1; if ($1<min) min=$1; if ($1>max) max=$1}
        END {
            printf "SUMMARY label=%s n=%d mean=%.2f min=%.2f max=%.2f\n",
                label, NR, sum/NR, min, max
        }
    '
}

prose_prompt="Write a continuous detailed essay about designing reliable local AI inference infrastructure. Use complete paragraphs, no headings, no conclusion, and continue until stopped."
code_prompt="Write a single Python module implementing a thread-safe LRU cache with TTL, tests, type hints, and detailed docstrings. Output code only and continue until stopped."

echo "STOPPING only Qwen 3.6"
ktxsvc stop models/qwen-3.6-27b
qwen36_stopped=1
free_mib="$(nvidia-smi \
    --query-gpu=memory.free \
    --format=csv,noheader,nounits | tr -d ' ')"
printf "GPU_AFTER_STOP free_mib=%s\n" "$free_mib"
if ((free_mib < 50000)); then
    echo "ERROR less than 50 GiB is free after Qwen 3.6 stop." >&2
    exit 1
fi

best_depth=""
best_score=0
for depth in 0 1 2 3 4; do
    echo "SWEEP_START depth=$depth"
    if ! start_qwen38 "$depth"; then
        echo "SWEEP_FAILED depth=$depth"
        stop_qwen38
        continue
    fi
    if ! run_generation 2053 qwen-3.8-27b \
        "qwen38-depth${depth}-sweep" prose 1 600 "$prose_prompt"; then
        echo "SWEEP_FAILED depth=$depth workload=prose"
        stop_qwen38
        continue
    fi
    prose_tps="$bench_tps"
    if ! run_generation 2053 qwen-3.8-27b \
        "qwen38-depth${depth}-sweep" code 1 600 "$code_prompt"; then
        echo "SWEEP_FAILED depth=$depth workload=code"
        stop_qwen38
        continue
    fi
    code_tps="$bench_tps"
    score="$(awk -v prose="$prose_tps" -v code="$code_tps" \
        'BEGIN {printf "%.6f", (prose+code)/2}')"
    printf "SWEEP_SUMMARY depth=%s prose_tps=%s code_tps=%s mixed_mean=%s\n" \
        "$depth" "$prose_tps" "$code_tps" "$score"
    if awk -v score="$score" -v best="$best_score" \
        'BEGIN {exit !(score > best)}'; then
        best_depth="$depth"
        best_score="$score"
    fi
    stop_qwen38
done

if [[ -z "$best_depth" ]]; then
    echo "ERROR every Qwen 3.8 sweep depth failed." >&2
    exit 1
fi
printf "BEST_DEPTH depth=%s mixed_mean=%s\n" "$best_depth" "$best_score"

start_qwen38 "$best_depth"
prose_values=()
code_values=()
total_draft=0
total_accepted=0

for run_number in 1 2 3; do
    run_generation 2053 qwen-3.8-27b \
        "qwen38-depth${best_depth}-repeat" prose "$run_number" 600 "$prose_prompt"
    prose_values+=("$bench_tps")
    total_draft=$((total_draft + bench_draft_n))
    total_accepted=$((total_accepted + bench_accepted))
done
for run_number in 1 2 3; do
    run_generation 2053 qwen-3.8-27b \
        "qwen38-depth${best_depth}-repeat" code "$run_number" 600 "$code_prompt"
    code_values+=("$bench_tps")
    total_draft=$((total_draft + bench_draft_n))
    total_accepted=$((total_accepted + bench_accepted))
done
summarize_values "qwen38-depth${best_depth}-prose" "${prose_values[@]}"
summarize_values "qwen38-depth${best_depth}-code" "${code_values[@]}"
awk -v drafted="$total_draft" -v accepted="$total_accepted" '
    BEGIN {
        pct = drafted > 0 ? 100*accepted/drafted : 0
        printf "ACCEPTANCE label=qwen38-best drafted=%d accepted=%d pct=%.2f\n",
            drafted, accepted, pct
    }
'

long_prompt="$(python3 - <<'PY'
print(
    ("alpha beta gamma delta epsilon zeta eta theta " * 4096)
    + "\nSummarize the repeated sequence in one sentence."
)
PY
)"
run_generation 2053 qwen-3.8-27b \
    "qwen38-depth${best_depth}" long-cold 1 64 "$long_prompt"
run_generation 2053 qwen-3.8-27b \
    "qwen38-depth${best_depth}" long-warm 2 64 "$long_prompt"

text_payload="$(jq -nc '{
    model: "qwen-3.8-27b",
    messages: [{
        role: "user",
        content: "Reply with exactly QWEN38_OK and nothing else."
    }],
    temperature: 0,
    max_tokens: 64,
    chat_template_kwargs: {enable_thinking: false}
}')"
text_response="$(curl --max-time 300 -fsS \
    http://127.0.0.1:2053/v1/chat/completions \
    -H 'Content-Type: application/json' -d "$text_payload")"
text_content="$(echo "$text_response" | jq -r \
    '.choices[0].message.content // empty')"
printf "CANARY type=text content=%q\n" "$text_content"
if [[ "$text_content" != *QWEN38_OK* ]]; then
    echo "ERROR text canary failed." >&2
    exit 1
fi

tool_payload="$(jq -nc '{
    model: "qwen-3.8-27b",
    messages: [{
        role: "user",
        content: "Call lookup_weather exactly once for Paris. Do not answer directly."
    }],
    tools: [{
        type: "function",
        function: {
            name: "lookup_weather",
            description: "Look up weather for a city.",
            parameters: {
                type: "object",
                properties: {city: {type: "string"}},
                required: ["city"]
            }
        }
    }],
    tool_choice: "required",
    temperature: 0,
    max_tokens: 128,
    chat_template_kwargs: {enable_thinking: false}
}')"
tool_response="$(curl --max-time 300 -fsS \
    http://127.0.0.1:2053/v1/chat/completions \
    -H 'Content-Type: application/json' -d "$tool_payload")"
echo "$tool_response" | jq -c \
    '{tool_calls: .choices[0].message.tool_calls,
      content: .choices[0].message.content}'
tool_name="$(echo "$tool_response" | jq -r \
    '.choices[0].message.tool_calls[0].function.name // empty')"
tool_city="$(echo "$tool_response" | jq -r '
    .choices[0].message.tool_calls[0].function.arguments |
    if type == "string" then (fromjson | .city) else .city end // empty
')"
printf "CANARY type=tool name=%s city=%s\n" "$tool_name" "$tool_city"
if [[ "$tool_name" != "lookup_weather" || "$tool_city" != "Paris" ]]; then
    echo "ERROR tool canary failed." >&2
    exit 1
fi

image_b64="$(python3 - <<'PY'
import base64
import struct
import zlib

width = height = 512
raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))


def chunk(kind, data):
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


png = (
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(raw))
    + chunk(b"IEND", b"")
)
print(base64.b64encode(png).decode())
PY
)"
image_payload="$(jq -nc \
    --arg image_url "data:image/png;base64,${image_b64}" \
    '{
        model: "qwen-3.8-27b",
        messages: [{
            role: "user",
            content: [
                {type: "image_url", image_url: {url: $image_url}},
                {
                    type: "text",
                    text: "What is the dominant color of this square? Reply with one color word."
                }
            ]
        }],
        temperature: 0,
        max_tokens: 64,
        chat_template_kwargs: {enable_thinking: false}
    }')"
image_response="$(curl --max-time 300 -fsS \
    http://127.0.0.1:2053/v1/chat/completions \
    -H 'Content-Type: application/json' -d "$image_payload")"
image_content="$(echo "$image_response" | jq -r \
    '.choices[0].message.content // empty')"
image_prompt_n="$(echo "$image_response" | jq -r \
    '.timings.prompt_n // 0')"
image_prompt_tps="$(echo "$image_response" | jq -r \
    '.timings.prompt_per_second // 0')"
printf "CANARY type=image content=%q prompt_n=%s prompt_tps=%s\n" \
    "$image_content" "$image_prompt_n" "$image_prompt_tps"
if [[ "${image_content,,}" != *red* ]]; then
    echo "ERROR image canary failed." >&2
    exit 1
fi

echo "QWEN38_GPU"
nvidia-smi \
    --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu \
    --format=csv,noheader
nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv
echo "QWEN38_SPEC_LOG"
rg -i 'spec|draft|accept' "$server_log" | tail -n 120 || true

stop_qwen38
echo "RESTORE_FOR_MATCHED_COMPARISON"
ktxsvc start models/qwen-3.6-27b
for _ in $(seq 1 120); do
    if [[ "$(health_code 2032)" == "200" ]]; then
        qwen36_stopped=0
        break
    fi
    sleep 5
done
if [[ "$qwen36_stopped" -ne 0 ]]; then
    echo "ERROR Qwen 3.6 did not become healthy for matched comparison." >&2
    exit 1
fi

qwen36_prose_values=()
qwen36_code_values=()
qwen36_total_draft=0
qwen36_total_accepted=0
for run_number in 1 2 3; do
    run_generation 2032 qwen-3.6-27b \
        qwen36-depth3-repeat prose "$run_number" 600 "$prose_prompt"
    qwen36_prose_values+=("$bench_tps")
    qwen36_total_draft=$((qwen36_total_draft + bench_draft_n))
    qwen36_total_accepted=$((qwen36_total_accepted + bench_accepted))
done
for run_number in 1 2 3; do
    run_generation 2032 qwen-3.6-27b \
        qwen36-depth3-repeat code "$run_number" 600 "$code_prompt"
    qwen36_code_values+=("$bench_tps")
    qwen36_total_draft=$((qwen36_total_draft + bench_draft_n))
    qwen36_total_accepted=$((qwen36_total_accepted + bench_accepted))
done
summarize_values qwen36-depth3-prose "${qwen36_prose_values[@]}"
summarize_values qwen36-depth3-code "${qwen36_code_values[@]}"
awk -v drafted="$qwen36_total_draft" -v accepted="$qwen36_total_accepted" '
    BEGIN {
        pct = drafted > 0 ? 100*accepted/drafted : 0
        printf "ACCEPTANCE label=qwen36-matched drafted=%d accepted=%d pct=%.2f\n",
            drafted, accepted, pct
    }
'
run_generation 2032 qwen-3.6-27b \
    qwen36-depth3 long-cold 1 64 "$long_prompt"
run_generation 2032 qwen-3.6-27b \
    qwen36-depth3 long-warm 2 64 "$long_prompt"

echo "PROTECTED_HEALTH"
for port in 4004 4001 2032 2039 4002 4003 2040; do
    code="$(health_code "$port")"
    printf "HEALTH port=%s code=%s\n" "$port" "$code"
    if [[ "$code" != "200" ]]; then
        echo "ERROR protected endpoint $port is unhealthy." >&2
        exit 1
    fi
done

echo "BENCHMARK_COMPLETE best_depth=$best_depth"
for log_path in "${log_paths[@]}"; do
    if [[ "$log_path" == /tmp/qwen38-depth*.log && -f "$log_path" ]]; then
        unlink "${log_path:?}"
    fi
done
