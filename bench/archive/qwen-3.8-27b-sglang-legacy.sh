#!/usr/bin/env bash
# Legacy direct-launch harness archived with the pre-reset benchmark log.
set -euo pipefail

export PATH=/home/francip/bin:/home/francip/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin

if [[ "$(hostname)" != "smarty" ]]; then
    echo "This benchmark must run on smarty." >&2
    exit 1
fi

IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:qwen38-27b}"
MODEL_DIR="${SGLANG_MODEL_DIR:-$HOME/data/models/qwen-3.8-27b-nvfp4}"
DRAFT_DIR="${SGLANG_DRAFT_DIR:-$HOME/data/models/qwen-3.8-27b-dspark}"
CACHE_DIR="${SGLANG_CACHE_DIR:-$HOME/.cache/sglang-qwen38-docker}"
RUNTIME="${SGLANG_RUNTIME:-docker}"
MODES="${SGLANG_MODES:-none eagle eagle-prod dspark dspark-bf16 dspark-prod}"
SGLANG_BIN="${SGLANG_BIN:-$HOME/src/models.server/.venv-sglang/bin/sglang}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${SGLANG_BIN%/sglang}/python}"
PORT="${PORT:-2053}"
CONTAINER=models-server-qwen38-sglang-bench
RESULTS_FILE="$(mktemp /tmp/qwen38-sglang-results.XXXXXX)"
RESPONSE_FILE="$(mktemp /tmp/qwen38-sglang-response.XXXXXX)"
server_log=""
server_pid=""
bench_tps=""

health_code() {
    curl --max-time 3 -s -o /dev/null -w '%{http_code}' \
        "http://127.0.0.1:${PORT}/health" || true
}

remove_server() {
    local state
    if [[ "$RUNTIME" == "docker" ]]; then
        state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || true)"
        if [[ "$state" == "running" ]]; then
            docker stop --time 45 "$CONTAINER" >/dev/null 2>&1 || true
        fi
        if [[ -n "$state" ]]; then
            docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        fi
        return
    fi

    if [[ -n "$server_pid" ]] && kill -0 -- "-$server_pid" 2>/dev/null; then
        kill -TERM -- "-$server_pid"
        for _ in $(seq 1 45); do
            kill -0 -- "-$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 -- "-$server_pid" 2>/dev/null; then
            kill -KILL -- "-$server_pid"
        fi
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=""
}

cleanup() {
    local command_status=$?
    trap - EXIT HUP INT TERM
    set +e
    if [[ "$RUNTIME" == "docker" && -n "$server_log" ]] \
        && docker inspect "$CONTAINER" >/dev/null 2>&1; then
        docker logs "$CONTAINER" >"$server_log" 2>&1 || true
    fi
    remove_server
    case "$RESULTS_FILE" in /tmp/qwen38-sglang-results.*) rm -f -- "$RESULTS_FILE" ;; esac
    case "$RESPONSE_FILE" in /tmp/qwen38-sglang-response.*) rm -f -- "$RESPONSE_FILE" ;; esac
    exit "$command_status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

for command in curl jq nvidia-smi; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "Missing required command: $command" >&2
        exit 1
    }
done
case "$RUNTIME" in
    docker)
        command -v docker >/dev/null 2>&1 || {
            echo "Missing required command: docker" >&2
            exit 1
        }
        docker image inspect "$IMAGE" >/dev/null
        ;;
    host)
        command -v setsid >/dev/null 2>&1 || {
            echo "Missing required command: setsid" >&2
            exit 1
        }
        [[ -x "$SGLANG_BIN" && -x "$SGLANG_PYTHON" ]] || {
            echo "Missing host SGLang runtime: $SGLANG_BIN" >&2
            exit 1
        }
        ;;
    *)
        echo "Unknown SGLang runtime: $RUNTIME" >&2
        exit 2
        ;;
esac
for path in \
    "$MODEL_DIR/config.json" \
    "$MODEL_DIR/model.safetensors.index.json" \
    "$DRAFT_DIR/config.json" \
    "$DRAFT_DIR/model.safetensors"
do
    [[ -f "$path" ]] || { echo "Missing model file: $path" >&2; exit 1; }
done
mkdir -p "$CACHE_DIR"

if [[ "$(health_code)" == "200" ]]; then
    echo "Port $PORT already has a healthy server; refusing to create a duplicate." >&2
    exit 1
fi
initial_free_mib="$(nvidia-smi --query-gpu=memory.free \
    --format=csv,noheader,nounits | tr -d ' ')"
if ((initial_free_mib < 90000)); then
    echo "Only ${initial_free_mib} MiB is free; run this benchmark inside the Smarty GPU service block." >&2
    exit 1
fi

if [[ "$RUNTIME" == "docker" ]]; then
    image_id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
    image_digests="$(docker image inspect -f '{{join .RepoDigests ","}}' "$IMAGE")"
    printf 'RUNTIME type=docker ref=%s id=%s digests=%s\n' \
        "$IMAGE" "$image_id" "$image_digests"
else
    runtime_commit="$(git -C "$HOME/src/models.server/.engines/sglang" rev-parse HEAD 2>/dev/null || true)"
    printf 'RUNTIME type=host bin=%s commit=%s\n' "$SGLANG_BIN" "${runtime_commit:-unknown}"
fi

start_sglang() {
    local mode="$1"
    local memory_fraction=0.85
    local ssm_dtype=float32
    local model_path draft_path
    local -a serve_args sizing_args spec_args

    if [[ "$RUNTIME" == "docker" ]]; then
        model_path=/models/target
        draft_path=/models/draft
    else
        model_path="$MODEL_DIR"
        draft_path="$DRAFT_DIR"
    fi

    sizing_args=()
    spec_args=()
    case "$mode" in
        none)
            sizing_args=(--mamba-full-memory-ratio 4.59)
            ;;
        eagle)
            sizing_args=(--mamba-full-memory-ratio 4.59)
            spec_args=(
                --speculative-algorithm EAGLE
                --speculative-num-steps 3
                --speculative-eagle-topk 1
                --speculative-num-draft-tokens 4
                --enable-linear-replayssm-spec
            )
            ;;
        eagle-prod)
            memory_fraction=0.45
            sizing_args=(--max-mamba-cache-size 5 --max-running-requests 1)
            spec_args=(
                --speculative-algorithm EAGLE
                --speculative-num-steps 3
                --speculative-eagle-topk 1
                --speculative-num-draft-tokens 4
                --enable-linear-replayssm-spec
            )
            ;;
        dspark)
            sizing_args=(--mamba-full-memory-ratio 11.93)
            spec_args=(
                --speculative-algorithm DSPARK
                --speculative-draft-model-path "$draft_path"
                --speculative-draft-attention-backend flashinfer
            )
            ;;
        dspark-bf16)
            ssm_dtype=bfloat16
            sizing_args=(--mamba-full-memory-ratio 6.08)
            spec_args=(
                --speculative-algorithm DSPARK
                --speculative-draft-model-path "$draft_path"
                --speculative-draft-attention-backend flashinfer
            )
            ;;
        dspark-prod)
            memory_fraction=0.45
            sizing_args=(--max-mamba-cache-size 5 --max-running-requests 1)
            spec_args=(
                --speculative-algorithm DSPARK
                --speculative-draft-model-path "$draft_path"
                --speculative-draft-attention-backend flashinfer
            )
            ;;
        *)
            echo "Unknown SGLang mode: $mode" >&2
            return 2
            ;;
    esac

    remove_server
    server_log="$(mktemp "/tmp/qwen38-sglang-${mode}.XXXXXX.log")"
    local start_ms
    start_ms="$(date +%s%3N)"
    serve_args=(
        serve
        --trust-remote-code
        --model-path "$model_path"
        --served-model-name qwen-3.8-27b
        --context-length 262144
        --kv-cache-dtype fp8_e4m3
        --mem-fraction-static "$memory_fraction"
        --attention-backend flashinfer
        --chunked-prefill-size 2048
        --reasoning-parser qwen3
        --tool-call-parser qwen3_coder
        --default-chat-template-kwargs '{"reasoning_effort":"medium"}'
        --mamba-radix-cache-strategy extra_buffer
        --mamba-ssm-dtype "$ssm_dtype"
        --host 0.0.0.0
        --port "$PORT"
        "${sizing_args[@]}"
        "${spec_args[@]}"
    )
    if [[ "$RUNTIME" == "docker" ]]; then
        docker run -d \
            --pull never \
            --name "$CONTAINER" \
            --gpus all \
            --shm-size 32g \
            --ipc=host \
            -p "127.0.0.1:${PORT}:${PORT}" \
            -v "$MODEL_DIR:/models/target:ro" \
            -v "$DRAFT_DIR:/models/draft:ro" \
            -v "$CACHE_DIR:/root/.cache" \
            --env CUDA_CACHE_PATH=/root/.cache/cuda \
            --env HF_HUB_OFFLINE=1 \
            --env TRITON_CACHE_DIR=/root/.cache/triton \
            --env TRANSFORMERS_OFFLINE=1 \
            "$IMAGE" sglang "${serve_args[@]}" >/dev/null
    else
        CUDA_CACHE_PATH="$CACHE_DIR/cuda" \
        HF_HUB_OFFLINE=1 \
        TRITON_CACHE_DIR="$CACHE_DIR/triton" \
        TRANSFORMERS_OFFLINE=1 \
            setsid "$SGLANG_BIN" "${serve_args[@]}" >"$server_log" 2>&1 &
        server_pid=$!
    fi

    local free_mib state end_ms
    for _ in $(seq 1 450); do
        if [[ "$(health_code)" == "200" ]]; then
            end_ms="$(date +%s%3N)"
            free_mib="$(nvidia-smi --query-gpu=memory.free \
                --format=csv,noheader,nounits | tr -d ' ')"
            if ((free_mib < 10240)); then
                echo "SGLang crossed the 10 GiB free-VRAM floor after startup." >&2
                return 1
            fi
            printf 'SERVER mode=%s load_ms=%s free_mib=%s log=%s\n' \
                "$mode" "$((end_ms - start_ms))" "$free_mib" "$server_log"
            if [[ "$RUNTIME" == "docker" ]]; then
                docker exec -i "$CONTAINER" python3 - <<'PY'
import importlib.metadata as metadata

for package in ("sglang", "torch", "flashinfer-python"):
    try:
        print(f"VERSION package={package} version={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"VERSION package={package} version=unknown")
PY
            else
                "$SGLANG_PYTHON" - <<'PY'
import importlib.metadata as metadata

for package in ("sglang", "torch", "flashinfer-python"):
    try:
        print(f"VERSION package={package} version={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"VERSION package={package} version=unknown")
PY
            fi
            return 0
        fi
        if [[ "$RUNTIME" == "docker" ]]; then
            state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || true)"
        elif [[ -n "$server_pid" ]] && kill -0 -- "-$server_pid" 2>/dev/null; then
            state=running
        else
            state=exited
        fi
        if [[ "$state" != "running" ]]; then
            echo "SGLang mode $mode exited during startup." >&2
            if [[ "$RUNTIME" == "docker" ]]; then
                docker logs --tail 200 "$CONTAINER" >&2 || true
            else
                tail -n 200 "$server_log" >&2 || true
            fi
            return 1
        fi
        free_mib="$(nvidia-smi --query-gpu=memory.free \
            --format=csv,noheader,nounits | tr -d ' ')"
        if ((free_mib < 10240)); then
            echo "SGLang crossed the 10 GiB free-VRAM floor while loading." >&2
            return 1
        fi
        sleep 2
    done

    echo "SGLang mode $mode did not become healthy." >&2
    if [[ "$RUNTIME" == "docker" ]]; then
        docker logs --tail 200 "$CONTAINER" >&2 || true
    else
        tail -n 200 "$server_log" >&2 || true
    fi
    return 1
}

stop_sglang() {
    if [[ "$RUNTIME" == "docker" && -n "$server_log" ]] \
        && docker inspect "$CONTAINER" >/dev/null 2>&1; then
        docker logs "$CONTAINER" >"$server_log" 2>&1 || true
    fi
    remove_server
    for _ in $(seq 1 30); do
        [[ "$(health_code)" != "200" ]] && return 0
        sleep 1
    done
    echo "Port $PORT did not close after stopping SGLang." >&2
    return 1
}

run_generation() {
    local mode="$1"
    local workload="$2"
    local run_number="$3"
    local max_tokens="$4"
    local prompt="$5"
    local payload elapsed completion finish error

    payload="$(printf '%s' "$prompt" | jq -Rsc \
        --argjson max_tokens "$max_tokens" \
        '{
            model: "qwen-3.8-27b",
            messages: [{role: "user", content: .}],
            temperature: 0,
            max_tokens: $max_tokens,
            ignore_eos: true,
            chat_template_kwargs: {enable_thinking: false}
        }')"
    if ! elapsed="$(printf '%s' "$payload" | curl --max-time 900 -fsS \
        -o "$RESPONSE_FILE" -w '%{time_total}' \
        "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' --data-binary @-)"; then
        jq -c . "$RESPONSE_FILE" >&2 || true
        return 1
    fi
    error="$(jq -r '.error.message // empty' "$RESPONSE_FILE")"
    [[ -z "$error" ]] || { echo "Generation error: $error" >&2; return 1; }
    completion="$(jq -r '.usage.completion_tokens // 0' "$RESPONSE_FILE")"
    finish="$(jq -r '.choices[0].finish_reason // "unknown"' "$RESPONSE_FILE")"
    if [[ "$completion" -ne "$max_tokens" || "$finish" != "length" ]]; then
        echo "Forced generation returned completion=$completion finish=$finish" >&2
        return 1
    fi
    bench_tps="$(awk -v tokens="$completion" -v seconds="$elapsed" \
        'BEGIN {printf "%.4f", tokens/seconds}')"
    printf 'RESULT mode=%s workload=%s run=%s completion=%s wall_seconds=%s wall_tps=%s\n' \
        "$mode" "$workload" "$run_number" "$completion" "$elapsed" "$bench_tps"
}

prose_prompt='Write a continuous detailed essay about designing reliable local AI inference infrastructure. Use complete paragraphs, no headings, no conclusion, and continue until stopped.'
code_prompt='Write a single Python module implementing a thread-safe LRU cache with TTL, tests, type hints, and detailed docstrings. Output code only and continue until stopped.'
warmup_prompt='Write a short paragraph about reliable inference.'

for mode in $MODES; do
    echo "SWEEP_START mode=$mode"
    if ! start_sglang "$mode"; then
        echo "SWEEP_FAILED mode=$mode stage=start"
        stop_sglang || true
        continue
    fi
    if ! run_generation "$mode" warmup 0 64 "$warmup_prompt"; then
        echo "SWEEP_FAILED mode=$mode stage=warmup"
        stop_sglang
        continue
    fi
    if ! run_generation "$mode" prose 1 600 "$prose_prompt"; then
        echo "SWEEP_FAILED mode=$mode stage=prose"
        stop_sglang
        continue
    fi
    prose_tps="$bench_tps"
    if ! run_generation "$mode" code 1 600 "$code_prompt"; then
        echo "SWEEP_FAILED mode=$mode stage=code"
        stop_sglang
        continue
    fi
    code_tps="$bench_tps"
    printf '%s\tprose\t%s\n' "$mode" "$prose_tps" >> "$RESULTS_FILE"
    printf '%s\tcode\t%s\n' "$mode" "$code_tps" >> "$RESULTS_FILE"
    curl --max-time 5 -fsS "http://127.0.0.1:${PORT}/metrics" 2>/dev/null \
        | grep -E '^sglang:spec_(accept_length|accept_rate|block_accept_length)' \
        | tail -n 12 || true
    stop_sglang
done

[[ -s "$RESULTS_FILE" ]] || { echo "Every SGLang sweep mode failed." >&2; exit 1; }
awk -F '\t' '
    {sum[$1]+=$3; n[$1]++}
    END {for (mode in sum) printf "SWEEP_SUMMARY mode=%s mixed_mean=%.4f\n", mode, sum[mode]/n[mode]}
' "$RESULTS_FILE" | sort
best_mode="$(awk -F '\t' '
    {sum[$1]+=$3; n[$1]++}
    END {for (mode in sum) printf "%s\t%.8f\n", mode, sum[mode]/n[mode]}
' "$RESULTS_FILE" | sort -t $'\t' -k2,2nr | head -n 1 | cut -f 1)"
echo "BEST_MODE mode=$best_mode"

start_sglang "$best_mode"
run_generation "$best_mode" warmup 0 64 "$warmup_prompt"
for run_number in 1 2 3; do
    run_generation "$best_mode" prose "$run_number" 600 "$prose_prompt"
    printf '%s\tprose-repeat\t%s\n' "$best_mode" "$bench_tps" >> "$RESULTS_FILE"
done
for run_number in 1 2 3; do
    run_generation "$best_mode" code "$run_number" 600 "$code_prompt"
    printf '%s\tcode-repeat\t%s\n' "$best_mode" "$bench_tps" >> "$RESULTS_FILE"
done
awk -F '\t' -v mode="$best_mode" '
    $2 ~ /-repeat$/ {sum[$2]+=$3; n[$2]++}
    END {for (workload in sum) printf "REPEAT_SUMMARY mode=%s workload=%s n=%d mean=%.2f\n", mode, workload, n[workload], sum[workload]/n[workload]}
' "$RESULTS_FILE" | sort

echo "OFFICIAL_RANDOM_BENCHMARK"
if [[ "$RUNTIME" == "docker" ]]; then
    benchmark_tokenizer=/models/target
else
    benchmark_tokenizer="$MODEL_DIR"
fi
bench_args=(
    -m sglang.benchmark.serving
    --backend sglang-oai
    --host 127.0.0.1
    --port "$PORT"
    --model "$benchmark_tokenizer"
    --served-model-name qwen-3.8-27b
    --tokenizer "$benchmark_tokenizer"
    --dataset-name random
    --random-input-len 8192
    --random-output-len 1024
    --random-range-ratio 1
    --num-prompts 1
    --max-concurrency 1
    --request-rate inf
    --flush-cache
)
if [[ "$RUNTIME" == "docker" ]]; then
    docker exec "$CONTAINER" python3 "${bench_args[@]}"
else
    "$SGLANG_PYTHON" "${bench_args[@]}"
fi

text_payload="$(jq -nc '{
    model: "qwen-3.8-27b",
    messages: [{role: "user", content: "Reply with exactly QWEN38_OK and nothing else."}],
    temperature: 0,
    max_tokens: 64,
    chat_template_kwargs: {enable_thinking: false}
}')"
text_response="$(curl --max-time 300 -fsS \
    "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$text_payload")"
text_content="$(echo "$text_response" | jq -r '.choices[0].message.content // empty')"
printf 'CANARY type=text content=%q\n' "$text_content"
[[ "$text_content" == *QWEN38_OK* ]] || { echo "Text canary failed." >&2; exit 1; }

tool_payload="$(jq -nc '{
    model: "qwen-3.8-27b",
    messages: [{role: "user", content: "Call lookup_weather exactly once for Paris. Do not answer directly."}],
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
    "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$tool_payload")"
tool_name="$(echo "$tool_response" | jq -r '.choices[0].message.tool_calls[0].function.name // empty')"
tool_city="$(echo "$tool_response" | jq -r '
    .choices[0].message.tool_calls[0].function.arguments |
    if type == "string" then (fromjson | .city) else .city end // empty
')"
printf 'CANARY type=tool name=%s city=%s\n' "$tool_name" "$tool_city"
[[ "$tool_name" == lookup_weather && "$tool_city" == Paris ]] || {
    echo "Tool canary failed." >&2
    exit 1
}

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
image_payload="$(jq -nc --arg image_url "data:image/png;base64,${image_b64}" '{
    model: "qwen-3.8-27b",
    messages: [{
        role: "user",
        content: [
            {type: "image_url", image_url: {url: $image_url}},
            {type: "text", text: "What is the dominant color of this square? Reply with one color word."}
        ]
    }],
    temperature: 0,
    max_tokens: 64,
    chat_template_kwargs: {enable_thinking: false}
}')"
image_response="$(curl --max-time 300 -fsS \
    "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$image_payload")"
image_content="$(echo "$image_response" | jq -r '.choices[0].message.content // empty')"
printf 'CANARY type=image content=%q\n' "$image_content"
echo "$image_content" | grep -qi red || { echo "Image canary failed." >&2; exit 1; }

reasoning_probe() {
    local label="$1"
    local effort="${2:-}"
    local transport="${3:-none}"
    local payload response prompt_tokens
    payload="$(jq -nc --arg effort "$effort" --arg transport "$transport" '{
        model: "qwen-3.8-27b",
        messages: [{role: "user", content: "Hi"}],
        temperature: 0,
        max_tokens: 1
    } + (
        if $transport == "top" then {reasoning_effort: $effort}
        elif $transport == "template" then {chat_template_kwargs: {reasoning_effort: $effort}}
        else {} end
    )')"
    response="$(curl --max-time 300 -fsS \
        "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$payload")"
    prompt_tokens="$(echo "$response" | jq -r '.usage.prompt_tokens // 0')"
    printf 'REASONING label=%s prompt_tokens=%s\n' "$label" "$prompt_tokens"
    echo "$prompt_tokens"
}
default_tokens="$(reasoning_probe default | tail -n 1)"
medium_tokens="$(reasoning_probe medium medium template | tail -n 1)"
top_xhigh_tokens="$(reasoning_probe top-xhigh xhigh top | tail -n 1)"
template_xhigh_tokens="$(reasoning_probe template-xhigh xhigh template | tail -n 1)"
printf 'REASONING_SUMMARY default=%s medium=%s top_xhigh=%s template_xhigh=%s\n' \
    "$default_tokens" "$medium_tokens" "$top_xhigh_tokens" "$template_xhigh_tokens"
[[ "$default_tokens" -eq "$medium_tokens" \
    && "$template_xhigh_tokens" -gt "$medium_tokens" ]] || {
    echo "Reasoning default/override check failed." >&2
    exit 1
}

curl --max-time 5 -fsS "http://127.0.0.1:${PORT}/metrics" 2>/dev/null \
    | grep -E '^sglang:spec_(accept_length|accept_rate|block_accept_length)' \
    | tail -n 12 || true
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu \
    --format=csv,noheader
stop_sglang
echo "BENCHMARK_COMPLETE best_mode=$best_mode"
