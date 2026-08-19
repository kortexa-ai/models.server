#!/bin/bash
set -euo pipefail

VARIANT="${1:-}"
MAX_SECONDS="${2:-900}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET_REPO="unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL"
FIXTURE_SOURCE="/Users/francip/.omp/agent/sessions/-src/2026-08-18T17-49-54-690Z_01a015fe-7c82-7000-8b72-e721a5d7b2a4.jsonl"
FIXTURE_CUTOFF="2026-08-19T06:47:08.743Z"
DFLASH_MODEL="/home/francip/data/models/huggingface/mrchuy/Qwen3.8-27B-DFlash-drafter-bootstrap-GGUF/3c89ca499fa04f89a0b4b5ca9b5867953261db39/Qwen3.8-27B-DFlash-bootstrap-Q8_0.gguf"

usage() {
    echo "Usage: $0 mtp1|mtp2|mtp3|dflash [max-seconds]" >&2
    exit 2
}

[[ "$(hostname)" == "smarty" ]] || { echo "Run this on smarty." >&2; exit 1; }
[[ "$MAX_SECONDS" =~ ^[0-9]+$ ]] || usage

SPEC_ARGS=()
case "$VARIANT" in
    mtp1) SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max 1) ;;
    mtp2) SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max 2) ;;
    mtp3) SPEC_ARGS=(--spec-type draft-mtp --spec-draft-n-max 3) ;;
    dflash)
        [[ -f "$DFLASH_MODEL" ]] || { echo "Missing DFlash model: $DFLASH_MODEL" >&2; exit 1; }
        SPEC_ARGS=(--spec-type draft-dflash --spec-draft-model "$DFLASH_MODEL"
            --spec-draft-device CUDA0 --spec-draft-ngl all
            --spec-draft-n-max 3 --spec-draft-n-min 0)
        ;;
    *) usage ;;
esac

if ktxsvc status qwen-3.8-27b 2>&1 | grep -q 'Active: active (running)'; then
    echo "Managed qwen-3.8-27b is still running; stop it with ktxsvc first." >&2
    exit 1
fi
if ss -ltn 'sport = :2053' | grep -q LISTEN; then
    echo "Port 2053 is already in use." >&2
    exit 1
fi

POWER_LIMIT="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)"
[[ "$POWER_LIMIT" == "450" ]] || {
    echo "Expected the existing 450 W cap; found ${POWER_LIMIT} W. Refusing to change it." >&2
    exit 1
}

RUN_ID="$(date +%Y%m%dT%H%M%S)-${VARIANT}-yolo"
RUN_DIR="$HERE/runs/$RUN_ID"
REMOTE_DIR="/tmp/qwen-xid8-${RUN_ID}"
mkdir -p "$RUN_DIR"

START_ISO="$(date --iso-8601=seconds)"
{
    printf 'started=%s\nvariant=%s\napproval_mode=yolo\n' "$START_ISO" "$VARIANT"
    printf 'fixture_cutoff=%s\ntarget=%s\n' "$FIXTURE_CUTOFF" "$TARGET_REPO"
    llama-server --version
    nvidia-smi --query-gpu=name,driver_version,power.limit,temperature.gpu,memory.used --format=csv,noheader
} > "$RUN_DIR/metadata.txt"

SERVER_PID=""
TELEMETRY_PID=""
OMP_PID=""
cleanup() {
    set +e
    [[ -n "$OMP_PID" ]] && kill "$OMP_PID" 2>/dev/null
    [[ -n "$TELEMETRY_PID" ]] && kill "$TELEMETRY_PID" 2>/dev/null
    [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
    [[ -n "$OMP_PID" ]] && wait "$OMP_PID" 2>/dev/null
    [[ -n "$TELEMETRY_PID" ]] && wait "$TELEMETRY_PID" 2>/dev/null
    [[ -n "$SERVER_PID" ]] && wait "$SERVER_PID" 2>/dev/null
    journalctl -k --since "$START_ISO" --no-pager > "$RUN_DIR/kernel.log"
}
trap cleanup EXIT INT TERM

(
    while true; do
        date --iso-8601=ns | tr '\n' ','
        nvidia-smi --query-gpu=timestamp,temperature.gpu,fan.speed,power.draw,power.limit,utilization.gpu,memory.used \
            --format=csv,noheader,nounits
        sleep 1
    done
) > "$RUN_DIR/gpu.csv" &
TELEMETRY_PID="$!"

stdbuf -oL -eL llama-server \
    -hf "$TARGET_REPO" --alias qwen-3.8-27b \
    --host 0.0.0.0 --port 2053 --cors-origins localhost --jinja \
    -c 262144 -ngl 99 --threads -1 --parallel 1 --no-context-shift \
    --temp 0.6 --top-k 20 --top-p 0.95 --load-mode none \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --reasoning-effort medium --log-timestamps \
    "${SPEC_ARGS[@]}" > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID="$!"

for _ in $(seq 1 90); do
    curl -fsS http://127.0.0.1:2053/health >/dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "llama-server exited during load." >&2; exit 1; }
    sleep 1
done
curl -fsS http://127.0.0.1:2053/health >/dev/null

ssh -o BatchMode=yes snappy "mkdir -p '$REMOTE_DIR' && jq -c 'select((.timestamp == null) or (.timestamp <= \"$FIXTURE_CUTOFF\"))' '$FIXTURE_SOURCE' > '$REMOTE_DIR/session.jsonl'"

ssh -tt -o BatchMode=yes snappy "cd /Users/francip/src && REPRO_FIXTURE='$REMOTE_DIR/session.jsonl' expect -c '
log_user 0
set timeout $MAX_SECONDS
set fixture \$env(REPRO_FIXTURE)
spawn omp --resume \$fixture --model smarty/qwen-3.8-27b --thinking high --approval-mode yolo --no-title --max-time ${MAX_SECONDS}s --extension /Applications/GooeyPi.app/Contents/Resources/extensions/omp-work-schedules.ts --extension /Applications/GooeyPi.app/Contents/Resources/extensions/omp-work-browser.ts --extension /Applications/GooeyPi.app/Contents/Resources/extensions/omp-work-collaboration.ts
after 6000
send -- \"/retry\\r\"
expect { timeout { send -- \"\\003\"; after 1000; exit 124 } eof }
'" > "$RUN_DIR/omp-control.log" 2>&1 &
OMP_PID="$!"

OUTCOME="timeout"
for _ in $(seq 1 $((MAX_SECONDS / 5))); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        OUTCOME="server-exited"
        break
    fi
    if ssh -o BatchMode=yes snappy "tail -n 1 '$REMOTE_DIR/session.jsonl' | jq -e '.message.role == \"assistant\" and .message.stopReason == \"stop\"'" >/dev/null 2>&1; then
        OUTCOME="completed"
        break
    fi
    sleep 5
done

printf 'outcome=%s\nfinished=%s\nremote_fixture=%s/session.jsonl\n' \
    "$OUTCOME" "$(date --iso-8601=seconds)" "$REMOTE_DIR" >> "$RUN_DIR/metadata.txt"
echo "$OUTCOME: $RUN_DIR"
[[ "$OUTCOME" == "completed" ]]
