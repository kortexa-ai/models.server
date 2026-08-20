#!/bin/bash
set -euo pipefail

TAG="${1:-}"
MAX_SECONDS="${2:-900}"
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SOURCE_SESSION="$HERE/omp-session.jsonl"
SOURCE_SHA256="2570600738ad744b4adca283a9dc1d272ac665bd7db21bd526ed9068d0f3b3e3"
SERVICE="kortexa-ai-llm-qwen-3.8-27b.service"

[[ "$(hostname)" == "smarty" ]] || { echo "Run this on smarty." >&2; exit 1; }
[[ -n "$TAG" && "$TAG" =~ ^[a-z0-9-]+$ ]] || {
    echo "Usage: $0 <run-tag> [max-seconds]" >&2
    exit 2
}
[[ "$MAX_SECONDS" =~ ^[0-9]+$ ]] || { echo "max-seconds must be an integer." >&2; exit 2; }
echo "$SOURCE_SHA256  $SOURCE_SESSION" | sha256sum -c - >/dev/null

MAIN_PID="$(systemctl show "$SERVICE" -p MainPID --value)"
[[ "$MAIN_PID" =~ ^[1-9][0-9]*$ && -r "/proc/$MAIN_PID/cmdline" ]] || {
    echo "$SERVICE is not running." >&2
    exit 1
}
POWER_LIMIT="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)"
[[ "$POWER_LIMIT" == 450 ]] || { echo "Expected the existing 450 W cap; found $POWER_LIMIT W." >&2; exit 1; }

RUN_ID="$(date +%Y%m%dT%H%M%S)-$TAG"
RUN_DIR="$HERE/runs/$RUN_ID"
REMOTE_DIR="/tmp/qwen-xid8-$RUN_ID"
START_ISO="$(date --iso-8601=seconds)"
START_EPOCH="$(date +%s)"
mkdir -p "$RUN_DIR"

{
    printf 'started=%s\ntag=%s\napproval_mode=yolo\npower_limit_w=%s\n' \
        "$START_ISO" "$TAG" "$POWER_LIMIT"
    printf 'source_session_sha256=%s\nserver_pid=%s\n' "$SOURCE_SHA256" "$MAIN_PID"
    printf 'server_cmdline='
    tr '\0' ' ' < "/proc/$MAIN_PID/cmdline"
    printf '\nserver_environment='
    tr '\0' '\n' < "/proc/$MAIN_PID/environ" | grep '^GGML_CUDA_DISABLE_GRAPHS=' || true
} > "$RUN_DIR/metadata.txt"

TELEMETRY_PID=""
OMP_PID=""
cleanup() {
    set +e
    [[ -n "$OMP_PID" ]] && kill "$OMP_PID" 2>/dev/null
    [[ -n "$TELEMETRY_PID" ]] && kill "$TELEMETRY_PID" 2>/dev/null
    [[ -n "$OMP_PID" ]] && wait "$OMP_PID" 2>/dev/null
    [[ -n "$TELEMETRY_PID" ]] && wait "$TELEMETRY_PID" 2>/dev/null
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

ssh -o BatchMode=yes snappy "mkdir -p '$REMOTE_DIR'"
scp -q "$SOURCE_SESSION" "snappy:$REMOTE_DIR/session.jsonl"

ssh -tt -o BatchMode=yes snappy "cd /Users/francip/src/magicapp && REPRO_FIXTURE='$REMOTE_DIR/session.jsonl' expect -c '
log_user 0
set timeout $MAX_SECONDS
set fixture \$env(REPRO_FIXTURE)
spawn omp --resume \$fixture --model smarty/qwen-3.8-27b --thinking medium --approval-mode yolo --no-title --max-time ${MAX_SECONDS}s
after 4000
send -- \"/retry\\r\"
expect { timeout { send -- \"\\003\"; after 1000; exit 124 } eof }
'" > "$RUN_DIR/omp-control.log" 2>&1 &
OMP_PID="$!"

OUTCOME="timeout"
for _ in $(seq 1 $((MAX_SECONDS / 5))); do
    if [[ ! -r "/proc/$MAIN_PID/cmdline" ]]; then
        OUTCOME="server-exited"
        break
    fi
    if ssh -o BatchMode=yes snappy "tail -n 1 '$REMOTE_DIR/session.jsonl' | jq -e '.message.role == \"assistant\" and .message.stopReason == \"stop\"'" >/dev/null 2>&1; then
        OUTCOME="completed"
        break
    fi
    if ! kill -0 "$OMP_PID" 2>/dev/null; then
        OUTCOME="client-exited"
        break
    fi
    sleep 5
done

if kill -0 "$OMP_PID" 2>/dev/null; then
    [[ "$OUTCOME" == completed ]] && sleep 2
    kill "$OMP_PID" 2>/dev/null || true
fi
set +e
wait "$OMP_PID"
OMP_STATUS="$?"
set -e
OMP_PID=""
ELAPSED_SECONDS=$(( $(date +%s) - START_EPOCH ))
if [[ "$OUTCOME" == client-exited && "$ELAPSED_SECONDS" -ge $((MAX_SECONDS - 10)) ]]; then
    OUTCOME="timeout"
fi

kill "$TELEMETRY_PID" 2>/dev/null || true
wait "$TELEMETRY_PID" 2>/dev/null || true
TELEMETRY_PID=""

scp -q "snappy:$REMOTE_DIR/session.jsonl" "$RUN_DIR/replayed-session.jsonl"
ssh -o BatchMode=yes snappy "rm -rf '${REMOTE_DIR:?}'"
journalctl -u "$SERVICE" --since "$START_ISO" --no-pager > "$RUN_DIR/service.log"
journalctl -k --since "$START_ISO" --no-pager > "$RUN_DIR/kernel.log"
printf 'omp_status=%s\noutcome=%s\nelapsed_seconds=%s\nfinished=%s\n' \
    "$OMP_STATUS" "$OUTCOME" "$ELAPSED_SECONDS" "$(date --iso-8601=seconds)" >> "$RUN_DIR/metadata.txt"

echo "$OUTCOME: $RUN_DIR"
[[ "$OUTCOME" == completed ]]
