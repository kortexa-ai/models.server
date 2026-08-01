#!/usr/bin/env bash
# Operational watcher for Smarty's one-off Kimi K3 WASTE experiment.
set -uo pipefail

repo=/home/francip/src/waste
src=/mnt/data/k3
out=/home/francip/models/k3.waste
run=/home/francip/models/k3.waste.runs
supervisor_log="$run/supervisor.log"
log="$run/post-conversion.log"

mkdir -p "$run"
exec >>"$log" 2>&1

say() {
    printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
    say "FAILED: $*"
    printf '%s\n' "$*" >"$run/.post-failed"
    exit 1
}

say "post-conversion watcher armed"

reports=0
missing_sessions=0
while ! grep -q 'conversion exited with status 0' "$supervisor_log" 2>/dev/null; do
    if grep -Eq 'conversion exited with status ([1-9][0-9]*|-[0-9]+)' \
        "$supervisor_log" 2>/dev/null; then
        die "conversion exited nonzero; see $supervisor_log and $run/conversion.log"
    fi

    if tmux has-session -t waste-k3-download 2>/dev/null ||
       tmux has-session -t waste-k3-supervisor 2>/dev/null ||
       tmux has-session -t waste-k3-convert 2>/dev/null; then
        missing_sessions=0
    else
        missing_sessions=$((missing_sessions + 1))
        if (( missing_sessions >= 3 )); then
            die "download/conversion sessions vanished without a successful conversion status"
        fi
    fi

    reports=$((reports + 1))
    if (( reports % 30 == 0 )); then
        complete=$(sort -u "$src/.download-state" 2>/dev/null |
            sed '/^$/d' | wc -l)
        bytes=$(find "$src" -maxdepth 1 -type f \
            -name 'model-*-of-000096.safetensors' -printf '%s\n' 2>/dev/null |
            awk '{s+=$1} END {printf "%.0f",s+0}')
        say "waiting: $complete/96 shards, ${bytes:-0} source bytes present"
    fi
    sleep 60
done

say "conversion reported status 0; auditing published container"
pgrep -af '[t]ools/convert.py.*--src /mnt/data/k3' &&
    die "converter process still exists after its successful exit marker"
[[ -s "$out/manifest.json" ]] || die "manifest.json missing or empty"

cd "$repo" || die "cannot enter $repo"

say "round-tripping one expert per layer against source weights"
if ! uv run --quiet --with torch --with safetensors --no-project python \
    tools/verify_container.py --container "$out" --src "$src" --experts 1 \
    >"$run/verify.txt" 2>&1; then
    die "container round-trip command failed; see $run/verify.txt"
fi
grep -q '^PASS' "$run/verify.txt" ||
    die "container round-trip did not report PASS; see $run/verify.txt"
say "container round-trip PASS"

make -s || die "WASTE build failed"
./waste info "$out" --json >"$run/info.json" 2>"$run/info.stderr" ||
    die "waste info failed"
./waste plan "$out" --budget 46G --json >"$run/plan.json" \
    2>"$run/plan.stderr" || die "46G memory plan failed"

available=$(free -b | awk '/^Mem:/ {print $7}')
(( available >= 64 * 1024 * 1024 * 1024 )) ||
    die "less than 64 GiB memory available before benchmark"
say "container opens and 46G plan succeeds; ${available} bytes memory available"

printf 'threads\ttok_per_s\tresult\n' >"$run/bench-threads.tsv"
best_threads=0
best_rate=0
for threads in 4 8 16 24 32; do
    result="$run/bench-thread-${threads}.json"
    errors="$run/bench-thread-${threads}.stderr"
    say "benchmarking $threads threads, 8 generated tokens"
    if timeout 3600 ./waste bench "$out" -n 8 --budget 46G \
        --threads "$threads" --json >"$result" 2>"$errors"; then
        rate=$(sed -n 's/.*"tok_per_s":\([0-9.][0-9.]*\).*/\1/p' "$result" |
            tail -1)
        if [[ -n "$rate" ]]; then
            printf '%s\t%s\tok\n' "$threads" "$rate" \
                >>"$run/bench-threads.tsv"
            if awk -v new="$rate" -v old="$best_rate" \
                'BEGIN {exit !(new > old)}'; then
                best_rate=$rate
                best_threads=$threads
            fi
            say "$threads threads: $rate tok/s"
        else
            printf '%s\t\tmissing-rate\n' "$threads" \
                >>"$run/bench-threads.tsv"
            say "$threads threads returned no parseable rate"
        fi
    else
        printf '%s\t\tfailed\n' "$threads" >>"$run/bench-threads.tsv"
        say "$threads-thread benchmark failed; continuing sweep"
    fi
done
(( best_threads > 0 )) || die "every thread-count benchmark failed"
printf '%s\n' "$best_threads" >"$run/best-threads"
say "best cold setting: $best_threads threads at $best_rate tok/s"

say "learning a hotlist with the winning thread count"
timeout 3600 ./waste bench "$out" -n 16 --budget 46G \
    --threads "$best_threads" --learn --json \
    >"$run/bench-learn-first.json" 2>"$run/bench-learn-first.stderr" ||
    die "first learned-cache benchmark failed"
timeout 3600 ./waste bench "$out" -n 16 --budget 46G \
    --threads "$best_threads" --learn --json \
    >"$run/bench-learn-warm.json" 2>"$run/bench-learn-warm.stderr" ||
    die "warm learned-cache benchmark failed"

say "running a short deterministic generation"
timeout 3600 ./waste run "$out" 'The capital of France is' \
    -n 6 --temp 0 --budget 46G --threads "$best_threads" \
    >"$run/generated.txt" 2>&1 ||
    die "engine generation failed; see $run/generated.txt"

port=18080
while ss -H -ltn | awk '{print $4}' | grep -Eq "[:.]${port}$"; do
    port=$((port + 1))
    (( port <= 18120 )) || die "no free localhost port in 18080..18120"
done
printf '%s\n' "$port" >"$run/server-port"

tmux has-session -t waste-k3-serve 2>/dev/null &&
    die "waste-k3-serve session already exists"
say "starting localhost OpenAI-compatible server on port $port"
tmux new-session -d -s waste-k3-serve -c "$repo" \
    "exec python3 -m serve '$out' --host 127.0.0.1 --port '$port' --budget 46G --threads '$best_threads' --max-tokens 8 --no-thinking >>'$run/server.log' 2>&1" ||
    die "could not create server tmux session"

healthy=0
for _ in $(seq 1 120); do
    if curl -fsS --max-time 5 "http://127.0.0.1:$port/health" \
        >"$run/health.json"; then
        healthy=1
        break
    fi
    tmux has-session -t waste-k3-serve 2>/dev/null ||
        die "server exited before becoming healthy; see $run/server.log"
    sleep 5
done
(( healthy == 1 )) || die "server did not become healthy in ten minutes"

say "server healthy; issuing four-token chat smoke test"
curl --fail-with-body -sS --max-time 3600 \
    -H 'Content-Type: application/json' \
    -d '{"model":"k3","messages":[{"role":"user","content":"Reply with exactly: hello puny human"}],"max_completion_tokens":4,"temperature":0,"stream":false}' \
    "http://127.0.0.1:$port/v1/chat/completions" \
    >"$run/server-response.json" ||
    die "server chat request failed; see $run/server.log"

python3 - "$run/server-response.json" <<'PY' ||
import json
import sys

path = sys.argv[1]
response = json.load(open(path))
if "error" in response:
    raise SystemExit(f"server returned error: {response['error']}")
choices = response.get("choices")
if not choices:
    raise SystemExit("server returned no choices")
print(json.dumps({
    "model": response.get("model"),
    "usage": response.get("usage"),
    "waste": response.get("waste"),
    "message": choices[0].get("message"),
}, ensure_ascii=False))
PY
    die "server response validation failed"

date '+%F %T %Z' >"$run/.post-success"
say "POST-CONVERSION SUCCESS: verified, benchmarked, generated, and serving on 127.0.0.1:$port"
