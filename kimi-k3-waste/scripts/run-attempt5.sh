#!/usr/bin/env bash
# Guarded one-off runner for the Kimi K3 WASTE conversion on Smarty.
set -euo pipefail

repo=/home/francip/src/waste
src=/mnt/data/k3
out=/home/francip/models/k3.waste
run=/home/francip/models/k3.waste.runs
script_path="$run/run-attempt5.sh"
expected_index=a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd
verify_log="$run/source-verify.log"

say() {
    printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

abort_conversion() {
    say "GUARD ABORT: $*"
    tmux send-keys -t waste-k3-supervisor C-c 2>/dev/null || true
    sleep 15
    tmux has-session -t waste-k3-supervisor 2>/dev/null &&
        tmux send-keys -t waste-k3-supervisor C-c 2>/dev/null || true
    exit 1
}

rotate_attempt4() {
    local stem src_path dst_path
    for stem in supervisor conversion post-conversion progress gpu-guard; do
        src_path="$run/$stem.log"
        dst_path="$run/$stem-attempt4.log"
        if [[ -e "$src_path" ]]; then
            [[ ! -e "$dst_path" ]] || {
                say "refusing to overwrite $dst_path"
                exit 1
            }
            mv "$src_path" "$dst_path"
        fi
    done
    if [[ -e "$run/.post-failed" ]]; then
        [[ ! -e "$run/.post-failed.attempt4" ]] || {
            say "refusing to overwrite $run/.post-failed.attempt4"
            exit 1
        }
        mv "$run/.post-failed" "$run/.post-failed.attempt4"
    fi
}

launch() {
    mkdir -p "$run"
    grep -Fq 'All checksums match.' "$verify_log" || {
        say "source verifier has not reported a full checksum pass"
        exit 1
    }
    grep -Fxq 'VERIFY_EXIT=0' "$verify_log" || {
        say "source verifier did not exit successfully"
        exit 1
    }

    local actual_index free_mib layer size
    actual_index=$(taskset -c 16 sha256sum \
        "$src/model.safetensors.index.json" | awk '{print $1}')
    [[ "$actual_index" == "$expected_index" ]] || {
        say "source index changed after verification"
        exit 1
    }

    [[ "$(uname -r)" == 7.0.0-28-generic ]] || {
        say "unexpected kernel: $(uname -r)"
        exit 1
    }
    nvidia-smi >/dev/null
    free_mib=$(nvidia-smi --query-gpu=memory.free \
        --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 16384 )) || {
        say "only ${free_mib:-unknown} MiB GPU memory is free"
        exit 1
    }

    for layer in 1 2 3; do
        size=$(stat -c %s "$out/experts-L${layer}.bin")
        [[ "$size" == 11116478464 ]] || {
            say "published layer $layer has unexpected size $size"
            exit 1
        }
        size=$(stat -c %s "$out/codebooks-L${layer}.bin")
        [[ "$size" == 37008 ]] || {
            say "published codebook layer $layer has unexpected size $size"
            exit 1
        }
    done

    pgrep -af '[t]ools/convert.py.*--src /mnt/data/k3' && {
        say "a converter process already exists"
        exit 1
    }
    for session in waste-k3-supervisor waste-k3-gpu-guard \
                   waste-k3-progress waste-k3-post waste-k3-serve; do
        tmux has-session -t "$session" 2>/dev/null && {
            say "tmux session already exists: $session"
            exit 1
        }
    done

    rotate_attempt4
    say "attempt 5 launch: kernel $(uname -r), ${free_mib} MiB GPU free" \
        >"$run/supervisor.log"

    tmux new-session -d -s waste-k3-supervisor -c "$repo" \
        "$script_path convert"
    tmux new-session -d -s waste-k3-gpu-guard \
        "$script_path guard"
    tmux new-session -d -s waste-k3-progress \
        "$script_path progress"
    tmux new-session -d -s waste-k3-post \
        "$run/post-conversion.sh"

    say "attempt 5 sessions started"
    tmux list-sessions
}

convert() {
    cd "$repo"
    say "conversion command starting" >>"$run/supervisor.log"
    set +e
    taskset -c 16-31 env \
        OMP_NUM_THREADS=8 \
        MKL_NUM_THREADS=8 \
        OPENBLAS_NUM_THREADS=8 \
        NUMEXPR_NUM_THREADS=8 \
        PYTHONUNBUFFERED=1 \
        WASTE_DISABLE_NATIVE_VQ=1 \
        uv run --with torch --with safetensors python tools/convert.py \
            --src "$src" --out "$out" --jobs 1 --device cuda \
            >"$run/conversion.log" 2>&1
    rc=$?
    set -e
    say "conversion exited with status $rc" >>"$run/supervisor.log"
    exit "$rc"
}

guard() {
    exec >>"$run/gpu-guard.log" 2>&1
    local started cycle free_mib actual_index errors
    started=$(date '+%Y-%m-%d %H:%M:%S')
    cycle=0
    say "guard armed: 8192 MiB floor, source index $expected_index"

    while tmux has-session -t waste-k3-supervisor 2>/dev/null; do
        free_mib=$(nvidia-smi --query-gpu=memory.free \
            --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ') ||
            abort_conversion "nvidia-smi failed"
        [[ "$free_mib" =~ ^[0-9]+$ ]] ||
            abort_conversion "could not parse free GPU memory: $free_mib"
        (( free_mib >= 8192 )) ||
            abort_conversion "GPU free memory fell to $free_mib MiB"

        if (( cycle % 4 == 0 )); then
            actual_index=$(taskset -c 16 sha256sum \
                "$src/model.safetensors.index.json" | awk '{print $1}')
            [[ "$actual_index" == "$expected_index" ]] ||
                abort_conversion "source index checksum changed to $actual_index"
        fi

        errors=$(sudo -n journalctl -k -b --since "$started" --no-pager |
            grep -E 'BUG: unable|Oops:|Hardware Error|Machine check|MCE|Xid' |
            tail -5 || true)
        [[ -z "$errors" ]] || abort_conversion "kernel/GPU error: $errors"

        if (( cycle % 20 == 0 )); then
            say "healthy: $free_mib MiB GPU free"
        fi
        cycle=$((cycle + 1))
        sleep 15
    done
    say "supervisor session ended; guard exiting"
}

progress() {
    exec >>"$run/progress.log" 2>&1
    local layers bytes temps
    while tmux has-session -t waste-k3-supervisor 2>/dev/null; do
        layers=$(find "$out" -maxdepth 1 -type f -name 'experts-L*.bin' |
            wc -l)
        bytes=$(find "$out" -maxdepth 1 -type f -name 'experts-L*.bin' \
            -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum + 0}')
        temps=$(find "$out" -maxdepth 1 -type f -name '*.tmp' -printf '%f:%s ')
        say "published_layers=$layers expert_bytes=$bytes temps=${temps:-none}"
        tail -1 "$run/conversion.log" 2>/dev/null || true
        sleep 300
    done
    say "supervisor session ended; progress watcher exiting"
}

case "${1:-launch}" in
    launch) launch ;;
    convert) convert ;;
    guard) guard ;;
    progress) progress ;;
    *) printf 'usage: %s [launch|convert|guard|progress]\n' "$0" >&2; exit 2 ;;
esac
