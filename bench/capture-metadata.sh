#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <model-id> <output-directory>" >&2
}

[[ $# -eq 2 ]] || {
    usage
    exit 2
}

MODEL_ID="$1"
OUTPUT_DIR="$2"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_FILE="${PROJECT_ROOT}/${MODEL_ID}/model.json"
LLAMA_ROOT="$(cd "${PROJECT_ROOT}/../llama.cpp" && pwd)"

for command_name in curl git jq nvidia-smi python3 sha256sum ss; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "Missing required command: $command_name" >&2
        exit 1
    }
done
[[ -f "$MODEL_FILE" ]] || {
    echo "Unknown model or missing model.json: $MODEL_ID" >&2
    exit 2
}

mkdir -p "$OUTPUT_DIR"
cp "$MODEL_FILE" "${OUTPUT_DIR}/model.json"
python3 "${PROJECT_ROOT}/scripts/parse-config.py" "$MODEL_FILE" \
    >"${OUTPUT_DIR}/effective-config.env"

MODEL_PORT="$(jq -er '.port' "$MODEL_FILE")"
BASE_URL="http://127.0.0.1:${MODEL_PORT}"
curl --max-time 5 -fsS "${BASE_URL}/health" >"${OUTPUT_DIR}/health.json"
curl --max-time 10 -fsS "${BASE_URL}/props" >"${OUTPUT_DIR}/props.json"

SOCKET_LINE="$(ss -H -ltnp "sport = :${MODEL_PORT}" 2>/dev/null || true)"
SERVER_PID="$(printf '%s\n' "$SOCKET_LINE" | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n 1)"
[[ -n "$SERVER_PID" && -r "/proc/${SERVER_PID}/environ" ]] || {
    echo "Could not resolve a readable listener PID for port $MODEL_PORT" >&2
    exit 1
}

SERVER_EXE="$(readlink -f "/proc/${SERVER_PID}/exe")"
tr '\0' ' ' <"/proc/${SERVER_PID}/cmdline" \
    | sed 's/[[:space:]]*$//' >"${OUTPUT_DIR}/process-command.txt"
tr '\0' '\n' <"/proc/${SERVER_PID}/environ" \
    | rg '^(CUDA_|GGML_|LLAMA_)' \
    | LC_ALL=C sort >"${OUTPUT_DIR}/process-environment.txt" || true
tr '\n' ' ' <"/proc/${SERVER_PID}/cgroup" \
    | sed 's/[[:space:]]*$//' >"${OUTPUT_DIR}/process-cgroup.txt"

if rg -q '^GGML_CUDA_ENABLE_UNIFIED_MEMORY=' \
    "${OUTPUT_DIR}/process-environment.txt"; then
    UNIFIED_MEMORY_STATE="enabled"
else
    UNIFIED_MEMORY_STATE="disabled"
fi
mapfile -t SERVER_GPU_UUIDS < <(
    nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits \
        | awk -F, -v pid="$SERVER_PID" '
            {
                current_pid=$1
                uuid=$2
                gsub(/[[:space:]]/, "", current_pid)
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", uuid)
                if (current_pid == pid) print uuid
            }
        ' | LC_ALL=C sort -u
)
if ((${#SERVER_GPU_UUIDS[@]} > 0)); then
    GPU_RESIDENT="true"
else
    GPU_RESIDENT="false"
fi
if ((${#SERVER_GPU_UUIDS[@]} != 1)); then
    echo "Expected live PID $SERVER_PID on exactly one GPU; found ${#SERVER_GPU_UUIDS[@]}." >&2
    exit 1
fi
SERVER_GPU_UUID="${SERVER_GPU_UUIDS[0]}"
if rg -q -- '(^| )--device none( |$)' "${OUTPUT_DIR}/process-command.txt"; then
    CUDA_INFERENCE="false"
else
    CUDA_INFERENCE="$GPU_RESIDENT"
fi
if [[ "$CUDA_INFERENCE" == "false" ]]; then
    CUDA_GRAPH_STATE="not_applicable"
elif rg -q '^GGML_CUDA_DISABLE_GRAPHS=' \
    "${OUTPUT_DIR}/process-environment.txt"; then
    CUDA_GRAPH_STATE="disabled"
else
    CUDA_GRAPH_STATE="enabled"
fi

"$SERVER_EXE" --version >"${OUTPUT_DIR}/server-version.txt" 2>&1 || true
if [[ -x "${PROJECT_ROOT}/.venv-bench/bin/python" ]]; then
    "${PROJECT_ROOT}/.venv-bench/bin/python" - <<'PY' \
        >"${OUTPUT_DIR}/python-packages.txt"
from importlib.metadata import distributions

packages = sorted(
    f"{item.metadata['Name']}=={item.version}"
    for item in distributions()
    if item.metadata.get("Name")
)
print("\n".join(packages))
PY
else
    : >"${OUTPUT_DIR}/python-packages.txt"
fi

nvidia-smi --id="$SERVER_GPU_UUID" \
    --query-gpu=name,uuid,driver_version,pstate,power.limit,power.default_limit,clocks.current.graphics,clocks.current.memory,temperature.gpu,memory.total,memory.used,memory.free \
    --format=csv >"${OUTPUT_DIR}/gpu-before.csv"

CAPTURED_LOCAL="$(date --iso-8601=seconds)"
CAPTURED_UTC="$(date --utc --iso-8601=seconds)"
MODEL_SHA256="$(sha256sum "$MODEL_FILE" | awk '{print $1}')"
SERVER_SHA256="$(sha256sum "$SERVER_EXE" | awk '{print $1}')"
PROJECT_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
LLAMA_COMMIT="$(git -C "$LLAMA_ROOT" rev-parse HEAD)"
PROJECT_STATUS="$(git -C "$PROJECT_ROOT" status --porcelain)"
LLAMA_STATUS="$(git -C "$LLAMA_ROOT" status --porcelain)"
HOST_NAME="$(hostname)"
KERNEL="$(uname -srvmo)"
GPU_POWER_LIMIT="$(nvidia-smi --id="$SERVER_GPU_UUID" --query-gpu=power.limit --format=csv,noheader,nounits | awk '{printf "%.0f", $1}')"
PROCESS_COMMAND="$(<"${OUTPUT_DIR}/process-command.txt")"
PROCESS_CGROUP="$(<"${OUTPUT_DIR}/process-cgroup.txt")"
SERVER_VERSION="$(<"${OUTPUT_DIR}/server-version.txt")"

jq -n \
    --arg schema_version "1" \
    --arg captured_local "$CAPTURED_LOCAL" \
    --arg captured_utc "$CAPTURED_UTC" \
    --arg model_id "$MODEL_ID" \
    --arg model_sha256 "$MODEL_SHA256" \
    --arg base_url "$BASE_URL" \
    --argjson port "$MODEL_PORT" \
    --argjson pid "$SERVER_PID" \
    --arg process_command "$PROCESS_COMMAND" \
    --arg process_cgroup "$PROCESS_CGROUP" \
    --arg server_executable "$SERVER_EXE" \
    --arg server_sha256 "$SERVER_SHA256" \
    --arg server_version "$SERVER_VERSION" \
    --arg project_commit "$PROJECT_COMMIT" \
    --arg project_status "$PROJECT_STATUS" \
    --arg llama_commit "$LLAMA_COMMIT" \
    --arg llama_status "$LLAMA_STATUS" \
    --arg hostname "$HOST_NAME" \
    --arg kernel "$KERNEL" \
    --arg gpu_uuid "$SERVER_GPU_UUID" \
    --arg gpu_power_limit_w "$GPU_POWER_LIMIT" \
    --arg cuda_graphs "$CUDA_GRAPH_STATE" \
    --arg unified_memory "$UNIFIED_MEMORY_STATE" \
    --argjson gpu_resident "$GPU_RESIDENT" \
    --argjson cuda_inference "$CUDA_INFERENCE" \
    '{
        schema_version: ($schema_version | tonumber),
        captured_local: $captured_local,
        captured_utc: $captured_utc,
        model: {
            id: $model_id,
            config_sha256: $model_sha256
        },
        endpoint: {
            base_url: $base_url,
            port: $port
        },
        process: {
            pid: $pid,
            command: $process_command,
            cgroup: $process_cgroup,
            executable: $server_executable,
            executable_sha256: $server_sha256,
            version: $server_version,
            gpu_resident: $gpu_resident,
            cuda_inference: $cuda_inference,
            cuda_graphs: $cuda_graphs,
            cuda_unified_memory: $unified_memory
        },
        source: {
            models_server_commit: $project_commit,
            models_server_status: $project_status,
            llama_cpp_commit: $llama_commit,
            llama_cpp_status: $llama_status
        },
        system: {
            hostname: $hostname,
            kernel: $kernel,
            gpu_uuid: $gpu_uuid,
            gpu_power_limit_w: ($gpu_power_limit_w | tonumber)
        }
    }' >"${OUTPUT_DIR}/manifest.json"

if [[ "$UNIFIED_MEMORY_STATE" == "enabled" ]]; then
    echo "Refusing benchmark: live PID $SERVER_PID has GGML_CUDA_ENABLE_UNIFIED_MEMORY." >&2
    exit 3
fi

printf 'Captured model=%s pid=%s power_limit_w=%s cuda_graphs=%s unified_memory=%s\n' \
    "$MODEL_ID" "$SERVER_PID" "$GPU_POWER_LIMIT" \
    "$CUDA_GRAPH_STATE" "$UNIFIED_MEMORY_STATE"
