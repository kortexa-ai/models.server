#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/qwen-3.8-flash-next"
CONFIG="${MODEL_DIR}/model.json"
NODE="$(hostname -s)"
case "$NODE" in
    static|shock) ;;
    *) echo "vllm-spark-tp2 is only configured for static and shock (got ${NODE})." >&2; exit 2 ;;
esac

config_value() {
    python3 - "$CONFIG" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value[key]
if isinstance(value, (dict, list)):
    print(json.dumps(value, separators=(",", ":")))
else:
    print(value)
PY
}

NODE_JSON="$(config_value "vllm_spark_tp2.nodes.${NODE}")"
RANK="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["rank"])' "$NODE_JSON")"
HOST_IP="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["ip"])' "$NODE_JSON")"
MODEL_ID="$(config_value id)"
MODEL_PATH="$(config_value vllm_spark_tp2.model_path)"
IMAGE="$(config_value vllm_spark_tp2.image)"
IMAGE_ID="$(config_value vllm_spark_tp2.image_id)"
MAX_MODEL_LEN="$(config_value vllm_spark_tp2.max_model_len)"
MAX_NUM_SEQS="$(config_value vllm_spark_tp2.max_num_seqs)"
GPU_MEMORY_UTILIZATION="$(config_value vllm_spark_tp2.gpu_memory_utilization)"
MAX_BATCHED_TOKENS="$(config_value vllm_spark_tp2.max_num_batched_tokens)"
KV_CACHE_DTYPE="$(config_value vllm_spark_tp2.kv_cache_dtype)"
ENABLE_PREFIX_CACHING="$(config_value vllm_spark_tp2.enable_prefix_caching)"
SPECULATIVE_CONFIG="$(config_value vllm_spark_tp2.speculative_config)"
HF_OVERRIDES="$(config_value vllm_spark_tp2.hf_overrides)"
MASTER_ADDR="$(config_value vllm_spark_tp2.nodes.static.ip)"
MASTER_PORT="$(config_value vllm_spark_tp2.master_port)"
CONTAINER="qwen38fn-${NODE}"

if [[ "${1:-}" == "--stop" ]]; then
    docker stop --time 60 "$CONTAINER" >/dev/null 2>&1 || true
    exit 0
fi

if [[ ! -f "$MODEL_PATH/config.json" || ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
    echo "Model files are missing under ${MODEL_PATH}." >&2
    exit 3
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Pinned vLLM image is not present locally: ${IMAGE}" >&2
    exit 5
fi
ACTUAL_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
if [[ "$ACTUAL_IMAGE_ID" != "$IMAGE_ID" ]]; then
    echo "vLLM image ID mismatch: expected ${IMAGE_ID}, got ${ACTUAL_IMAGE_ID}." >&2
    exit 5
fi
if docker inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "Container ${CONTAINER} already exists; inspect it before starting another." >&2
    exit 6
fi

CACHE_DIR="${QWEN_CACHE_DIR:-${HOME}/.cache/qwen38fn-vllm}"
mkdir -p "$CACHE_DIR"
GRAPH_ARGS=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}')
SPEC_ARGS=(--speculative-config "$SPECULATIVE_CONFIG")
if [[ "$ENABLE_PREFIX_CACHING" == "true" ]]; then
    PREFIX_CACHE_ARGS=(--enable-prefix-caching)
else
    PREFIX_CACHE_ARGS=(--no-enable-prefix-caching)
fi

if [[ "$RANK" == "1" ]]; then
    HEADLESS_ARGS=(--headless)
else
    HEADLESS_ARGS=()
fi

echo "Starting ${MODEL_ID} on ${NODE} rank=${RANK} TP=2 context=${MAX_MODEL_LEN}."
exec docker run --rm --name "$CONTAINER" --gpus all \
    --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
    --device /dev/infiniband:/dev/infiniband \
    -v "${MODEL_PATH}:/models/qwen38fn:ro" -v "${CACHE_DIR}:/root/.cache" \
    -e VLLM_HOST_IP="$HOST_IP" -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a \
    -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA="$(config_value vllm_spark_tp2.nccl.hca)" \
    -e NCCL_IB_GID_INDEX="$(config_value vllm_spark_tp2.nccl.gid_index)" \
    -e NCCL_IB_ROCE_VERSION_NUM="$(config_value vllm_spark_tp2.nccl.roce_version)" \
    -e NCCL_IB_ADDR_FAMILY=AF_INET \
    -e NCCL_SOCKET_IFNAME="$(config_value vllm_spark_tp2.nccl.socket_interface)" \
    -e GLOO_SOCKET_IFNAME="$(config_value vllm_spark_tp2.nccl.socket_interface)" \
    -e TP_SOCKET_IFNAME="$(config_value vllm_spark_tp2.nccl.socket_interface)" \
    -e MN_IF_NAME="$(config_value vllm_spark_tp2.nccl.socket_interface)" \
    -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=1 -e NCCL_CUMEM_ENABLE=0 \
    -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    "$IMAGE" /models/qwen38fn --served-model-name "$MODEL_ID" \
    --host 0.0.0.0 --port "$(config_value port)" --trust-remote-code \
    --quantization modelopt --tensor-parallel-size "$(config_value vllm_spark_tp2.tensor_parallel_size)" \
    --hf-overrides "$HF_OVERRIDES" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" --no-enable-flashinfer-autotune "${PREFIX_CACHE_ARGS[@]}" \
    --reasoning-parser "$(config_value vllm_spark_tp2.reasoning_parser)" \
    --enable-auto-tool-choice --tool-call-parser "$(config_value vllm_spark_tp2.tool_call_parser)" \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    "${SPEC_ARGS[@]}" "${GRAPH_ARGS[@]}" \
    --distributed-executor-backend mp --nnodes 2 --node-rank "$RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" "${HEADLESS_ARGS[@]}"
