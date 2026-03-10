#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==4.57.6}"
TRANSFORMERS_TAG_SUFFIX="${TRANSFORMERS_SPEC#transformers==}"
TAG="${TAG:-local/trtllm-qwen35:transformers-${TRANSFORMERS_TAG_SUFFIX}}"

echo "Building TensorRT-LLM derivative image"
echo "Base image: ${BASE_IMAGE}"
echo "Transformers spec: ${TRANSFORMERS_SPEC}"
echo "Target tag: ${TAG}"

exec docker build \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC}" \
    -f "${SCRIPT_DIR}/Dockerfile.transformers-5.3" \
    -t "${TAG}" \
    "${SCRIPT_DIR}"
