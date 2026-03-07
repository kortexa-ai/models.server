#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

IMAGE_TAG="${IMAGE_TAG:-kortexa/qwen-3.5-27b-vllm:ngc-26.02}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/vllm:26.02-py3}"
VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-vllm>=0.17.0}"
VLLM_WHEEL_INDEX="${VLLM_WHEEL_INDEX:-https://wheels.vllm.ai/nightly}"
VLLM_INSTALL_ARGS="${VLLM_INSTALL_ARGS:---no-deps}"
TRANSFORMERS_REF="${TRANSFORMERS_REF:-main}"

echo "Building ${IMAGE_TAG} from ${BASE_IMAGE}"

docker build \
    -f Dockerfile.vllm-ngc \
    -t "${IMAGE_TAG}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg VLLM_PIP_SPEC="${VLLM_PIP_SPEC}" \
    --build-arg VLLM_WHEEL_INDEX="${VLLM_WHEEL_INDEX}" \
    --build-arg VLLM_INSTALL_ARGS="${VLLM_INSTALL_ARGS}" \
    --build-arg TRANSFORMERS_REF="${TRANSFORMERS_REF}" \
    "$@" \
    .

echo ""
echo "Build complete."
echo "Run ./run-vllm-docker.sh to start the server."
