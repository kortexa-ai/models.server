#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6}"
TRTLLM_REPO="${TRTLLM_REPO:-https://github.com/NVIDIA/TensorRT-LLM.git}"
TRTLLM_REF="${TRTLLM_REF:-main}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==5.3.0}"
TRTLLM_PRECOMPILED_LOCATION="${TRTLLM_PRECOMPILED_LOCATION:-}"
FULL_SOURCE_BUILD="${FULL_SOURCE_BUILD:-0}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-120-real}"
JOB_COUNT="${JOB_COUNT:-4}"
SAFE_REF="$(printf '%s' "${TRTLLM_REF}" | tr '/:@' '---')"
SAFE_TRANSFORMERS="$(printf '%s' "${TRANSFORMERS_SPEC}" | sed 's/[^A-Za-z0-9._-]/-/g')"
BUILD_FLAVOR="precompiled"
if [[ "${FULL_SOURCE_BUILD}" == "1" ]]; then
    BUILD_FLAVOR="sourcebuild-${CUDA_ARCHITECTURES}"
fi
TAG="${TAG:-local/trtllm-main:${SAFE_REF}-${SAFE_TRANSFORMERS}-${BUILD_FLAVOR}}"

echo "Building TensorRT-LLM main-source image"
echo "Base image: ${BASE_IMAGE}"
echo "TensorRT-LLM repo: ${TRTLLM_REPO}"
echo "TensorRT-LLM ref: ${TRTLLM_REF}"
echo "Transformers spec: ${TRANSFORMERS_SPEC}"
echo "Full source build: ${FULL_SOURCE_BUILD}"
if [[ "${FULL_SOURCE_BUILD}" == "1" ]]; then
    echo "CUDA architectures: ${CUDA_ARCHITECTURES}"
    echo "Job count: ${JOB_COUNT}"
elif [[ -n "${TRTLLM_PRECOMPILED_LOCATION}" ]]; then
    echo "Precompiled source override: ${TRTLLM_PRECOMPILED_LOCATION}"
else
    echo "Precompiled source override: auto-download matching wheel"
fi
echo "Target tag: ${TAG}"

exec docker build \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "TRTLLM_REPO=${TRTLLM_REPO}" \
    --build-arg "TRTLLM_REF=${TRTLLM_REF}" \
    --build-arg "TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC}" \
    --build-arg "TRTLLM_PRECOMPILED_LOCATION=${TRTLLM_PRECOMPILED_LOCATION}" \
    --build-arg "FULL_SOURCE_BUILD=${FULL_SOURCE_BUILD}" \
    --build-arg "CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    --build-arg "JOB_COUNT=${JOB_COUNT}" \
    -f "${SCRIPT_DIR}/Dockerfile.main-source" \
    -t "${TAG}" \
    "${SCRIPT_DIR}"
