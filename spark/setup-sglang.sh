#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

require_command uv

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/.venv-sglang}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
SGLANG_SPEC="${SGLANG_SPEC:-git+https://github.com/sgl-project/sglang.git#subdirectory=python}"

echo "Creating SGLang venv at ${VENV_PATH} with ${PYTHON_BIN}"
uv venv --python "${PYTHON_BIN}" "${VENV_PATH}"

echo "Installing CUDA 13 PyTorch stack to match the working containers closely..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    --index-url "${TORCH_INDEX_URL}" \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

echo "Installing current SGLang code without its pinned wheel dependencies..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade --no-deps "${SGLANG_SPEC}"

echo "Installing a container-like SGLang runtime stack..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    IPython \
    aiohttp \
    'apache-tvm-ffi>=0.1.5,<0.2' \
    'anthropic>=0.20.0' \
    blobfile==3.0.0 \
    build \
    compressed-tensors \
    cuda-python==13.1.1 \
    datasets \
    decord2 \
    einops \
    fastapi \
    flashinfer-cubin==0.6.5 \
    flashinfer-python==0.6.5 \
    gguf \
    grpcio==1.78.0 \
    grpcio-health-checking==1.78.0 \
    grpcio-reflection==1.78.0 \
    hf_transfer \
    huggingface_hub \
    interegular \
    llguidance==0.7.30 \
    modelscope \
    msgspec \
    ninja \
    numpy \
    'nvidia-cutlass-dsl>=4.3.4' \
    nvidia-ml-py \
    openai==2.6.1 \
    openai-harmony==0.0.4 \
    orjson \
    outlines==0.1.11 \
    packaging \
    partial-json-parser==0.2.1.1.post7 \
    pillow \
    'prometheus-client>=0.20.0' \
    psutil \
    py-spy \
    pybase64 \
    pydantic \
    python-multipart \
    'pyzmq>=25.1.2' \
    quack-kernels==0.2.4 \
    requests \
    scipy \
    sentencepiece \
    setproctitle \
    sgl-kernel==0.3.21 \
    smg-grpc-proto==0.4.2 \
    soundfile==0.13.1 \
    tiktoken \
    timm==1.0.16 \
    torchao==0.9.0 \
    tqdm \
    transformers==5.3.0 \
    triton==3.6.0 \
    uvicorn \
    uvloop \
    xgrammar==0.1.32

echo "Reapplying the CUDA 13 PyTorch stack after SGLang dependency resolution..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade --reinstall \
    --index-url "${TORCH_INDEX_URL}" \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

echo "Ensuring CUDA 13 user-space wheels are present for a self-contained venv..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    nvidia-cublas==13.1.0.3 \
    nvidia-cuda-cupti==13.0.85 \
    nvidia-cuda-nvrtc==13.0.88 \
    nvidia-cuda-runtime==13.0.96 \
    nvidia-cudnn-cu13==9.15.1.9 \
    nvidia-cufft==12.0.0.61 \
    nvidia-cufile==1.15.1.6 \
    nvidia-curand==10.4.0.35 \
    nvidia-cusolver==12.0.4.66 \
    nvidia-cusparse==12.6.3.3 \
    nvidia-cusparselt-cu13==0.8.0 \
    nvidia-nccl-cu13==2.28.9 \
    nvidia-nvjitlink==13.0.88 \
    nvidia-nvshmem-cu13==3.4.5 \
    nvidia-nvtx==13.0.85

echo "Adding CUDA 12 compatibility libs required by the current sgl-kernel wheel..."
uv pip install --python "${VENV_PATH}/bin/python" --upgrade \
    nvidia-cuda-runtime-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cudnn-cu12==9.16.0.29

configure_cuda_env "${VENV_PATH}/bin/python"

echo "Verifying SGLang environment..."
"${VENV_PATH}/bin/python" - <<'PY'
from importlib.metadata import version
import torch

print("python =", __import__("sys").version.split()[0])
print("torch =", torch.__version__)
print("torch.cuda =", torch.version.cuda)
print("cudnn =", torch.backends.cudnn.version())
print("cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device =", torch.cuda.get_device_name(0))
    print("capability =", torch.cuda.get_device_capability(0))
print("sglang =", version("sglang"))
print("sgl-kernel =", version("sgl-kernel"))
print("transformers =", version("transformers"))
print("triton =", version("triton"))
print("flashinfer-python =", version("flashinfer-python"))
PY
