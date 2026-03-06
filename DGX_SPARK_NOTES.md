# Running LLM Models on NVIDIA DGX Spark (GB10 Blackwell)

## Hardware
- **GPU:** NVIDIA GB10 (Blackwell architecture, sm_121, compute capability 12.1)
- **CUDA:** 13.0 (driver 580.126.09+)
- **Platform:** aarch64 (ARM64), Ubuntu 24.04
- **DGX Spark Version:** 7.4.0

---

## Qwen 3.5 Family

Five variants, all sharing the same `Qwen3_5ForConditionalGeneration` architecture:

| Model | Port | GGUF Repo | Quantization |
|-------|------|-----------|-------------|
| qwen-3.5-2b | 2030 | unsloth/Qwen3.5-2B-GGUF | Q8_0 |
| qwen-3.5-4b | 2029 | unsloth/Qwen3.5-4B-GGUF | Q8_0 |
| qwen-3.5-9b | 2024 | unsloth/Qwen3.5-9B-GGUF | Q8_0 |
| qwen-3.5-27b | 2026 | unsloth/Qwen3.5-27B-GGUF | Q4_K_M |
| qwen-3.5-35b-a3b | 2027 | unsloth/Qwen3.5-35B-A3B-GGUF | UD-Q8_K_XL |

### What's Working: llama-server (llama.cpp)

All five models run via **llama-server** with GGUF quantizations. It works but is slow — no
continuous batching means one request at a time, so concurrent users queue up.

- Built from source with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native`
- Also needs `-DLLAMA_OPENSSL=ON` for HuggingFace downloads via `-hf` flag
- Build script: `api.server/llama.cpp/build-llama.sh` (auto-detects platform)
- OpenAI-compatible API with `--jinja` flag
- Performance: ~527 tok/s prompt, ~77 tok/s generation (Qwen3.5-2B Q8_0)

### What's NOT Working: vLLM

We want vLLM for continuous batching and better concurrent throughput. Two approaches tried, both blocked:

#### vLLM via pip (bare-metal)
- PyPI wheels: Only CPU torch on aarch64. No Blackwell GPU support.
- Even with torch cu128 nightly, vLLM 0.16 doesn't support `Qwen3_5ForConditionalGeneration`
- Building from source fails (Triton cmake error with LLVM on aarch64)
- **Dead end on bare-metal.**

#### NVIDIA Official vLLM Docker Container
Tested both `nvcr.io/nvidia/vllm:26.01-py3` and `nvcr.io/nvidia/vllm:26.02-py3`.

The 26.02 container ships:
- **vLLM 0.15.1** — does NOT have Qwen 3.5 model support. The model files in `vllm/model_executor/models/` include `qwen3.py`, `qwen3_moe.py`, `qwen3_next.py`, but no `qwen3_5.py`. The `Qwen3_5ForConditionalGeneration` architecture is simply not registered.
- **transformers 4.57.5** — knows about Qwen 3.5, but vLLM's model registry doesn't.

The container works fine for Qwen 3 and older models (confirmed with `Qwen/Qwen2.5-Math-1.5B-Instruct`), but cannot load any Qwen 3.5 variant.

**Blocked until NVIDIA ships a newer container with vLLM >= 0.17+ that adds Qwen 3.5 support, or we find a way to upgrade vLLM inside the container without breaking CUDA/Blackwell compatibility.**

### Next Steps: Building vLLM from Source

The key insight: **vLLM `main` branch has `qwen3_5.py`** — the model is supported in latest source.
The problem is getting vLLM to compile on DGX Spark (aarch64 + Blackwell + CUDA 13.0).

#### Community Resources

1. **[dgx-spark-vllm-setup](https://github.com/eelbaz/dgx-spark-vllm-setup)** — Build script that compiles Triton + vLLM from source with Blackwell patches. Default targets vLLM commit `66a168a19` (v0.11.1rc3, too old for Qwen 3.5). We're running it with `--vllm-version main` to get latest vLLM. **Status: IN PROGRESS** — Triton compiling (needs `python3.12-dev` apt package installed first, otherwise cmake fails with "Could NOT find Python3 (missing: Development.Module)"). Installs PyTorch 2.10.0+cu130.

2. **[vllm-dgx-spark](https://github.com/mark-ramsey-ri/vllm-dgx-spark)** — Docker wrapper around NVIDIA's official container (`nvcr.io/nvidia/vllm:25.11-py3`). Adds nice multi-node orchestration with InfiniBand/Ray, but uses the same old vLLM that lacks Qwen 3.5. **Not useful for our case** — same underlying problem as the NVIDIA container.

3. **[Installing vLLM on DGX Spark from Source](https://medium.com/@anveshkumarchavidi/installing-vllm-on-nvidia-dgx-spark-from-source-4dde137ff3ef)** — Step-by-step Medium guide. Haven't tried yet.

---

## Legacy Findings

### LLM Serving Options (General)

#### vLLM via NVIDIA Docker Container (for supported models)
NVIDIA provides a pre-built vLLM container for DGX Spark that handles all CUDA/Blackwell compatibility:
```bash
export LATEST_VLLM_VERSION=26.02-py3
docker pull nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION}
docker run -it --gpus all -p 8000:8000 \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} \
    vllm serve "Qwen/Qwen2.5-Math-1.5B-Instruct"
```
- OpenAI-compatible API
- Continuous batching for concurrent requests
- Supports Qwen3, Llama-3.x, Phi-4, Nemotron3
- Two-Spark tensor parallelism for 70B+ models
- See: https://build.nvidia.com/spark/vllm/instructions

#### SGLang via pip
- torch cu126: missing sm_121 kernels
- torch cu128: transformers version conflicts
- Fundamentally blocked by PyTorch + Blackwell kernel gap in pip wheels

### PyTorch CUDA on Blackwell — SOLVED

PyTorch CUDA works on DGX Spark using cu128 nightly wheels + NVIDIA pip packages.
This is needed for projects like vision.server (YOLO, RF-DETR, ml-sharp) — NOT for LLM serving.

#### The Problem
The DGX Spark ships CUDA 13.0, but torch cu128 links against CUDA 12 libraries.
The system's `.so.12` compat symlinks have **wrong symbol versions** (they're actually CUDA 13
libraries with `.so.12` names, but missing the CUDA 12 symbol version tags).

#### The Fix
1. Install torch from the nightly cu128 index:
```bash
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

2. Install the CUDA 12 compatibility libraries via pip:
```bash
uv pip install nvidia-cufft-cu12 nvidia-nccl-cu12 nvidia-nvshmem-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12
```

These provide the actual CUDA 12 `.so` files that torch was built against:
| pip package | provides | why needed |
|---|---|---|
| nvidia-cufft-cu12 | libcufft.so.11 | FFT operations |
| nvidia-nccl-cu12 | libnccl.so.2 | multi-GPU communication |
| nvidia-nvshmem-cu12 | libnvshmem_host.so.3 | shared memory |
| nvidia-cuda-runtime-cu12 | libcudart.so.12 | correct symbol versions |
| nvidia-cuda-cupti-cu12 | libcupti.so.12 | profiling/tracing |

Torch also auto-installs: `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cusparselt-cu12`

#### Automating in projects
Add a `cuda-compat` optional dependency group to `pyproject.toml`:
```toml
[project.optional-dependencies]
cuda-compat = [
    "nvidia-cufft-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvshmem-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-cupti-cu12",
]
```
Then in `setup.sh`, detect GPU and install accordingly:
```bash
if [[ "$OS" == "Linux" ]] && command -v nvidia-smi &> /dev/null; then
    uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
    uv pip install -e '.[cuda-compat]'
else
    uv pip install -e .
fi
```
**Important:** Use `.venv/bin/python` instead of `uv run` after install — `uv run` re-resolves
dependencies and can downgrade cu128 torch back to CPU torch from PyPI.

#### What does NOT fix the problem
- System updates / driver updates (CUDA 13 system libs will never have CUDA 12 symbol versions)
- `cuda-compat-12-8` apt package (30KB shim for Tesla forward compat, not full libraries)
- `libcudart.so.12` symlink to `.so.13` (wrong symbol versions, not just wrong soname)

### llama.cpp Build Notes
- Build script: `api.server/llama.cpp/build-llama.sh`
- Uses `CMAKE_CUDA_ARCHITECTURES=native` which auto-detects `121a-real` for Blackwell
- Do NOT hardcode architecture numbers — Blackwell uses `121a` suffix (not just `121`)
- The script handles Mac (Metal), Intel Linux (CUDA), DGX Spark (CUDA), and Pi (CPU-only)
- Docker fallback: `docker-build-llama.sh` for Ubuntu 25.x / glibc >= 2.41

### CUDA Graphs — Docker Workaround
- CUDA graphs **DO NOT work** bare-metal on GB10 (driver 580, CUDA 13.0)
- CUDA graphs **WORK** inside NVIDIA's Docker container (`nvcr.io/nvidia/vllm:26.01-py3`)
- Container uses CUDA 13.1 forward compatibility (driver 590.48.01) on host kernel driver 580.126.09
- Docker flags needed: `--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864`

### Environment Setup

The CUDA environment variables are set in `.bashrc` inside the Ubuntu/CUDA block (gated by
`$SH_OS_DISTRO == Ubuntu` and presence of `/usr/local/cuda`). They apply to all Ubuntu machines
with CUDA (DGX Spark, Intel+RTX 6000, etc.).

### Useful Links
- [NVIDIA vLLM for DGX Spark](https://build.nvidia.com/spark/vllm/instructions) - Official container instructions
- [NGC vLLM container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm)
- [DGX Spark porting guide - dependencies](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/dependencies.html)
- [dgx-spark-vllm-setup](https://github.com/eelbaz/dgx-spark-vllm-setup) - Community pip installer (Triton build may fail)
- [vLLM Blackwell issue #31128](https://github.com/vllm-project/vllm/issues/31128)
- [DGX Spark ML setup guide](https://github.com/natolambert/dgx-spark-setup)
