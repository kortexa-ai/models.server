# DGX Spark Architecture & Caveats

## Hardware

- **GPU:** NVIDIA GB10 (Blackwell architecture, sm_121, compute capability 12.1)
- **CUDA:** 13.0 (driver 580.126.09+)
- **Platform:** aarch64 (ARM64), Ubuntu 24.04
- **DGX Spark Version:** 7.4.0
- **Memory:** 128GB unified (shared CPU/GPU)

## Qwen 3.5 Family

Five variants, all sharing the same `Qwen3_5ForConditionalGeneration` architecture:

| Model | Port | GGUF Repo | Quantization |
|-------|------|-----------|-------------|
| qwen-3.5-2b | 2030 | unsloth/Qwen3.5-2B-GGUF | Q8_0 |
| qwen-3.5-4b | 2029 | unsloth/Qwen3.5-4B-GGUF | Q8_0 |
| qwen-3.5-9b | 2024 | unsloth/Qwen3.5-9B-GGUF | Q8_0 |
| qwen-3.5-27b | 2026 | unsloth/Qwen3.5-27B-GGUF | Q4_K_M |
| qwen-3.5-35b-a3b | 2027 | unsloth/Qwen3.5-35B-A3B-GGUF | UD-Q8_K_XL |

**Important:** Qwen 3.5 (`Qwen3_5ForConditionalGeneration`) is a different architecture from Qwen 3 (`Qwen3ForCausalLM`). Not all engines support it yet — see engine-specific docs.

---

## CUDA Graphs

- CUDA graphs **DO NOT work** bare-metal on GB10 (driver 580, CUDA 13.0)
- CUDA graphs **WORK** inside NVIDIA's Docker containers (e.g. `nvcr.io/nvidia/vllm:26.01-py3`)
- Container uses CUDA 13.1 forward compatibility (driver 590.48.01) on host kernel driver 580.126.09
- Docker flags needed: `--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864`

---

## PyTorch CUDA on Blackwell

PyTorch CUDA works on DGX Spark using cu128/cu130 nightly wheels + NVIDIA pip packages.
This is needed for projects like vision.server (YOLO, RF-DETR, ml-sharp) and for bare-metal vLLM/SGLang.

### The Problem

The DGX Spark ships CUDA 13.0, but torch cu128 links against CUDA 12 libraries.
The system's `.so.12` compat symlinks have **wrong symbol versions** (they're actually CUDA 13
libraries with `.so.12` names, but missing the CUDA 12 symbol version tags).

### The Fix

1. Install torch from a CUDA-aware nightly index:
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

### Automating in projects

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

**Important:** Use `.venv/bin/python` instead of `uv run` after install — `uv run` re-resolves
dependencies and can downgrade cu128 torch back to CPU torch from PyPI.

### What does NOT fix the problem

- System updates / driver updates (CUDA 13 system libs will never have CUDA 12 symbol versions)
- `cuda-compat-12-8` apt package (30KB shim for Tesla forward compat, not full libraries)
- `libcudart.so.12` symlink to `.so.13` (wrong symbol versions, not just wrong soname)

---

## GPU Memory Management

- Stale vLLM EngineCore processes don't die when the parent is killed
- Must `pkill -9 -f "VLLM"` before starting new instances
- Docker can leave root-owned cache files in `~/.cache/vllm/` and `~/.cache/huggingface/` that block bare-metal runs
- Solution: bare-metal launchers should set custom cache paths (see ENGINE_VLLM.md)

---

## Environment Setup

The CUDA environment variables are set in `.bashrc` inside the Ubuntu/CUDA block (gated by
`$SH_OS_DISTRO == Ubuntu` and presence of `/usr/local/cuda`). They apply to all Ubuntu machines
with CUDA (DGX Spark, Intel+RTX 6000, etc.).

---

## Useful Links

- [NVIDIA vLLM for DGX Spark](https://build.nvidia.com/spark/vllm/instructions) — Official container instructions
- [NGC vLLM container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm)
- [DGX Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/dependencies.html)
- [dgx-spark-vllm-setup](https://github.com/eelbaz/dgx-spark-vllm-setup) — Community pip installer
- [vLLM Blackwell issue #31128](https://github.com/vllm-project/vllm/issues/31128)
- [DGX Spark ML setup guide](https://github.com/natolambert/dgx-spark-setup)
