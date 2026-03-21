# DGX Spark (GB10) Inference Guide

Everything we know about running LLMs on the NVIDIA DGX Spark (GB10, SM 12.1 Blackwell, 128GB unified memory, aarch64).

## Hardware Facts

- **GPU**: NVIDIA GB10, SM 12.1 (Blackwell class)
- **Memory**: 128GB unified (shared between CPU and GPU)
- **Usable GPU memory**: ~100-105 GiB after OS/services (check with `torch.cuda.mem_get_info()`)
- **Memory bandwidth**: 221 GB/s (vs 3.35 TB/s on H100 — memory-bound workloads dominate)
- **CUDA**: 13.0+ required (driver forward compatibility mode)

## Inference Backends

### 1. llama.cpp (llama-server)

The simplest and often fastest option for small-to-mid models with GGUF quants.

```bash
# Already installed system-wide, or:
# Build from source with CUDA + unified memory support
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=1
```

**Performance**: ~39 tok/s for Qwen 3.5 4B Q8_0

### 2. vLLM (bare-metal venv)

The `bench/.venv-vllm` venv for models that don't need Docker patches.

```bash
cd models.server/bench
./setup-vllm.sh
```

**What it installs**:
- Python 3.12 venv via `uv`
- vLLM nightly (cu130)
- transformers 5.3.0
- ninja (for FlashInfer JIT)

**Known issues**:
- FlashInfer CUTLASS SM120 kernels require JIT compilation (~35 min first time)
- Pre-compile with `./precompile-flashinfer.sh` (single-threaded, safe)
- Stale FlashInfer cache (`~/.cache/flashinfer/`) can reference old venv paths — delete and rebuild if venv moves
- The `fused_recurrent.py` Triton kernel for GDN/linear attention is unoptimized — Qwen 3.5 models run at ~3 tok/s

**Performance**: ~22 tok/s for Nemotron Nano (FlashInfer CUTLASS), ~10 tok/s (Marlin)

### 3. eugr/spark-vllm-docker

Community Docker image with SM121-specific patches. **Best option for Nemotron models.**

```bash
cd ~/src/spark-vllm-docker
./build-and-copy.sh          # Standard build (~4 min, uses prebuilt wheels)
./build-and-copy.sh --tf5    # With transformers 5.x (for INT4 AutoRound models)
```

**Image**: `vllm-node:latest` (~25GB)

**What it includes**:
- vLLM 0.17.2rc1 with SM121 patches
- Pre-compiled FlashInfer wheels for SM121
- Marlin NVFP4 backend (FP4→BF16 decompression on tensor cores)
- `fastsafetensors` for fast weight loading
- Ray for multi-node TP

**Mods** (in `mods/` directory, applied at runtime):
- `fix-qwen3-coder-next`: Crash fix + slowness fix + **Triton allocator fix** (critical for Qwen 3.5)
- `fix-qwen3.5-autoround`: RoPE `set()` type error fix for transformers 5.x
- `fix-qwen3.5-chat-template`: Custom `unsloth.jinja` chat template
- `nemotron-nano`: Downloads reasoning parser
- `nemotron-super`: Downloads reasoning parser

**Running with mods** (mount + apply inside container):
```bash
docker run --rm --privileged --gpus all --ipc host --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/src/spark-vllm-docker/mods:/workspace/mods:ro \
    -e VLLM_NVFP4_GEMM_BACKEND=marlin \
    -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
    -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    -e HF_TOKEN \
    -w /workspace/vllm \
    vllm-node \
    bash -c '
        cd /workspace/mods/fix-qwen3-coder-next && bash run.sh
        cd /workspace/vllm
        exec vllm serve MODEL_ID --port 8000 ...
    '
```

**Performance**: ~45 tok/s Nemotron Nano, ~12 tok/s Nemotron Super

**Does NOT fix**: Qwen 3.5 GDN linear attention slowness (~3 tok/s)

### 4. buildspark/vllm-gb10-mtp

Pre-built Docker image with GDN fixes + MTP speculative decoding. **Only option for fast Qwen 3.5 35B-A3B.**

```bash
docker pull buildspark/vllm-gb10-mtp
```

**Image**: `buildspark/vllm-gb10-mtp:latest` (~25GB)

**What it includes**:
- vLLM 0.17.0rc1 (commit `a3189a08b`)
- FlashInfer v0.6.1 compiled for SM121
- transformers 5.2.0
- GDN NaN guard + Triton allocator fix
- MTP speculative decoding (89-94% acceptance rate)
- SM121 Marlin MoE 128-thread fix (shared memory race condition)

**Critical notes**:
- **`--privileged` is REQUIRED** — Triton needs elevated permissions for SM121 kernel loading
- **FP8 weight quantization is broken on SM121** — use BF16 weights only
- **FP8 KV cache works fine** — `--kv-cache-dtype fp8_e4m3`
- **First boot takes 8-10 min** (torch.compile + FlashInfer autotuning + CUDA graphs), cached after
- **`HF_HUB_DISABLE_XET=1` required** — broken xet downloader in base image
- **`VLLM_USE_FLASHINFER_MOE_FP8=0` required** — FlashInfer FP8 MoE not supported on SM121
- **RoPE fix needed for 27B**: Apply `fix-qwen3.5-autoround` transformers patch (set() type error)
- **Needs ~100 GiB free GPU memory** — stop other models/services first

**Reference command** (from buildspark README):
```bash
docker run --privileged --gpus all --ipc=host --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e VLLM_USE_FLASHINFER_MOE_FP8=0 \
    -e VLLM_FLASHINFER_MOE_BACKEND=latency \
    -e HF_HUB_DISABLE_XET=1 \
    -e HF_TOKEN \
    buildspark/vllm-gb10-mtp \
    vllm serve Qwen/Qwen3.5-35B-A3B \
        --served-model-name qwen3.5:35b-a3b \
        --port 8000 --max-num-seqs 12 \
        --gpu-memory-utilization 0.85 \
        --enable-prefix-caching \
        --max-num-batched-tokens 4096 \
        --kv-cache-dtype fp8_e4m3 \
        --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
        --host 0.0.0.0
```

**Performance**: ~20 tok/s Qwen 3.5 35B-A3B (BF16+MTP+FP8 KV)

**Does NOT fix**: Qwen 3.5 27B dense model (~3 tok/s — GDN fix only covers MoE model class)

### 5. TensorRT-LLM (Docker)

NVIDIA's inference engine. Multiple build paths in `trtllm.spark/`.

**Pre-built NGC image** (limited model support):
```bash
# NGC 1.3.0rc6 — transformers 4.57.1 (too old for Qwen 3.5)
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6
```

**Source-built image** (Qwen 3.5 support):
```bash
cd models.server/trtllm.spark
./build-main-image.sh    # Precompiled overlay (~15 min)
# or
FULL_SOURCE_BUILD=1 ./build-main-image.sh  # From source (~2-3 hrs)
```

Produces: `local/trtllm-main:main-transformers--5.3.0-sourcebuild-120-real` (~48GB)

**Performance**: ~36 tok/s Nemotron Nano

### 6. SGLang

Alternative to vLLM. Setup:
```bash
cd models.server/bench
./setup-sglang.sh
```

Creates `.venv-sglang` with CUDA 13 PyTorch stack + dual CUDA 12 compat libs.

**Status**: Installed but not extensively benchmarked in current session.

## FlashInfer Kernel Pre-compilation

NVFP4 models on bare-metal vLLM require FlashInfer CUTLASS SM120 kernels. These JIT-compile on first use and can OOM or freeze the machine.

**Pre-compile safely** (single-threaded, ~35 min total):
```bash
cd models.server
./precompile-flashinfer.sh
```

Compiles and caches at `~/.cache/flashinfer/`:
- FP4 GEMM (CUTLASS SM120) — 1s if cached, ~5 min first time
- Fused MOE (CUTLASS SM120) — ~27 min first time
- FP8 GEMM (CUTLASS SM120) — ~8 min first time

**When to re-run**: After clearing `~/.cache/flashinfer/`, upgrading vLLM/FlashInfer, or moving the venv.

## Model Performance Summary (Single DGX Spark)

| Model | Size | Backend | Quant | tok/s | Notes |
|-------|------|---------|-------|------:|-------|
| Nemotron 3 Nano 30B-A3B | 3.5B active | eugr Docker + Marlin | NVFP4 | **45** | Best overall |
| Qwen 3.5 4B | 4B dense | llama-server | GGUF Q8 | **39** | Simple, reliable |
| Nemotron 3 Nano 30B-A3B | 3.5B active | TRT-LLM | NVFP4 | 36 | |
| Qwen 3.5 35B-A3B | 3B active | buildspark Docker | BF16+MTP | **20** | Needs ~100GB free, --privileged |
| Nemotron 3 Super 120B-A12B | 12B active | eugr Docker + Marlin | NVFP4 | **12** | |
| Qwen 3.5 27B | 27B dense | buildspark Docker | BF16 | 3 | GDN kernel bottleneck |

## Known Issues & Gotchas

### Qwen 3.5 GDN Linear Attention
All Qwen 3.5 models (27B, 35B-A3B) use hybrid GDN (Gated Delta Net) attention layers. The `fused_recurrent.py` Triton kernel is unoptimized on SM121, causing ~30x slowdown. Only `buildspark/vllm-gb10-mtp` has partial fixes (for the 35B MoE model class only).

### SM121 vs SM120
GB10 reports SM 12.1 but many tools only recognize SM 12.0. This causes:
- Fallback to generic kernels in FlashInfer, CUTLASS, Triton
- `--privileged` needed for Triton to load custom kernels
- Marlin MoE 256-thread kernel has shared memory race at TP=1 (128-thread fix in buildspark/namake-taro)

### Unified Memory Gotchas
- `torch.cuda.mem_get_info()` reports GPU-visible memory, not system total
- Models can oversubscribe GPU memory (pages to system RAM) but thrash badly
- The `gpu-memory-utilization` flag is checked against free memory at startup, not total
- Kill all other model containers before starting large models

### Thinking Token Overhead (Qwen 3.5)
Qwen 3.5 generates verbose "Thinking Process:" reasoning by default. On older vLLM (0.17.0rc1), `chat_template_kwargs` and `enable_thinking` extra body params are NOT supported. Workaround: use a custom chat template with `{%- set enable_thinking = false -%}` prepended.

### Docker `--privileged`
Required for buildspark image (Triton SM121 kernels). NOT required for eugr's vllm-node or TRT-LLM images.

### HuggingFace Hub
- `HF_HUB_DISABLE_XET=1` needed in buildspark image (broken xet downloader)
- `huggingface_hub >= 1.6.0` breaks gpt-oss models — pin to 1.5.0 if needed

## Docker Images Inventory

| Image | Size | Purpose | Built |
|-------|------|---------|-------|
| `vllm-node:latest` | 25.5GB | eugr spark-vllm-docker | 2026-03-20 |
| `buildspark/vllm-gb10-mtp:latest` | 25GB | Qwen 3.5 + MTP | 2026-03-15 |
| `local/trtllm-main:...-sourcebuild-120-real` | 48.2GB | TRT-LLM from source | 2026-03-10 |
| `local/trtllm-main:...-5.3.0` | 39.4GB | TRT-LLM precompiled overlay | 2026-03-09 |
| `vllm/vllm-openai:cu130-nightly` | 20.1GB | Stock vLLM nightly | 2026-03-08 |
| `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6` | 33.9GB | NGC stock TRT-LLM | 2026-02-26 |

## External Resources

- **eugr/spark-vllm-docker**: https://github.com/eugr/spark-vllm-docker
- **buildspark/vllm-gb10-mtp**: https://hub.docker.com/r/buildspark/vllm-gb10-mtp (source: https://github.com/buildsparklabs/vllm-gb10-mtp)
- **namake-taro MXFP4 patches**: https://github.com/namake-taro/vllm-custom
- **NVIDIA DGX Spark forums**: https://forums.developer.nvidia.com/c/ai-data-science/dgx-spark/
- **Intel AutoRound quants**: https://huggingface.co/Intel (search for `int4-AutoRound`)
