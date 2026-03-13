# vLLM

## Current State

**Bare-metal: WORKING** — All Qwen 3.5 sizes verified (0.8B, 4B, 9B, 35B-A3B, 27B).

### Environment

- Python 3.12.3
- `vllm 0.17.1rc1` (cu130 nightly)
- `torch 2.10.0+cu130`
- `transformers 5.3.0`
- `triton 3.6.0`
- `flashinfer-python 0.6.4`

### Setup & Run

```bash
# Setup
./bench/setup-vllm.sh           # Creates bench/.venv-vllm

# Run
./bench/run-vllm.sh 0.8b        # Uses model resolver for short names
./bench/run-vllm.sh 27b         # Works for all sizes
```

Setup also runs from the main `./setup.sh` on Linux.

### Operational Notes

- Kill stale processes before starting new instances: `pkill -9 -f "VLLM"`
- For large models (27B, 35B-A3B), use `CONTEXT_LENGTH=32768` and `--enforce-eager`
- Default `gpu-memory-utilization` is 0.9; for 35B-A3B use `GPU_MEMORY_UTILIZATION=0.85`

---

## Nemotron-3-Super 120B-A12B (NVFP4)

Dedicated launcher: `nemotron-3-super-120b-a12b/run-vllm.sh`

This model uses NVIDIA's trained-in NVFP4 quantization (~69.5 GB weights) with the Marlin dequant backend. Requires special env vars and a custom reasoning parser plugin.

### Key Configuration

| Setting | Value |
|---------|-------|
| Model | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` |
| KV cache dtype | `fp8` (halves KV memory — critical on 128 GB) |
| Attention backend | `TRITON_ATTN` |
| Reasoning parser | `super_v3` (via `--reasoning-parser-plugin`) |
| Tool call parser | `qwen3_coder` |
| Context length | 32768 |
| GPU memory utilization | 0.7 |
| Total GPU memory | ~86.8 GB |
| Gen TPS | 12.3–12.7 |

### Required Env Vars (Marlin NVFP4)

```bash
export VLLM_NVFP4_GEMM_BACKEND=marlin
export VLLM_TEST_FORCE_FP8_MARLIN=1
export VLLM_MARLIN_USE_ATOMIC_ADD=1
```

### Reasoning Parser Plugin

vLLM 0.17.1 doesn't include `super_v3` — it's loaded via `--reasoning-parser-plugin` pointing to `super_v3_reasoning_parser.py` (downloaded from the HF model repo). This parser extends DeepSeekR1 to handle Nemotron's thinking-off mode properly.

### HF Cache Ownership

Docker runs leave root-owned files in `~/.cache/huggingface`. Fix before bare-metal runs:

```bash
sudo find ~/.cache/huggingface -not -user $(whoami) -exec chown $(whoami):$(whoami) {} +
```

---

## Docker Setup

### Working Images

| Image | vLLM | Status |
|-------|------|--------|
| `vllm/vllm-openai:cu130-nightly` | 0.17.0rc1.dev164+ | Working for Qwen 3.5 |
| `nvcr.io/nvidia/vllm:26.02-py3` | 0.15.1 | Working for Qwen 3 only — no Qwen 3.5 support |
| `nvcr.io/nvidia/vllm:26.01-py3` | ~0.15 | Working for Qwen 3 only |

### Docker Run Example

```bash
docker run --rm --name vllm-qwen35-08b --gpus all --network host --ipc host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint vllm vllm/vllm-openai:cu130-nightly \
  serve Qwen/Qwen3.5-0.8B \
  --host 0.0.0.0 --port 2211 \
  --served-model-name qwen-3.5-0.8b \
  --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.25
```

Note: default `--gpu-memory-utilization 0.9` fails startup on this machine — reduce to 0.25 for Docker.

---

## Bare-Metal Setup Details

### Cache Path Fix

Docker leaves root-owned cache files that block bare-metal runs. The launcher forces:

- `VLLM_CACHE_ROOT=bench/.cache/vllm`
- `VLLM_CONFIG_ROOT=bench/.config/vllm`
- `TORCHINDUCTOR_CACHE_DIR=bench/.cache/vllm/torch_compile_cache`
- `HF_HOME=bench/.cache/huggingface`
- `HF_HUB_CACHE=bench/.cache/huggingface/hub`
- `TRANSFORMERS_CACHE=bench/.cache/huggingface/transformers`

### `bench/setup-vllm.sh`

Creates venv at `bench/.venv-vllm`:
1. Installs vLLM nightly for CUDA 13 from `wheels.vllm.ai/nightly/cu130`
2. Upgrades transformers to 5.3.0
3. Installs ninja (needed for FlashInfer JIT kernel compilation)
4. Configures CUDA env
5. Verifies installation

### `bench/run-vllm.sh`

Launches vLLM with model resolver. Prepends venv `bin/` to PATH so FlashInfer JIT subprocesses find ninja.

---

## Bare-Metal Attempt History

### Early Attempts (Pre-March 9)

**vLLM via pip:**
- PyPI wheels: Only CPU torch on aarch64. No Blackwell GPU support.
- Even with torch cu128 nightly, vLLM 0.16 doesn't support `Qwen3_5ForConditionalGeneration`
- Building from source fails (Triton cmake error with LLVM on aarch64)

**eelbaz/dgx-spark-vllm-setup (from-source build):**

With `--vllm-version main`:
- Build: SUCCESS (with fixes — needs `python3.12-dev`, `MAX_JOBS=4`)
- Runtime: FAILS — EngineCore child process dies silently after NCCL/distributed init
- Same failure with `--enforce-eager`, with `VLLM_ENGINE_ITERATION_TIMEOUT_S=600`, with OPT-125M
- Root cause likely: `flashinfer-cubin` not available for Blackwell, JIT compilation segfaults

With default version (v0.11.1rc3):
- Build: FAILS — `flashinfer-python==0.4.1` broken `pyproject.toml` metadata

**mark-ramsey-ri/vllm-dgx-spark:**
- Docker wrapper around `nvcr.io/nvidia/vllm:25.11-py3` — too old for Qwen 3.5

### March 9 Breakthrough

Fresh bare-metal vLLM (cu130 nightly wheel) working for `Qwen/Qwen3.5-0.8B`:
- `/health` returned 200
- `/v1/models` returned `qwen-3.5-0.8b`
- Real `POST /v1/chat/completions` completed successfully

---

## 4B Quantization Sweep (March 9)

Tested multiple 4-bit quantized checkpoints with bare-metal vLLM:

| Checkpoint | Method | TPS (warm) | Output Quality |
|------------|--------|------------|----------------|
| AxionML/Qwen3.5-4B-NVFP4 | modelopt_fp4 | 38.45 | BROKEN (corrupted text) |
| QuantTrio/Qwen3.5-4B-AWQ | awq_marlin | 33.15 | Valid |
| olka-fi/Qwen3.5-4B-MXFP4 | compressed-tensors | 33.82 | Valid (not native FP4) |
| Intel/Qwen3.5-4B-int4-AutoRound | — | — | Failed (tokenizer error) |
| osoleve/Qwen3.5-4B-Base-Text-NVFP4 | modelopt | — | Failed (Qwen3_5TextConfig mismatch) |

Notes:
- Axion NVFP4 produces corrupted output even with `--enforce-eager` and transformers 5.3.0
- AWQ Marlin is the best working 4-bit option but slower than llama.cpp Q4_K_M
- olka-fi MXFP4 uses weight-only FP4 compression via Marlin, not native Blackwell FP4
