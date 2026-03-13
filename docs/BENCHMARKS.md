# Benchmark Results Log

All benchmarks on NVIDIA DGX Spark (GB10 Blackwell, 128GB unified memory).
TPS = generation tokens per second (warm, single-request, no-reasoning unless noted).

---

## March 12, 2026

### Nemotron-3-Super 120B-A12B on DGX Spark

Architecture: NemotronHForCausalLM (Mamba-2 hybrid + LatentMoE, 120B total / 12B active).

#### llama.cpp — Q4_K_M (GGUF, unsloth, 16k context, Q4_0 KV cache)

| Metric | Value |
|--------|-------|
| Cold gen TPS | 10.9 |
| Warm gen TPS | 11.7–12.6 |
| Prompt TPS | 41–46 |
| Model size | ~82.5 GB |

#### vLLM 0.17.1rc1 Bare-Metal — NVFP4 (Marlin backend, FP8 KV cache, 32k context)

| Metric | Value |
|--------|-------|
| Warm gen TPS | 12.3–12.7 |
| GPU memory | 86.8 GB |

Notes:
- NVFP4 is trained-in quantization (not post-training), weights ~69.5 GB
- Requires Marlin env vars: `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_TEST_FORCE_FP8_MARLIN=1`, `VLLM_MARLIN_USE_ATOMIC_ADD=1`
- Uses `super_v3` reasoning parser via plugin file (not built into vLLM 0.17.1)
- Model card mandates `temperature=1.0, top_p=0.95` for all tasks
- TRT-LLM OOM'd even at batch_size=1 / 8k context — 69.5 GB model + KV cache exceeds 128 GB budget

---

## March 11, 2026

### vLLM 0.17.1rc1 Bare-Metal — Qwen 3.5 Full Sweep (BF16)

| Model | TTFT (warm) | Gen TPS | Context | Notes |
|-------|-------------|---------|---------|-------|
| Qwen3.5-0.8B | 206ms | 98.1 | 32768 | |
| Qwen3.5-4B | ~15s | 22.8 | 32768 | High TTFT under investigation |
| Qwen3.5-9B | 4.5s | 13.5 | 32768 | |
| Qwen3.5-35B-A3B | 4.3s | 17.4 | 32768 | MoE, only 3B active — faster than 9B dense |
| Qwen3.5-27B | 4.1s | 3.8 | 32768 | --enforce-eager |

### TRT-LLM PyTorch Backend — Qwen 3 (BF16, Docker Sourcebuild)

Image: `local/trtllm-main:main-transformers--5.3.0-sourcebuild-120-real`

Note: These are Qwen **3** models (not 3.5 — TRT-LLM doesn't support Qwen 3.5 yet).

| Model | TTFT (warm) | Gen TPS | TPS (overall) |
|-------|-------------|---------|---------------|
| Qwen3-0.6B | 180ms | 56.2 | 55.2 |
| Qwen3-4B | 730ms | 22.6 | 21.5 |
| Qwen3-8B | 1304ms | 14.0 | 13.6 |

---

## March 9, 2026

### vLLM Docker (`vllm/vllm-openai:cu130-nightly`) — Qwen 3.5 (BF16)

| Model | Context | Cold TPS | Warm TPS | Notes |
|-------|---------|----------|----------|-------|
| Qwen3.5-0.8B | 32768 | 2.69 | 103.77 | gpu-memory-utilization=0.25 |
| Qwen3.5-9B | 262144 | — | 12.03 | Avg of 3 runs (13.01, 13.01, 10.08) |

### vLLM Bare-Metal — Qwen 3.5 (BF16)

| Model | Context | Cold TPS | Warm TPS |
|-------|---------|----------|----------|
| Qwen3.5-0.8B | 32768 | 4.01 | 103.84 |
| Qwen3.5-9B | 262144 | 3.40 | 12.14 |

### SGLang Docker — Qwen 3.5-0.8B (BF16)

| Image | Cold TPS | Warm TPS |
|-------|----------|----------|
| scitrera/dgx-spark-sglang:0.5.9-t5 | — | 86.97 |
| scitrera/dgx-spark-sglang:0.5.9-dev1-329817e2-t5 | — | 93.11 |
| Same dev image, Q4_K_M comparison | 27.18 | 85.03 |

### SGLang Docker — Qwen 3.5-9B (BF16, 262144 ctx)

| Engine | Cold TPS | Warm TPS |
|--------|----------|----------|
| SGLang (scitrera dev) | 10.32 | 12.69 |

### llama.cpp — Qwen 3.5-0.8B

| Quant | Cold TPS | Warm TPS |
|-------|----------|----------|
| Q8_0 | 70.27 | 70.60 |
| Q4_K_M | 72.17 | 69.38 |

Server-side decode: ~72 tok/s (Q8_0), ~71 tok/s (Q4_K_M).

### llama.cpp — Qwen 3.5-9B (Q4_K_M, 262144 ctx)

| Metric | Value |
|--------|-------|
| Cold TPS | 24.39 |
| Warm TPS | 24.83 |

Server-side decode: ~25 tok/s.
Startup memory: model 4861 MiB, context 2354 MiB, compute 808 MiB → ~8023 MiB.

### 4B Quantization Sweep (vLLM Bare-Metal + llama.cpp)

| Engine / Checkpoint | Quant | Warm TPS | Valid Output |
|---------------------|-------|----------|--------------|
| llama.cpp Q4_K_M | GGUF 4-bit | 38.21 | Yes |
| vLLM AxionML NVFP4 | modelopt_fp4 | 38.45 | No (corrupted) |
| llama.cpp Q8_0 | GGUF 8-bit | 35.96 | Yes |
| vLLM olka-fi MXFP4 | compressed-tensors | 33.82 | Yes |
| vLLM QuantTrio AWQ | awq_marlin | 33.15 | Yes |

### 0.8B Engine Comparison Summary

| Engine | Warm TPS |
|--------|----------|
| vLLM (Docker or bare-metal) | ~104 |
| SGLang (Docker dev) | ~93 |
| SGLang (Docker stable) | ~87 |
| llama.cpp Q8_0 | ~71 |
| llama.cpp Q4_K_M | ~69 |

### 9B Engine Comparison Summary

| Engine | Warm TPS | Context |
|--------|----------|---------|
| llama.cpp Q4_K_M | 24.83 | 262144 |
| SGLang Docker | 12.69 | 262144 |
| vLLM bare-metal | 12.14 | 262144 |
| vLLM Docker | 12.03 | 262144 |

Note: llama.cpp wins at 9B because the GGUF Q4_K_M quantization is much more memory-efficient than BF16, allowing more room for compute. The BF16 engines (vLLM, SGLang) are constrained by the larger model footprint at this size.

---

## March 8, 2026

### SGLang Docker — Qwen 3.5-0.8B First Bring-Up

- `scitrera/dgx-spark-sglang:0.5.9-t5` successfully booted `Qwen/Qwen3.5-0.8B`
- Confirmed: weight load, KV cache allocation, CUDA graph capture, API up on port 2131
- Smoke test passed: `Reply with exactly: ok` → `ok`

### vLLM Docker — Qwen 3.5-0.8B First Bring-Up

- `vllm/vllm-openai:cu130-nightly` booted `Qwen/Qwen3.5-0.8B`
- Required `--gpu-memory-utilization 0.25` (default 0.9 fails)
- Resolved as `Qwen3_5ForConditionalGeneration`
- API up on port 2211, chat completions working
