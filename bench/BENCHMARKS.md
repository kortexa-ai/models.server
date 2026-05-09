# Benchmark Results Log

Each entry tags its machine. Early entries are **sparky** (DGX Spark GB10 Blackwell, 128 GB unified). Later entries are **smarty** (RTX PRO 6000 Blackwell, 96 GB VRAM).
TPS = generation tokens per second (warm, single-request, no-reasoning unless noted).

> **Note to future Claude:** not every bench run lands here — some live only in chat transcripts. If the user asks about speeds and the answer isn't in this file, extensively grep `~/.claude/projects/` on **both smarty and snappy** (over ssh) for unlogged TPS numbers before answering. Then offer to roll any findings into this file.

---

## April 22–24, 2026 — smarty (RTX PRO 6000 Blackwell)

### Qwen 3.6 bring-up sweep + Gemma 4 retest (llama.cpp UD-Q4_K_XL, q8_0 KV, n=4)

| Model | Engine | Quant | Gen TPS |
|-------|--------|-------|---------|
| Qwen 3.6 35B-A3B PRISM-NVFP4 | vLLM Marlin | NVFP4 | **195.2** |
| Qwen 3.6 35B-A3B | llama.cpp | UD-Q4_K_XL | **187.1** |
| Qwen 3.5 35B-A3B | llama.cpp | UD-Q4_K_XL | 181.2 |
| Qwen 3.6 35B-A3B PRISM-NVFP4 | vLLM FlashInfer CUTLASS | NVFP4 | 174.6 |
| Qwen 3.6 35B-A3B PRISM-NVFP4 | vLLM 0.19.2 + flashinfer 0.6.8 + torch 2.11 | NVFP4 | 174.4 |
| Gemma 4 26B-A4B | llama.cpp | UD-Q4_K_XL | 164.5 |
| Qwen 3.6 27B (dense) | llama.cpp | UD-Q4_K_XL | 68.0 |
| Gemma 4 31B (dense) | llama.cpp | UD-Q4_K_XL | 59.0 |

**Notes:**
- MoE wins on raw throughput at total-param scale — only ~3-4B active per token vs 27-31B for dense.
- Gemma 4 31B at UD-Q4_K_XL was retested later and hit ~64 tok/s (vs 59 in the sweep, vs 42 at Q8_0 on Apr 5).
- PRISM-NVFP4 vLLM Marlin was the peak; FlashInfer CUTLASS and the bumped vLLM 0.19.2 / flashinfer 0.6.8 / torch 2.11 stack landed lower (~175).

---

## April 5, 2026 — smarty (RTX PRO 6000 Blackwell)

### Gemma 4 family bring-up (llama.cpp Q8_0)

| Model | Type | Active params | Gen TPS | Prompt TPS |
|-------|------|---------------|---------|------------|
| Gemma 4 E2B | dense (PLE) | ~2B | **250** | 2,793 |
| Gemma 4 26B-A4B | MoE | 4B | **171** | 1,387 |
| Gemma 4 31B | dense | 31B | **42** | 646 |

**Notes:**
- E2B on Q8_0 is an absolute screamer — model fits in L2-ish, no routing overhead.
- 26B-A4B / 31B ratio (171 vs 42 ≈ 4x) lines up with active-params (4B vs 31B).
- vLLM with batching hit ~324 tok/s aggregate on 26B-A4B (chat-side note, not formal sweep).
- Gemma 4 E4B never got a clean bench in the transcripts.
- Gemma 4 uses 512-token sliding window attention — llama.cpp can't reuse KV across requests, so prompt processing eats a re-process penalty. Generation TPS is unaffected.

---

## March 21, 2026 — sparky (DGX Spark)

### vLLM Docker (`vllm-node:latest`) — Qwen 3.5-35B-A3B (Intel AutoRound int4)

**BREAKTHROUGH:** After extensive investigation, achieved **50 tok/s** (target was 50-79 tok/s).

| Metric | Value |
|--------|-------|
| Warm gen TPS (avg) | **50.2** |
| TPS range | 46.2 - 51.6 |
| Model size | 19.3 GiB |
| KV cache | 60+ GiB available |
| Context | 32768 |
| Quantization | Intel AutoRound int4 (GPTQ-compatible) |

**Configuration:**
```bash
docker run --rm --name vllm-test-35b \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm-node:latest \
  Intel/Qwen3.5-35B-A3B-int4-AutoRound \
  --port 2242 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.7 \
  --load-format fastsafetensors \
  --kv-cache-dtype fp8
```

**Key findings:**
- vLLM 0.17.2rc1.dev7 has proper SM121 (Blackwell) support
- FlashInfer attention backend works correctly on DGX Spark
- CUDA graphs captured successfully (51 PIECEWISE + 35 FULL)
- FP8 KV cache critical for memory bandwidth on unified memory

**Comparison with previous attempts:**

| Configuration | TPS | Notes |
|--------------|-----|-------|
| vLLM 0.17.1rc1 bare-metal BF16 | 17.4 | Baseline, slow |
| vLLM 0.17.2rc1 Docker + AutoRound int4 | **50.2** | 3x improvement! |
| Target (community reports) | 50-79 | Achieved! |

**Remaining optimization opportunities:**
- Apply MXFP4 patches → could reach 60-70 tok/s
- Use TP=2 (if 2 GPUs available) → could double throughput
- Kill competing processes → could increase to 0.85 memory utilization

---

### vLLM Docker (`vllm-node:latest`) — Qwen 3.5 Full Sweep

**Date:** March 22, 2026

Complete benchmark of all Qwen 3.5 sizes in both BF16 and int4:

| Model | BF16 TPS | int4 TPS | Speedup | BF16 Mem | int4 Mem |
|-------|----------|----------|---------|----------|----------|
| Qwen3.5-4B | 20.6 | **43.9** | 2.1x | 8.6 GiB | 3.7 GiB |
| Qwen3.5-9B | 12.3 | **32.7** | 2.7x | 17.7 GiB | 8.1 GiB |
| Qwen3.5-27B | 2.5 | **12.6** | 5.0x | ~55 GiB | 17.6 GiB |
| Qwen3.5-35B-A3B | 17.4 | **50.2** | 2.9x | ~67 GiB | 19.3 GiB |

**Key findings:**
1. **int4 quantization helps all models** - 2-5x speedup depending on size
2. **27B shows biggest gain (5x)** - most memory-bound model benefits most
3. **35B MoE int4 is fastest** - despite being "largest", MoE only activates 3B params
4. **BF16 for large models is unusable** - 27B crawls at 2.5 tok/s

**Recommendation:** Always use int4 quantization on DGX Spark. The 35B-A3B int4 at 50 tok/s is the sweet spot.

**Updated run.sh scripts (March 22):**
All Qwen 3.5 `run.sh` scripts now auto-detect DGX Spark and use vLLM Docker with int4:
- `GPU_MEM_FRAC` env var controls memory allocation (default varies by model size)
- On Spark: uses `vllm/vllm-openai:vllm-node` with AutoRound int4 models
- On other Linux: uses llama.cpp with GGUF
- On macOS: uses mlx-vlm

**Comparison with llama.cpp (4B):**

| Engine | Quant | TPS |
|--------|-------|-----|
| vLLM Docker | AutoRound int4 | **43.9** |
| llama.cpp | Q4_K_M GGUF | 38.2 |

vLLM Docker int4 beats llama.cpp GGUF for the 4B model.

---

### Nemotron 3 Nano 30B NVFP4 — Not Working

**Status:** CUDA graph capture hangs / CUTLASS TMA errors

```
[ERROR] Assertion failed: Failed to initialize cutlass TMA WS grouped gemm
```

The `vllm-node:latest` image has CUTLASS SM120 kernel issues with Nemotron models.
Needs investigation or different Docker image.

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
