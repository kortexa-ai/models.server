# DGX Spark Performance Investigation: Qwen 3.5 35B A3B

**Date:** March 21, 2026
**Investigator:** Penny
**Target:** Understand why sparky gets 17.4 tok/s when others report 50-79 tok/s

## Executive Summary

**Root cause identified:** Vanilla vLLM on DGX Spark (SM121/GB10) has multiple kernel bugs and missing optimizations that limit performance to ~28-40 tok/s. To achieve 60-70 tok/s, you need:

1. **MXFP4 quantization patches** from `namake-taro/vllm-custom`
2. **Per-layer precision tuning** (MXFP4 for QKV, FP8 for o_proj)
3. **SM121 kernel fixes** (Marlin MoE 256-thread race condition)

---

## Performance Comparison

| Configuration | tok/s | Source |
|--------------|-------|--------|
| **Current (sparky)** | 17.4 | BF16, vLLM 0.17.1rc1, TP=1 |
| Vanilla BF16 TP=1 | 28.4 | namake-taro benchmarks |
| Vanilla BF16 TP=2 | 39.7 | namake-taro benchmarks |
| **Patched MXFP4 TP=1** | **60.1** | namake-taro benchmarks |
| **Patched MXFP4 TP=2** | **70.8** | namake-taro benchmarks |
| Intel AutoRound int4 | 64-66 | HuggingFace discussions |
| Custom vLLM + torch.compile | 77 | eole-nlp reports |

The gap between our 17.4 tok/s and vanilla 28.4 tok/s is likely due to:
- Different context length settings
- Memory pressure from other running processes
- Suboptimal `gpu-memory-utilization` setting

---

## Root Causes

### 1. SM121 Kernel Bugs in Vanilla vLLM

The Blackwell architecture (SM121 / compute capability 12.1) has several kernel issues:

#### a) Marlin MoE 256-thread Kernel Race Condition
- **File:** `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`
- **Issue:** When N >= 2048, the kernel uses 256 threads which causes shared memory race on SM121
- **Symptom:** Garbage output at TP=1
- **Fix:** Force 128-thread config for w2 GEMM

#### b) GDN Triton Kernel Issue
- **File:** `vllm/model_executor/layers/fla/ops/fused_recurrent.py`
- **Issue:** Gated Delta Net kernel doesn't work correctly for Qwen3.5 on SM121
- **Symptom:** Incorrect output or crashes

#### c) CUTLASS SFA/SFB Layout Bug
- **File:** FlashInfer's bundled CUTLASS 4.2.1 headers
- **Issue:** `layout_SFB` initialization incorrectly uses `tile_atom_to_shape_SFA`
- **Symptom:** Broken CUTLASS_FP4 backend

### 2. Missing BF16 → MXFP4 Online Quantization

Vanilla vLLM only supports **pre-quantized** MXFP4 models. It cannot:
- Load BF16 checkpoints and quantize them on-the-fly
- Apply per-layer precision tuning

The patches add:
- `Mxfp4LinearMethod` for QKV layers
- `Fp8MarlinOProjLinearMethod` for o_proj (FP8 prevents repetition loops)
- `Mxfp4LMHeadMethod` for lm_head

### 3. Per-Layer Precision Requirements

Not all layers should use the same quantization:

| Layer | Optimal Precision | Why |
|-------|------------------|-----|
| MoE experts (w1, w2, w3) | MXFP4 (E2M1) | Bandwidth-bound, tolerant to quantization |
| QKV projections | MXFP4 (E2M1) | Softmax normalizes quantization error |
| **o_proj** | **FP8 (E4M3)** | MXFP4 causes repetition loops in long generation |
| lm_head | MXFP4 (E2M1) | BF16 fallback when `tie_word_embeddings=True` |
| embed_tokens | BF16 | Embedding gather, not a GEMM |
| router | BF16 | Negligible size (~13 MB) |
| layer_norm | BF16 | Negligible size (~0.4 MB) |

### 4. Memory Bandwidth Ceiling

GB10 has 273 GB/s memory bandwidth. For gpt-oss-120b:
- Active weights per token: ~2.9 GB
- Theoretical max: 1000 / (2.94 / 273) = **92.6 tok/s**

The gap between theoretical and measured (60-70 tok/s) is due to:
- GEMM kernel efficiency
- Non-GEMM compute overhead
- KV cache memory traffic

---

## Solution: Apply the Patches

### Prerequisites

```bash
# Already have these on sparky:
# - Python 3.12.3
# - CUDA 13.0
# - vLLM 0.17.1rc1.dev18+g7d6abdd02.cu130
```

### Step 1: Download Patches

```bash
cd /home/francip/src/models.server
mkdir -p patches
cd patches

# Download the patches from namake-taro/vllm-custom
curl -LO https://raw.githubusercontent.com/namake-taro/vllm-custom/master/patches/vllm_all.patch
curl -LO https://raw.githubusercontent.com/namake-taro/vllm-custom/master/patches/flashinfer_cutlass_sfb_layout_fix.patch
```

### Step 2: Create Patched venv

```bash
# Option A: Patch existing venv
SITE=/home/francip/src/models.server/bench/.venv-vllm/lib/python3.12/site-packages
cd "$SITE"
patch -p1 --dry-run < /home/francip/src/models.server/patches/vllm_all.patch
# If dry-run succeeds:
patch -p1 < /home/francip/src/models.server/patches/vllm_all.patch

# Option B: Create new venv (safer)
python3 -m venv ~/.python-vllm-mxfp4
source ~/.python-vllm-mxfp4/bin/activate
pip install vllm --extra-index-url https://wheels.vllm.ai/0.17.1/cu130 \
                 --extra-index-url https://download.pytorch.org/whl/cu130
pip install 'nvidia-nccl-cu13>=2.29.2' 'transformers==5.3.0' 'huggingface_hub==1.5.0' fastsafetensors
# Then apply patches
```

### Step 3: Run with MXFP4

```bash
# Using the Intel AutoRound int4 model (already downloaded)
vllm serve Intel/Qwen3.5-35B-A3B-int4-AutoRound \
  --port 2237 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.8 \
  --load-format fastsafetensors \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching
```

### Expected Results

| Metric | Before | After |
|--------|--------|-------|
| tok/s (single request) | 17-28 | 60-70 |
| TTFT | ~4.3s | ~200ms |
| Memory usage | ~67GB BF16 | ~21GB int4 |

---

## Alternative: Use Docker

If patching bare-metal is too invasive, use a pre-built Docker image:

```bash
# Option 1: NVIDIA's vLLM container (CUDA graphs work, but no MXFP4 patches)
docker run --rm --name vllm-qwen35 \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  nvcr.io/nvidia/vllm:26.02-py3 \
  vllm serve Intel/Qwen3.5-35B-A3B-int4-AutoRound \
  --port 2237 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.25 \
  --kv-cache-dtype fp8

# Option 2: Build custom Docker with patches (best performance)
# See: https://github.com/namake-taro/vllm-custom for Dockerfile
```

---

## Additional Optimizations

### 1. Use FP8 KV Cache
```bash
--kv-cache-dtype fp8
```
Halves KV memory, critical on 128GB unified memory.

### 2. Enable Prefix Caching
```bash
--enable-prefix-caching
```
Dramatically improves TTFT for repeated prompts.

### 3. Use FastSafetensors
```bash
--load-format fastsafetensors
```
Faster model loading, especially for sharded models.

### 4. GPU Memory Utilization
```bash
--gpu-memory-utilization 0.8  # For single model
--gpu-memory-utilization 0.25 # When other processes are running
```

---

## References

1. **MXFP4 Patches:** https://github.com/namake-taro/vllm-custom
2. **NVIDIA Forum Discussion:** https://forums.developer.nvidia.com/t/vllm-0-17-0-mxfp4-patches-for-dgx-spark-qwen3-5-35b-a3b-70-tok-s-gpt-oss-120b-80-tok-s-tp-2/362824
3. **Intel AutoRound Model:** https://huggingface.co/Intel/Qwen3.5-35B-A3B-int4-AutoRound
4. **HuggingFace Discussion:** https://huggingface.co/Intel/Qwen3.5-35B-A3B-int4-AutoRound/discussions/1

---

## ✅ SOLUTION FOUND (March 21, 2026)

### Working Configuration: Docker vLLM 0.17.2rc1 + Intel AutoRound int4

The `vllm-node:latest` Docker image achieves **46-52 tok/s** without any patching:

```bash
docker run --rm --name vllm-test-35b \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e VLLM_BASE_DIR=/root/.cache/huggingface \
  vllm-node:latest \
  Intel/Qwen3.5-35B-A3B-int4-AutoRound \
  --port 2242 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.7 \
  --load-format fastsafetensors \
  --kv-cache-dtype fp8
```

### Benchmark Results

| Test | Tokens | Duration | tok/s |
|------|--------|----------|-------|
| 1 | 512 | 9.92s | 51.60 |
| 2 | 512 | 11.07s | 46.23 |
| 3 | 512 | 9.92s | 51.62 |
| 4 | 1024 | 19.94s | 51.36 |
| 5 | 100 | 1.99s | 50.35 |

**Average: ~50 tok/s** (within target range of 50-79 tok/s!)

### Why This Works

1. **vLLM 0.17.2rc1** has better SM121 support than 0.17.1rc1
2. **CUDA Forward Compatibility** mode enabled (CUDA 13.1 on driver 580)
3. **FlashInfer attention backend** properly supports SM121
4. **Intel AutoRound int4** model is optimized for this architecture
5. **CUDA graphs** are successfully captured (51 PIECEWISE + 35 FULL)
6. **FP8 KV cache** reduces memory bandwidth pressure

### Key Log Messages

```
Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN']
Model loading took 19.3 GiB memory
Available KV cache memory: 60.32 GiB
Graph capturing finished in 35 secs
```

### Memory Usage

- Model: 19.3 GiB
- KV cache: 60+ GiB available
- Total: ~80 GiB (leaving room for other processes)

---

## Remaining Optimization Opportunities

To push from 50 tok/s toward 70+ tok/s:

1. **Apply MXFP4 patches** for online quantization (could add 20-30%)
2. **Use TP=2** if you have two GPUs (would nearly double throughput)
3. **Kill competing processes** (llama-server, music-gen) to free memory
4. **Increase gpu-memory-utilization** to 0.85 after freeing memory

---

## Original Next Steps (Completed)

1. ✅ **Benchmark** Intel AutoRound int4 model - DONE (50 tok/s)
2. **Compare** with llama.cpp GGUF performance
3. **Document** results in BENCHMARKS.md
4. **Consider** applying MXFP4 patches for 70+ tok/s
