# SGLang

## Current State

**Docker: WORKING** — `scitrera/dgx-spark-sglang:0.5.9-dev1-329817e2-t5` runs Qwen 3.5 0.8B and 9B.

**Bare-metal: BLOCKED** — Env rebuilt with `torch 2.10.0+cu130` + source-installed SGLang + `sgl-kernel 0.3.21`, but blocked on native `sgl-kernel` buildability for GB10. The prebuilt wheel has Torch ABI symbol mismatches.

**NVFP4: BLOCKED** — `AxionML/Qwen3.5-*-NVFP4` checkpoints fail during weight loading with shape assertion errors in both tested Docker images.

---

## Docker Setup

### Working Images

| Image | SGLang | Status |
|-------|--------|--------|
| `scitrera/dgx-spark-sglang:0.5.9-t5` | 0.5.9 | Working, ~87 tok/s (0.8B) |
| `scitrera/dgx-spark-sglang:0.5.9-dev1-329817e2-t5` | 0.5.9-dev | Working, ~93 tok/s (0.8B) |
| `lmsysorg/sglang:spark` | 0.5.4.post2 | Fails — transformers 4.57.1 doesn't know qwen3_5 |

### Running via Docker

```bash
./sglang.spark/run-docker.sh --profile standard 0.8b
```

### Files

- `sglang.spark/run.sh` — bare-metal launcher
- `sglang.spark/run-docker.sh` — Docker launcher
- `sglang.spark/resolve_model.py` — model name resolution
- `sglang.spark/models.json` — model presets
- `sglang.spark/benchmark_sglang_vs_llama.py` — comparison harness

---

## Bare-Metal Attempts

### March 9, 2026 — Clean Rebuild

Created `bare.spark/.venv-sglang` (later moved to `bench/.venv-sglang`) with:

- Python 3.12.3
- `torch 2.10.0+cu130`
- Source-installed SGLang from `sgl-project/sglang`
- `sgl-kernel 0.3.21`
- `transformers 5.3.0`
- `triton 3.6.0`
- `flashinfer-python 0.6.5`

### What We Learned

1. Public `sglang==0.5.9` is not a good base — hard-pins `torch==2.9.1`, `transformers==4.57.1`, `flashinfer_python==0.6.3`, `cuda-python==12.9`
2. Installing current SGLang code with `--no-deps` plus a container-shaped runtime stack works better
3. Prebuilt `sgl-kernel` wheel doesn't work cleanly:
   - First wanted CUDA 12 runtime libs
   - After adding those, failed with Torch ABI symbol mismatch
4. Rebuilding `sgl-kernel` from source is the right direction, but blocked on:
   - Missing `libnuma-dev` on host
   - CMake 4 compatibility failure in pulled `dlpack` dependency

### Workarounds Applied

- Defaulted to `--attention-backend triton` (Qwen 3.5 on Blackwell asserts with other backends)
- Set `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` (bundled Triton ptxas 12.8 doesn't recognize sm_121a)
- Patched `GemmaRMSNorm` to use native PyTorch on GB10 (packaged `sgl_kernel` CUDA op throws `no kernel image is available for execution on the device`)

**Caveat:** These native fallbacks mean bare-metal SGLang numbers are a workaround path, not the intended optimized kernel stack.

---

## NVFP4 Status

Tested `AxionML/Qwen3.5-0.8B-NVFP4` in both scitrera Docker images:

Both reached `ModelOptModelLoader`, detected NVFP4 checkpoint, then failed:

```text
Parameter model.layers.0.linear_attn.in_proj_a.input_scale not found in params_dict
...
File ".../sglang/srt/layers/linear.py", line 418, in weight_loader
    assert param_data.shape == loaded_weight.shape
AssertionError
```

Also tried `olka-fi/Qwen3.5-9B-MXFP4` — blocked on `CompressedTensorsW4A16Sparse24` not supported.

---

## SGLang via pip (Early Attempts)

- torch cu126: missing sm_121 kernels
- torch cu128: transformers version conflicts
- Fundamentally blocked by PyTorch + Blackwell kernel gap in pip wheels
