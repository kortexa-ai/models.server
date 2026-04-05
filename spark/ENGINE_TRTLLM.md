# TensorRT-LLM

## Current State

**Sourcebuild Docker: WORKING** — Custom image built from TRT-LLM `main` with native sm_120 Blackwell binaries.

- Image: `local/trtllm-main:main-transformers--5.3.0-sourcebuild-120-real` (48.2GB)
- Dockerfile: `trtllm.spark/Dockerfile.main-source`
- Build: `FULL_SOURCE_BUILD=1 CUDA_ARCHITECTURES=120-real JOB_COUNT=4 ./trtllm.spark/build-main-image.sh`
- Run: `docker run --rm --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 --network host -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/.cache/tensorrt_llm:/root/.cache/tensorrt_llm" local/trtllm-main:main-transformers--5.3.0-sourcebuild-120-real python3 -m tensorrt_llm.commands.serve serve <MODEL> --host 0.0.0.0 --port 2250 --backend pytorch --tp_size 1 --max_seq_len 32768`

**Supports:** Qwen3, Qwen2, Llama, Mistral, Gemma3, DeepseekV3, etc.
**NOT supported:** Qwen 3.5 (`Qwen3_5ForConditionalGeneration`) — different architecture from Qwen 3.

**Bare-metal build script:** `trtllm.spark/build-bare-metal.sh` (untested, extracted from Docker learnings)

---

## Files

- `trtllm.spark/Dockerfile.main-source` — Full source build Dockerfile
- `trtllm.spark/build-main-image.sh` — Build script
- `trtllm.spark/run-docker.sh` — Docker launcher
- `trtllm.spark/build-bare-metal.sh` — Bare-metal build script (experimental)
- `trtllm.spark/extra-llm-api-config.yml` — Optional PyTorch backend config

---

## Official Release Image Attempts (Blocked)

### Attempt 1: Stock `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6`

- Ships `transformers 4.57.1` — doesn't know `qwen3_5` model type
- Works for Qwen 3 and older models

### Attempt 2: Upgrade to `transformers 4.57.6`

- Still fails inside TRT-LLM worker init — `qwen3_5` config-recognition error
- 4.57.6 was not enough inside TRT-LLM's own runtime path

### Attempt 3: Upgrade to `transformers 5.3.0`

- `qwen3_5` becomes recognizable
- But stock TRT-LLM imports legacy symbols that no longer exist in transformers 5.x:
  - `AutoModelForVision2Seq` removed
  - `get_parameter_device` removed
  - `load_sharded_checkpoint` removed
- Broader compatibility gap — not just one missing flag

---

## Full Source Build from `main`

### Build Harness

```bash
FULL_SOURCE_BUILD=1 CUDA_ARCHITECTURES=120-real JOB_COUNT=4 ./trtllm.spark/build-main-image.sh
```

### `transformers 5.x` Compatibility Patches

The Dockerfile applies ~12 compatibility fixes:

- `AutoModelForVision2Seq` → `AutoModelForImageTextToText` fallback
- `get_parameter_device` / `get_parameter_dtype` — replaced with local shims
- `load_sharded_checkpoint` — replaced with fallback impl
- `AutoConfig.register` — wrapped in try/except (transformers 5.x raises ValueError on duplicate registration)
- `DisabledTqdm` — pop 'disable' kwarg before `super().__init__`
- `rope_type="default"` — transformers 5.3 uses "default" for rope_type/scaling_type, patched to alias "default" → "rope_gpt_neox" / "none"
- `suffix_automaton` native binding optional import
- `record_global_timer` optional import with fallback
- `requirements.txt` pinned to `transformers==5.3.0`

### Build Attempts Summary

| # | Change | Result |
|---|--------|--------|
| 1 | Precompiled (no source build) | Built, untested — may lack sm_120 kernels |
| 2 | Full source build, JOB_COUNT=4 | Killed after 1.5hr (process termination) |
| 3 | Restart, cache reuse | Failed at 66% — nvshmem hash mismatch (LFS pointer) |
| 4 | Disable DeepEP (`-DBUILD_DEEP_EP=OFF`) | Failed at 77% — xqa_kernel_cubin.cpp is LFS pointer |
| 5 | Full LFS pull + DeepEP disabled | Failed at end — `deep_ep` target reference in wheel targets |
| 6 | Full LFS pull, no DeepEP override | 100% compiled, failed at wheel packaging (version==None) |
| 7 | Skip wheel build (`--skip_building_wheel`) | 100% compiled, failed at editable install (version==None) |
| 8 | No --install, manual pip install --no-deps | Crashed — `TRTLLM_USE_PRECOMPILED=1` overwrites built .so files |
| 9 | Manual .so copy after compile, PYTHONPATH | Working — all kernels compiled, server starts |
| 10 | Add rope_type patches | Working — Qwen3 models serve successfully |

### Key Learnings

1. **LFS pointer files**: `GIT_LFS_SKIP_SMUDGE=1` with selective pull leaves binary files as pointers → full `git lfs pull` required
2. **DeepEP / nvshmem**: Fixed by full LFS pull (no need for `-DBUILD_DEEP_EP=OFF`)
3. **version==None**: Shallow `--depth 1` clone has no git tags → `--skip_building_wheel` + PYTHONPATH
4. **TRTLLM_USE_PRECOMPILED=1 DESTROYS BUILD OUTPUT**: This flag tells setup.py to download/use OLD precompiled binaries, ignoring freshly built .so files. Fix: copy .so files manually.
5. **transformers 5.3 incompatibility**: 12+ source patches needed
6. **AutoConfig.register conflicts**: transformers 5.3 already registers model types → wrap in try/except
7. **DisabledTqdm double-disable**: Pop 'disable' kwarg before super().__init__
8. **rope_type="default"**: transformers 5.3 uses "default", TRT-LLM enums don't know it → sed alias
9. **trtllm-serve not found**: No pip install means no console_scripts → use `python3 -m tensorrt_llm.commands.serve`
10. **PYTHONPATH over pip install**: For source builds, skip pip install entirely, use `ENV PYTHONPATH="/opt/TensorRT-LLM"`

---

## TODO

- Update `run-docker.sh` to support sourcebuild image properly
- Try FP8/INT4 quantization for higher TPS
- Test bare-metal build script
- Monitor TRT-LLM for Qwen 3.5 support
