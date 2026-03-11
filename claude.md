# Claude Notes

## Current Work

- **TRT-LLM sourcebuild image WORKING** on DGX Spark (GB10 Blackwell)
  - Image: `local/trtllm-main:main-transformers--5.3.0-sourcebuild-120-real` (48.2GB)
  - Dockerfile: `trtllm.spark/Dockerfile.main-source`
  - Build: `FULL_SOURCE_BUILD=1 CUDA_ARCHITECTURES=120-real JOB_COUNT=4 ./trtllm.spark/build-main-image.sh`
  - Run: `docker run --rm --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 --network host -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/.cache/tensorrt_llm:/root/.cache/tensorrt_llm" local/trtllm-main:main-transformers--5.3.0-sourcebuild-120-real python3 -m tensorrt_llm.commands.serve serve <MODEL> --host 0.0.0.0 --port 2250 --backend pytorch --tp_size 1 --max_seq_len 32768`

### Benchmark Results (TRT-LLM PyTorch backend, BF16, DGX Spark)

| Model | TTFT (warm) | TPS (gen) | TPS (overall) |
|-------|-------------|-----------|---------------|
| Qwen3-0.6B | 180ms | 56.2 | 55.2 |
| Qwen3-4B | 730ms | 22.6 | 21.5 |
| Qwen3-8B | 1304ms | 14.0 | 13.6 |

Note: These are unquantized BF16. TRT-LLM PyTorch backend on Blackwell is not yet fully optimized.

### Supported Models

- TRT-LLM `main` supports: Qwen3, Qwen2, Llama, Mistral, Gemma3, DeepseekV3, etc.
- **NOT supported yet**: Qwen3.5 (`Qwen3_5ForConditionalGeneration`) — different architecture from Qwen3
- Run-docker.sh needs updating to use `python3 -m tensorrt_llm.commands.serve` instead of `trtllm-serve`

### What We Learned (attempts 1-10)

1. **LFS pointer files**: `GIT_LFS_SKIP_SMUDGE=1` with selective pull leaves binary files as pointers → full `git lfs pull` required for source build
2. **DeepEP / nvshmem**: Fixed by full LFS pull (no need for `-DBUILD_DEEP_EP=OFF`)
3. **version==None**: Shallow `--depth 1` clone has no git tags → `--skip_building_wheel` + PYTHONPATH
4. **TRTLLM_USE_PRECOMPILED=1 DESTROYS BUILD OUTPUT**: This flag tells setup.py to download/use OLD precompiled binaries, ignoring freshly built .so files. Fix: copy .so files manually.
5. **transformers 5.3 incompatibility**: 12+ source patches needed (get_parameter_device, AutoModelForVision2Seq, load_sharded_checkpoint, etc.)
6. **AutoConfig.register conflicts**: transformers 5.3 already registers model types → wrap in try/except
7. **DisabledTqdm double-disable**: Pop 'disable' kwarg before super().__init__
8. **rope_type="default"**: transformers 5.3 uses "default" for rope_type/scaling_type, TRT-LLM enums don't know it → sed alias "default" → "rope_gpt_neox"/"none"
9. **trtllm-serve not found**: No pip install means no console_scripts → use `python3 -m tensorrt_llm.commands.serve` instead
10. **PYTHONPATH over pip install**: For source builds, skip pip install entirely, use `ENV PYTHONPATH="/opt/TensorRT-LLM"`

### TODO
- Update `run-docker.sh` to support sourcebuild image properly
- Benchmark against llama.cpp/vLLM/SGLang with same models
- Try FP8/INT4 quantization for higher TPS
- Document results in docs/BENCHMARK_LOG.md
