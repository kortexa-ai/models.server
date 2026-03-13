# Claude Notes

## Documentation

Docs have been reorganized into focused files:

- `docs/DGX_SPARK.md` — Hardware specs, CUDA caveats, PyTorch compat, GPU memory management
- `docs/ENGINE_VLLM.md` — vLLM setup, Docker/bare-metal attempts, 4B quant sweep
- `docs/ENGINE_SGLANG.md` — SGLang Docker/bare-metal status, NVFP4 issues
- `docs/ENGINE_TRTLLM.md` — TRT-LLM sourcebuild, 10 build attempts, transformers 5.x patches
- `docs/ENGINE_LLAMACPP.md` — llama-server build notes, quantization options
- `docs/BENCHMARKS.md` — All benchmark results, dated reverse-chronological

## Current Status Summary

| Engine | Status | Qwen 3.5 Support |
|--------|--------|-------------------|
| llama.cpp | Working (all 5 models) | Yes |
| vLLM bare-metal | Working (all 5 sizes verified) | Yes |
| SGLang Docker | Working (0.8B, 9B tested) | Yes |
| SGLang bare-metal | Blocked (sgl-kernel build) | — |
| TRT-LLM Docker (sourcebuild) | Working (Qwen 3 only) | No |

## TODO

- SGLang setup/verification (after vLLM is stable)
- TRT-LLM: monitor for Qwen 3.5 support
- TRT-LLM: try FP8/INT4 quantization
- Test bare-metal TRT-LLM build script
