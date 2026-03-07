# kortexa.ai Qwen 3.5 27B LLM server

This directory now keeps the production-ish Spark `llama-server` path at the
top level and the experimental DGX Spark `vLLM` work isolated under
[`../vllm-spark/`](../vllm-spark/).

## Current Spark Default

For the single-user DGX Spark setup, the default Linux backend is:

- `llama-server`
- `unsloth/Qwen3.5-27B-GGUF:Q4_K_M`
- `262144` token context
- `q4_0` KV cache

Start it with:

```bash
../setup.sh
./run.sh
```

## Docs

- [`QWEN_SPARK.md`](./QWEN_SPARK.md) - benchmark notes, VRAM sizing, and the
  current recommended presets for `0.8B`, `2B`, `4B`, `9B`, `27B`, and
  `35B-A3B`
- [`vllm-spark/README.md`](../vllm-spark/README.md) - Dockerized Spark `vLLM`
  build/run instructions and benchmark assets

## Notes

- Do not upgrade NVIDIA drivers on DGX Spark for this setup.
- `run.sh` is the preferred Linux path here until the `vLLM` path proves a
  clear single-user latency win.

---

© 2025 kortexa.ai
