# llama.cpp (llama-server)

## Current State

**Status: WORKING** — All five Qwen 3.5 models run via llama-server with GGUF quantizations.

Limitations:
- No continuous batching — one request at a time, concurrent users queue up
- Lower TPS than vLLM/SGLang for small models, but competitive or faster for medium models (9B)

### Environment

- Built from source with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native`
- Also needs `-DLLAMA_OPENSSL=ON` for HuggingFace downloads via `-hf` flag
- Build script: `llama.cpp/build-llama.sh` (auto-detects platform)
- OpenAI-compatible API with `--jinja` flag

### Running Models

All five models are configured as systemd services on their assigned ports (see model table in DGX_SPARK.md).

---

## Build Notes

- Build script: `llama.cpp/build-llama.sh`
- Uses `CMAKE_CUDA_ARCHITECTURES=native` which auto-detects `121a-real` for Blackwell
- Do NOT hardcode architecture numbers — Blackwell uses `121a` suffix (not just `121`)
- The script handles Mac (Metal), Intel Linux (CUDA), DGX Spark (CUDA), and Pi (CPU-only)
- Docker fallback: `docker-build-llama.sh` for Ubuntu 25.x / glibc >= 2.41

### Quantization Options

Available GGUF quantizations from unsloth repos:

| Quant | Typical Use | Notes |
|-------|-------------|-------|
| Q8_0 | Default for small models (0.8B, 2B, 4B) | Best quality, larger |
| Q4_K_M | Default for 27B | Good balance of quality/speed |
| UD-Q8_K_XL | Used for 35B-A3B | Unsloth dynamic quant |
| Q4_0 | KV cache quantization | Used with `--cache-type-k`/`--cache-type-v` |

Performance reference (Qwen3.5-2B Q8_0): ~527 tok/s prompt, ~77 tok/s generation.

---

## Nemotron-3-Super 120B-A12B (GGUF)

Dedicated launcher: `nemotron-3-super-120b-a12b/run-llama.sh`

Uses unsloth's GGUF conversion: `unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF`

### Key Configuration

| Setting | Value |
|---------|-------|
| Quant | Q4_K_M (~82.5 GB) |
| Context | 16384 (limited by 128 GB memory budget) |
| KV cache | Q4_0 (`--cache-type-k q4_0 --cache-type-v q4_0`) |
| Temperature | 1.0 (model card mandated) |
| Top-P | 0.95 (model card mandated) |
| Flags | `--no-mmap --flash-attn on` |
| Cold gen TPS | 10.9 |
| Warm gen TPS | 11.7–12.6 |
| Prompt TPS | 41–46 |

### Notes

- Q4_K_M fits in 128 GB with 16k context. Higher context lengths will OOM.
- Model architecture: NemotronHForCausalLM (Mamba-2 hybrid + LatentMoE, 120B total / 12B active)
- Competitive with vLLM NVFP4 on single-request throughput (~12 tok/s both), but no continuous batching
