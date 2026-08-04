# Model Serving Infrastructure

Local LLM serving across multiple machines. Each model gets its own directory with configuration; shared engine scripts handle the actual launching.

## Quick Start

```bash
# Setup (once per machine)
./setup.sh

# Run a model
./run.sh qwen-3.5-4b                    # from root
cd qwen-3.5-4b && ../run.sh             # from model dir
./run.sh gemma-4-26b-a4b --engine vllm  # override engine
```

## Machines

| Host/IP | Hardware | Memory | OS | Primary Backend |
|---------|----------|------------|------|-----------------|
| **smarty** | RTX PRO 6000 Blackwell | 96 GB VRAM | Ubuntu Linux | `llama-server` (GGUF), bare-metal vLLM |
| **snappy** | Mac Mini M4 Pro | 64 GB unified | macOS | `mlx-vlm` (MLX) |
| **scrappy** | RTX 3070 Laptop | 8 GB VRAM | Windows 11 | — |
| **sparky** | DGX Spark GB10 | 128 GB unified | Ubuntu Linux | offline |
| **192.168.2.144** | Raspberry Pi 5 | 8 GB RAM | ARM Linux | `llama-server` CPU |
| **192.168.2.145** | Raspberry Pi 5 | 8 GB RAM | ARM Linux | `llama-server` CPU |

## Model Inventory

| Port | Model | Type | Quant | KV Cache | Context | Parallel |
|------|-------|------|-------|----------|---------|----------|
| 2025 | Qwen 3.5 9B | big dense | UD-Q4_K_XL | q8_0 | 64K | 2 |
| 2028 | Qwen 3.6 35B A3B | MoE | UD-Q4_K_XL | q8_0 | 64K | 8 |
| 2029 | Qwen 3.5 4B | small dense | UD-Q4_K_XL | q8_0 | 64K | 2 |
| 2030 | Qwen 3.5 2B | small dense | Q8_0 | q8_0 | 32K | 2 |
| 2031 | Qwen 3.5 0.8B | small dense | Q8_0 | q8_0 | 32K | 2 |
| 2032 | Qwen 3.6 27B | big dense | UD-Q4_K_XL / FP8 | q8_0 / fp8 | 262K | 1 / 4 |
| 2036 | Gemma 4 26B-A4B | MoE | UD-Q4_K_XL | q8_0 | 64K | 8 |
| 2037 | Gemma 4 31B | big dense | UD-Q4_K_XL | q8_0 | 64K | 2 |
| 2038 | Gemma 4 E4B | small dense | UD-Q4_K_XL | q8_0 | 64K | 2 |
| 2039 | Gemma 4 E2B | small dense | Q8_0 | q8_0 | 32K | 2 |
| 2040 | Qwen3 Embedding 0.6B | embedding | Q8_0 | q8_0 | 32K | 4 |
| 2041 | EmbeddingGemma 300M | embedding | Q4_0 | q8_0 | 2K | 1 |
| 2043 | Gemma 4 12B | big dense | UD-Q4_K_XL | q8_0 | 64K | 2 |
| 2045 | LFM2.5 230M | tiny dense / edge | Q8_0 (CPU Q4_K_M) | q8_0 (CPU q4_0) | 128K/slot | 4 |
| 2046 | LFM2.5 350M | tiny dense / edge | Q8_0 (CPU) | q8_0 | 128K/slot | 4 |
| 2047 | LFM2 350M Extract | structured extraction | Q8_0 (CPU) | q8_0 | 128K/slot | 4 |
| 2048 | LFM2.5 Encoder 350M | masked-LM encoder | FP32 (MPS / CPU) | — | 8K | 1 |
| 2042 | LFM2.5 Embedding 350M | embedding | Q8_0 (Metal / CPU) | q8_0 | 2K | 2 |
| 2052 | LFM2.5 VL 450M | tiny VLM / edge | Q8_0 / MLX 8-bit | q8_0 | 32K | 1 |
| 4007 | Penumbra | `control.server` discovery | — | — | — | — |

The `penumbra/model.json` entry intentionally lets `control.server` discover
Penumbra as a model. The actual runtime remains the separate `penumbra.server`
service on port 4007.

### Available and Reserved Ports

| Port | Status | Notes |
|------|--------|-------|
| 2026 | Available | Former Qwen 3.5 27B port |
| 2027 | Available | Former Qwen 3.5 35B A3B reservation |
| 2033 | Available | Former Nemotron 3 Super 120B A12B port |
| 2034 | Available | Former Nemotron 3 Nano 30B A3B port |
| 2035 | Available | Former Nemotron Cascade 2 30B A3B port |
| 2049 | Blocked | NFS port; Fetch implementations such as Node reject it as an unsafe port |
| 2050 | Reserved | Default `hermes-router` sidecar port; do not assign to a model |
| 2051 | Reserved | Default port for the `hermes-auxiliary-brain` managed llama.cpp server; do not assign to a model |

## Directory Structure

```
models.server/
├── run.sh                  # Single entry point — detects platform, dispatches
├── setup.sh                # Environment setup (MLX on macOS, vLLM on Linux)
├── scripts/
│   ├── run-llama.sh        # Generic llama.cpp launcher
│   ├── run-mlx.sh          # Generic MLX launcher
│   ├── run-vllm.sh         # Generic vLLM launcher
│   ├── run-cpu.sh          # Generic CPU-only launcher (Pi)
│   ├── run-transformers.sh # Generic Transformers launcher
│   ├── transformers-server.py # Server for non-generative tasks
│   ├── parse-config.py      # Reads model.json → shell variables
│   ├── setup-common.sh      # Shared helpers (CUDA env, venv paths)
│   ├── setup-vllm.sh        # Creates/updates .venv-vllm
│   ├── setup-mlx.sh         # Creates/updates .venv-mlx
│   └── setup-transformers.sh # Creates/updates the Transformers .venv
├── <model-id>/
│   ├── model.json          # All config: ports, quants, engine settings
│   ├── launchd/            # macOS service unit
│   └── systemd/            # Linux service unit
├── .venv-mlx/              # Shared MLX venv (macOS)
├── .venv-vllm/             # Shared vLLM venv (Linux)
├── llama.cpp/              # llama.cpp build scripts
├── whisper.cpp/            # whisper.cpp build scripts
└── bench/                  # Benchmark results
```

## Engine Auto-Detection

`run.sh` picks the engine automatically:

- A model's `default_engine` wins when configured. It may be a single engine or a per-platform map keyed by `uname` (the LFM embedder uses Metal-backed llama.cpp on Darwin and CPU-only llama.cpp on Linux).
- **macOS** → `mlx` (mlx-vlm or mlx-lm)
- **ARM Linux without CUDA** → `cpu` (Raspberry Pi)
- **Linux with CUDA** → `llama` (llama.cpp), or `vllm` if model has no GGUF (NVFP4)

Override with `--engine`: `./run.sh qwen-3.5-4b --engine vllm`

## Serving Backends

### llama-server (llama.cpp)
GGUF-quantized models via [llama.cpp](https://github.com/ggerganov/llama.cpp). OpenAI-compatible APIs at `/v1/chat/completions`, or `/v1/embeddings` for embedding models. CUDA + flash attention on smarty, Metal on snappy.

`model.context` is the total llama.cpp context. With `parallel > 1`, llama.cpp divides that total across slots. For example, LFM2.5 230M uses `context=512000` and `parallel=4`, which gives four 128K slots.

llama.cpp [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) adds MTP (Multi-Token Prediction) speculative decoding using draft heads baked into the main GGUF (no separate drafter file). Set `llama.mtp=true` in `model.json` to pass `--spec-type draft-mtp`; optional `llama.mtp_n_max` overrides `--spec-draft-n-max`. Requires a llama.cpp build from after PR #22673 and a GGUF repo that ships MTP heads (e.g. unsloth's `*-MTP-GGUF` variants). Qwen 3.6 27B explicitly uses depth 3, its fastest measured llama.cpp setting; see `bench/BENCHMARKS.md`.

### mlx-vlm / mlx-lm
Vision Language Models via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), and text-only MLX models via `mlx-lm` when `mlx.backend` is `mlx_lm`. macOS only (Apple Silicon / MLX). VLMs serve at `/chat/completions` (no `/v1` prefix); `mlx-lm` serves OpenAI-compatible `/v1` routes.

`mlx-lm` does not take a llama-style context flag. Use `mlx.prompt_concurrency` and `mlx.decode_concurrency` for request batching, plus optional prompt-cache fields. `mlx-vlm` exposes different knobs such as `mlx.max_kv_size`, `mlx.vision_cache_size`, and `mlx.prefill_step_size`; these are passed only when set.

`mlx-vlm>=0.6.0` supports speculative decoding on the server. Add optional `mlx.draft_model`, `mlx.draft_kind`, and `mlx.draft_block_size` fields in `model.json` to pass `--draft-model`, `--draft-kind`, and `--draft-block-size`; set `MLX_DISABLE_DRAFT=1` when launching to run without the configured drafter.

LFM2.5 VL 450M uses LiquidAI's official 8-bit MLX checkpoint on snappy. It has a 32K multimodal context and one request slot; `mlx-vlm>=0.6.6` is required for the LFM2-VL loader and tokenizer fixes.

**Gemma 4 MTP drafters** work but only help large/slow targets. E2B/E4B run with `mlx.draft_enabled=false` (MTP measured *slower* than no-drafter on E4B — 66.8 vs 70.6 tok/s; see `bench/BENCHMARKS.md`); 26B-A4B/31B keep `draft_enabled=true` pending an MLX bench. The Gemma 4 MTP rollback crash ([mlx-vlm#1260](https://github.com/Blaizzy/mlx-vlm/issues/1260), `AttributeError: 'list' object has no attribute 'max'`) is fixed upstream in `mlx-vlm 0.6.1` (our PR [#1261](https://github.com/Blaizzy/mlx-vlm/pull/1261)). The old local patch has been removed; the current setup floor also includes the later LFM2-VL fixes.

### vLLM
GPU-accelerated serving via [vLLM](https://github.com/vllm-project/vllm). Linux only (CUDA). Supports online FP8 quantization, Marlin NVFP4, and continuous batching for high-throughput concurrent serving.

vLLM treats context as per-sequence length. Use `vllm.max_model_len` for `--max-model-len` and `vllm.max_num_seqs` for request concurrency. If `vllm.max_model_len` is absent, the launcher falls back to `model.context`.

Set `vllm.kv_cache_bytes` to pass an exact `--kv-cache-memory-bytes` pool instead of sizing KV from a percentage of VRAM. The pool is shared dynamically by up to `vllm.max_num_seqs` requests: one request may consume the full pool, while concurrent requests divide it according to their live token counts. `gpu_memory_utilization` remains a startup admission guard when exact bytes are configured; it does not resize the pool. Native MTP is configured through `vllm.speculative_config`.

Qwen 3.6 27B uses a 10,194,124,800-byte FP8 KV pool, which vLLM 0.26.0 reports as exactly 262,144 aggregate tokens with MTP depth 4 and up to four scheduled requests. Four simultaneous 262K requests do not fit; four equally long requests can use about 65K tokens each. The default Linux engine remains llama.cpp because it is faster and smaller for the usual single request. Use `./run.sh qwen-3.6-27b --engine vllm` when continuous batching is more valuable; the measured four-request aggregate was 310 tok/s. Keep the vLLM attention backend on `auto`: forcing Triton caused an illegal-memory-access crash under four-way load.

### CPU llama-server
ARM Linux without CUDA auto-selects the `cpu` engine. This is mainly for the Raspberry Pi 5 nodes (`192.168.2.144` and `192.168.2.145`); LFM2.5 230M uses its `cpu` config with GGUF `Q4_K_M`, 512K total context across four 128K slots, q4 KV cache, flash attention, and `checkpoint_min_step=0` for effective warm prompt reuse. `Q4_K_M` matches Liquid's general recommended GGUF balance; flash attention is their Pi-specific note.

LFM2.5 350M and LFM2 350M Extract default to the same CPU engine on every platform. The CPU launcher explicitly disables device offload. Both use LiquidAI's official `Q8_0` GGUFs, 512K total context across four 128K slots, q8 KV cache, flash attention, and warm prompt reuse. Pass `--engine llama` only when GPU offload is intentionally wanted.

LFM2.5 VL 450M uses LiquidAI's official `Q8_0` GGUF and matching vision projector on llama.cpp and CPU backends. It keeps the model's 32K multimodal context in one slot; macOS defaults to the official 8-bit MLX checkpoint instead.

LFM2.5 Embedding 350M uses the official `Q8_0` GGUF on both platforms. Its per-platform default selects Metal-backed llama.cpp on snappy and CPU-only llama.cpp on Linux, including smarty. It exposes `/v1/embeddings` on port 2042 with two slots sharing 2K total context. Port 2049 is deliberately skipped because Fetch implementations block the historical NFS port.

### Transformers (MPS / CPU)

LFM2.5 Encoder 350M is a bidirectional masked-language model, not a causal LLM. LiquidAI does not publish a GGUF for this exact checkpoint, and llama.cpp cannot serve its masked-LM API. It therefore defaults to the small Transformers backend while the two generative 350M models stay on llama.cpp. The backend uses FP32 on every platform, selecting MPS on Apple Silicon and CPU elsewhere. CUDA is deliberately disabled so running it on smarty does not consume VRAM.

Install its runtime with `scripts/setup-transformers.sh`, then launch it with `./run.sh lfm2.5-encoder-350m`. It exposes `GET /health`, `GET /v1/models`, and `POST /v1/fill-mask`:

```bash
curl http://localhost:2048/v1/fill-mask \
  -H 'Content-Type: application/json' \
  -d '{"input":"The capital of France is <|mask|>.","top_k":5}'
```

## Quantization Standards

| Model size | Weight quant | KV cache | Context | Parallel slots |
|------------|-------------|----------|---------|----------------|
| >= 4B | UD-Q4_K_XL | q8_0 / fp8 | 64K | MoE: 8, big dense: 2, small: 2 |
| < 4B | Q8_0 | q8_0 / fp8 | 32K | 2 |

LFM2.5 230M is the small-edge exception: CUDA uses Q8_0, while Pi CPU uses Q4_K_M. It is also configured for four 128K slots on llama.cpp-style backends and four-way prompt/decode concurrency on `mlx-lm`.
The LFM 350M generative, vision-language, and embedding models use `Q8_0`; the generative models default to CPU, the VLM defaults to MLX on Darwin, and the embedder selects Metal on Darwin and CPU on Linux. LFM2.5 Encoder 350M stays FP32 because its bidirectional masked-LM checkpoint has no GGUF.

## Adding a New Model

1. Create `<model-id>/` directory
2. Add `model.json` with all engine config (see any existing model for the schema)
3. Add `launchd/` and `systemd/` service units
4. Follow the quantization standards above
5. Test: `./run.sh <model-id>`

## Service Management

### macOS (launchd)
```bash
ln -s ~/src/models.server/<model-id>/launchd/ai.kortexa.<model-id>.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/ai.kortexa.<model-id>.plist
launchctl start ai.kortexa.<model-id>
```

### Linux (systemd)
```bash
sudo ln -s ~/src/models.server/<model-id>/systemd/kortexa-ai-llm-<model-id>.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start kortexa-ai-llm-<model-id>
```
