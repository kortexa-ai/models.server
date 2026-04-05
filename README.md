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

| Hostname | Hardware | GPU Memory | OS | Primary Backend |
|----------|----------|------------|------|-----------------|
| **smarty** | RTX PRO 6000 Blackwell | 96 GB VRAM | Ubuntu Linux | `llama-server` (GGUF), bare-metal vLLM |
| **snappy** | Mac Mini M4 Pro | 64 GB unified | macOS | `mlx-vlm` (MLX) |
| **scrappy** | RTX 3070 Laptop | 8 GB VRAM | Windows 11 | — |
| **sparky** | DGX Spark GB10 | 128 GB unified | Ubuntu Linux | offline — artifacts in `spark/` |

## Model Inventory

| Port | Model | Type | Quant | KV Cache | Context | Parallel |
|------|-------|------|-------|----------|---------|----------|
| 2025 | Qwen 3.5 9B | big dense | UD-Q4_K_XL | q8_0 | 64K | 4 |
| 2026 | Qwen 3.5 27B | big dense | UD-Q4_K_XL | q8_0 | 64K | 4 |
| 2027 | Qwen 3.5 35B A3B | MoE | UD-Q4_K_XL | q8_0 | 64K | 8 |
| 2029 | Qwen 3.5 4B | small dense | UD-Q4_K_XL | q8_0 | 64K | 2 |
| 2030 | Qwen 3.5 2B | small dense | Q8_0 | q8_0 | 32K | 2 |
| 2031 | Qwen 3.5 0.8B | small dense | Q8_0 | q8_0 | 32K | 2 |
| 2033 | Nemotron 3 Super 120B A12B | MoE (NVFP4) | NVFP4 | fp8 | 64K | 8 |
| 2034 | Nemotron 3 Nano 30B A3B | MoE (NVFP4) | NVFP4 | fp8 | 64K | 8 |
| 2035 | Nemotron Cascade 2 30B A3B | MoE | UD-Q4_K_XL | q8_0 | 64K | 8 |
| 2036 | Gemma 4 26B-A4B | MoE | UD-Q4_K_XL | q8_0 | 64K | 8 |
| 2037 | Gemma 4 31B | big dense | UD-Q4_K_XL | q8_0 | 64K | 4 |
| 2038 | Gemma 4 E4B | small dense | UD-Q4_K_XL | q8_0 | 64K | 2 |
| 2039 | Gemma 4 E2B | small dense | Q8_0 | q8_0 | 32K | 2 |
| 4007 | Penumbra | custom | — | — | — | — |

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
│   ├── parse-config.py     # Reads model.json → shell variables
│   ├── setup-common.sh     # Shared helpers (CUDA env, venv paths)
│   ├── setup-vllm.sh       # Creates/updates .venv-vllm
│   └── setup-mlx.sh        # Creates/updates .venv-mlx
├── <model-id>/
│   ├── model.json          # All config: ports, quants, engine settings
│   ├── launchd/            # macOS service unit
│   └── systemd/            # Linux service unit
├── .venv-mlx/              # Shared MLX venv (macOS)
├── .venv-vllm/             # Shared vLLM venv (Linux)
├── llama.cpp/              # llama.cpp build scripts
├── whisper.cpp/            # whisper.cpp build scripts
├── bench/                  # Benchmark results
└── spark/                  # DGX Spark artifacts (offline)
```

## Engine Auto-Detection

`run.sh` picks the engine automatically:
- **macOS** → `mlx` (mlx-vlm)
- **ARM Linux without CUDA** → `cpu` (Raspberry Pi)
- **Linux with CUDA** → `llama` (llama.cpp), or `vllm` if model has no GGUF (NVFP4)

Override with `--engine`: `./run.sh qwen-3.5-4b --engine vllm`

## Serving Backends

### llama-server (llama.cpp)
GGUF-quantized models via [llama.cpp](https://github.com/ggerganov/llama.cpp). OpenAI-compatible API at `/v1/chat/completions`. CUDA + flash attention on smarty, Metal on snappy.

### mlx-vlm
Vision Language Models via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). macOS only (Apple Silicon / MLX). Uses `mlx-community/` quantized models. Serves at `/chat/completions` (no `/v1` prefix).

### vLLM
GPU-accelerated serving via [vLLM](https://github.com/vllm-project/vllm). Linux only (CUDA). Supports online FP8 quantization, Marlin NVFP4, and continuous batching for high-throughput concurrent serving.

## Quantization Standards

| Model size | Weight quant | KV cache | Context | Parallel slots |
|------------|-------------|----------|---------|----------------|
| >= 4B | UD-Q4_K_XL | q8_0 / fp8 | 64K | MoE: 8, big dense: 4, small: 2 |
| < 4B | Q8_0 | q8_0 / fp8 | 32K | 2 |

NVFP4 models (Nemotron Nano/Super) use vLLM with Marlin backend instead of llama.cpp.

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
