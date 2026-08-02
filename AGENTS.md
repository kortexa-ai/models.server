# Models Server

Run `hostname` to check which machine you're on before doing anything.

## Machines

| Hostname | Hardware | Memory | OS | Status |
|----------|----------|------------|------|--------|
| **smarty** | RTX PRO 6000 Blackwell | 96 GB VRAM | Ubuntu Linux | active |
| **snappy** | Mac Mini M4 Pro | 64 GB unified | macOS | active |
| **scrappy** | RTX 3070 Laptop | 8 GB VRAM | Windows 11 | active |
| **sparky** | DGX Spark GB10 | 128 GB unified | Ubuntu Linux | offline |
| **192.168.2.144** | Raspberry Pi 5 | 8 GB RAM | ARM Linux | active |
| **192.168.2.145** | Raspberry Pi 5 | 8 GB RAM | ARM Linux | active |

## How It Works

- `run.sh` is the single entry point — auto-detects platform and dispatches to the right engine
- All model config lives in `<model-id>/model.json` — engines, quants, ports, KV budgets
- Generic engine scripts live in `scripts/` — they read model.json, not hardcoded values
- Usage: `./run.sh qwen-3.5-4b` or `cd qwen-3.5-4b && ../run.sh`
- Override engine: `./run.sh qwen-3.5-4b --engine vllm`

## Rules

- Inspect service and port ownership before changing a running model. Use
  `ktxsvc` for managed services and leave unrelated processes alone.
- Prefer checking existing endpoints. Start or stop a model only when the task
  requires it, and never create a duplicate server.
- Quantization: >= 4B → `UD-Q4_K_XL`, < 4B → `Q8_0`
- Exception: LFM2.5 230M uses `Q4_K_M` for Pi CPU serving
- KV cache: `q8_0` (llama.cpp) / `fp8` (vLLM) everywhere
- Context: max supported by the model
- Parallel: MoE → 8, dense → 1
