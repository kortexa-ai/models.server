# Models Server

Run `hostname` to check which machine you're on before doing anything.

## Machines

| Hostname | Hardware | GPU Memory | OS | Status |
|----------|----------|------------|------|--------|
| **smarty** | RTX PRO 6000 Blackwell | 96 GB VRAM | Ubuntu Linux | active |
| **snappy** | Mac Mini M4 Pro | 64 GB unified | macOS | active |
| **scrappy** | RTX 3070 Laptop | 8 GB VRAM | Windows 11 | active |
| **sparky** | DGX Spark GB10 | 128 GB unified | Ubuntu Linux | offline — see `spark/` |

## How It Works

- `run.sh` is the single entry point — auto-detects platform and dispatches to the right engine
- All model config lives in `<model-id>/model.json` — engines, quants, ports, KV budgets
- Generic engine scripts live in `scripts/` — they read model.json, not hardcoded values
- Usage: `./run.sh qwen-3.5-4b` or `cd qwen-3.5-4b && ../run.sh`
- Override engine: `./run.sh qwen-3.5-4b --engine vllm`

## Rules

- **Never kill running processes** — if a port is busy, ask. Don't touch what you didn't start.
- **Never start servers** — the user starts them. Just curl the running endpoints.
- Quantization: >= 4B → `UD-Q4_K_XL`, < 4B → `Q8_0`
- KV cache: `q8_0` (llama.cpp) / `fp8` (vLLM) everywhere
- Context: >= 4B → 64K, < 4B → 32K
- Parallel: MoE → 8, big dense → 4, small dense → 2
