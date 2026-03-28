# Models Server

This repo is shared between three different machines. Always run `hostname` to check which machine you're on before picking the right way to run a model.

## Machines

| Hostname | Hardware | GPU Memory | Arch | Primary Backend |
|----------|----------|------------|------|-----------------|
| **smarty** | RTX PRO 6000 Blackwell Workstation | 96 GB VRAM | x86_64 Linux | `llama-server` (llama.cpp GGUF) |
| **sparky** | DGX Spark GB10 | 128 GB unified | aarch64 Linux | Docker vLLM, TensorRT-LLM |
| **snappy** | Mac Mini M4 Pro | 64 GB unified | arm64 macOS | `mlx-vlm`, `mlx-lm` |

## Key Differences

- **smarty**: Uses `llama-server` with GGUF quants. No Docker vLLM (`vllm-node` image doesn't exist here).
- **sparky**: Uses Docker-based vLLM (custom `vllm-node` image with Marlin NVFP4 backend) or TensorRT-LLM. Hostname contains "spark" — run.sh scripts detect this via `hostname`.
- **Mac**: Uses MLX-based servers (`mlx-vlm` for vision models, `mlx-lm` for text-only). Detected via `uname -s == Darwin`.

## Adding New Models

Run.sh scripts use platform detection (`uname -s`, `hostname`) to pick the right backend. When adding a new model, check which machines it needs to run on and add the appropriate branches. Not every model needs to support every machine.
