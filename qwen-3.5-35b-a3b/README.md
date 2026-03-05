# kortexa.ai Qwen 3.5 35B A3B LLM server

A simple service running Qwen 3.5 35B MoE (~3B active parameters).

Platform-aware quantization: Q4_K_XL on macOS (Mac Mini M4 Pro), Q8_K_XL on Linux (RTX 6000 Blackwell).

## Requirements

- llama.cpp with llama-server
- macOS: 16GB+ unified memory
- Linux: 24GB+ VRAM for Q8

## Model

Using `unsloth/Qwen3.5-35B-A3B-GGUF` with platform-specific quantization.

## References

- https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF

---

© 2025 kortexa.ai
