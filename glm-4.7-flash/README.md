# kortexa.ai GLM 4.7 Flash LLM server

A simple service running GLM 4.7 Flash (Q8_K_XL quant).

GLM-4.7-Flash is a 30B MoE reasoning model from Z.ai (~3.6B active parameters, 200K context).

## Requirements

- 24GB+ VRAM recommended for Q8 quantization
- llama.cpp with llama-server

## Model

Using `unsloth/GLM-4.7-Flash-GGUF` with UD-Q8_K_XL quantization.

## References

- https://unsloth.ai/docs/models/glm-4.7-flash
- https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF

---

© 2025 kortexa.ai
