# Models Server #14 — Hy-MT2 7B translation model

Add Tencent Hy-MT2 7B as a single-slot llama.cpp model using the publisher's
Q4_K_M GGUF, an 8,192-token context, Q8_0 KV cache, and recommended sampling
defaults. Provide standard Linux and macOS service definitions.

Validate the configuration and generic sampling argument plumbing. Benchmark
the same Mandarin-to-English and English-to-Mandarin prompts on Smarty's RTX
PRO 6000 and RTX 4090. Record prompt processing, generation throughput, and
VRAM evidence, then restore the production service roster.
