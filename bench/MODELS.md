# Benchmark Model Matrix

This is the planned post-reset benchmark scope. The first sweep focuses on
models that are current, newly added, or operationally important.

## Priority CUDA generation and vision sweep

| Model ID | Workload | Required comparisons |
|---|---|---|
| `qwen-3.8-27b` | text, tools, long context | Production MTP configuration; 450 W baseline; optional paired 600 W run |
| `qwen-3.8-27b-uncensored` | text, tools, long context | No-MTP Q4_K_M baseline; keep separate from the standard model |
| `gemma-4-e2b` | text, tools, vision | Small Gemma baseline |
| `gemma-4-e4b` | text, tools, vision | New small Gemma baseline |
| `lfm2.5-230m` | text | CUDA Q8_0; CPU/Pi results belong in a separate table |
| `lfm2.5-1.2b-instruct` | text | New instruct baseline |
| `lfm2.5-1.2b-thinking` | text, reasoning | Thinking and non-thinking workloads must be labeled separately |
| `lfm2.5-2.6b` | text, tools | Replaces the unauditable historical run |
| `lfm2.5-vl-450m` | text, vision | 512px and 1536px fixtures; CUDA and CPU are separate runs |
| `lfm2.5-vl-3b` | text, tools, vision | New 3B VLM baseline |

## Separate CPU and task-specific suites

These models should not be mixed into the CUDA generation ranking:

| Model ID | Suite |
|---|---|
| `lfm2.5-350m` | CPU generation and long-context behavior |
| `lfm2-350m-extract` | CPU extraction accuracy plus throughput |
| `lfm2.5-embedding-350m` | Embedding latency, throughput, dimension, and retrieval quality |
| `embeddinggemma-300m` | Embedding latency, throughput, dimension, and retrieval quality |
| `qwen3-embedding-0.6b` | Embedding latency, throughput, dimension, and retrieval quality |

`lfm2.5-encoder-350m` uses Transformers rather than llama.cpp and needs a
masked-language-model suite. It is outside this llama benchmark reset.

## Archived, not scheduled for the first sweep

- All Qwen 3.5 and Qwen 3.6 records are obsolete.
- Gemma 4 26B-A4B and Gemma 4 31B records are obsolete.
- Gemma 4 12B is deferred; add it only when there is a current operational
  reason to compare the middle-size model.
- DGX Spark results are obsolete. If that host returns, create a new machine
  baseline and rerun from scratch. Never compare its unified-memory GPU numbers
directly with smarty's discrete RTX PRO 6000 results.
