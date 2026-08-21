# Benchmark Model Matrix

This is the post-reset benchmark scope and sweep status. Results are in
[`BENCHMARKS.md`](BENCHMARKS.md).

## Priority CUDA generation and vision sweep

| Model ID | Workload | Status / required comparisons |
|---|---|---|
| `qwen-3.8-27b` | text, tools, long context | 450 W production MTP baseline complete; optional paired 600 W run remains |
| `qwen-3.8-27b-uncensored` | text, tools, long context | Pending; no-MTP Q4_K_M must remain separate from the standard model |
| `gemma-4-e2b` | text, tools, vision | 450 W baseline complete |
| `gemma-4-e4b` | text, tools, vision | 450 W baseline complete |
| `gemma-4-12b` | text, tools, vision | 450 W baseline complete |
| `lfm2.5-230m` | text | 450 W CUDA Q8_0 baseline complete; CPU/Pi results remain separate |
| `lfm2.5-350m` | text | 450 W CUDA Q8_0 baseline complete; CPU/Pi results remain separate |
| `lfm2.5-1.2b-instruct` | text | 450 W baseline complete |
| `lfm2.5-1.2b-thinking` | text, reasoning | 450 W serving baseline complete; future quality work must label thinking mode |
| `lfm2.5-2.6b` | text, tools | 450 W baseline complete |
| `lfm2.5-vl-450m` | text, vision | 450 W baseline complete; tool-call canary failure recorded |
| `lfm2.5-vl-3b` | text, tools, vision | 450 W baseline complete |

## Separate CPU and task-specific suites

These models should not be mixed into the CUDA generation ranking:

| Model ID | Suite |
|---|---|
| `lfm2.5-350m` | CPU baseline complete; 32K collapse documented; CUDA baseline is in the primary sweep |
| `lfm2-350m-extract` | Serving smoke complete; extraction accuracy and task throughput pending |
| `lfm2.5-embedding-350m` | Endpoint canary complete; embedding throughput and retrieval quality pending |
| `embeddinggemma-300m` | Embedding latency, throughput, dimension, and retrieval quality |
| `qwen3-embedding-0.6b` | Embedding latency, throughput, dimension, and retrieval quality |

`lfm2.5-encoder-350m` uses Transformers rather than llama.cpp and needs a
masked-language-model suite. It is outside this llama benchmark reset.

## Archived, not scheduled for the first sweep

- All Qwen 3.5 and Qwen 3.6 records are obsolete.
- Gemma 4 26B-A4B and Gemma 4 31B records are obsolete.
- DGX Spark results are obsolete. If that host returns, create a new machine
  baseline and rerun from scratch. Never compare its unified-memory GPU numbers
  directly with smarty's discrete RTX PRO 6000 results.
