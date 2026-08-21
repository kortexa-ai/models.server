# Benchmark Model Matrix

This is the post-reset benchmark scope and sweep status. Results are in
[`BENCHMARKS.md`](BENCHMARKS.md).

## Priority CUDA generation and vision sweep

| Model ID | Workload | Status / required comparisons |
|---|---|---|
| `qwen-3.8-27b` | text, tools, long context | 450 W production MTP baseline complete; optional paired 600 W run remains |
| `qwen-3.8-27b-uncensored` | text, tools, vision, long context | Pending fresh OrcaRouter Q4_K_M MTP-3 baseline and matched no-MTP comparison |
| `ornith-1.5-9b` | text, tools, vision, long context | 450 W MTP-2 baseline complete; matched official no-MTP and depth sweep retained |
| `ornith-1.5-35b-a3b` | text, tools, vision, long context | 450 W no-MTP baseline complete; native MTP lost its matched smoke |
| `qwen-3.5-9b` | text, tools, vision, long context | 450 W one-slot baseline complete; selected GGUF has no MTP head |
| `qwen-3.6-35b-a3b` | text, tools, vision, long context | 450 W one-slot MTP-3 baseline complete; matched no-MTP smoke retained |
| `gemma-4-e2b` | text, tools, vision | 450 W baseline complete |
| `gemma-4-e4b` | text, tools, vision | 450 W baseline complete |
| `gemma-4-12b` | text, tools, vision | 450 W baseline complete |
| `gemma-4-26b-a4b` | text, tools, vision, long context | 450 W one-slot baseline complete; llama GGUF has no MTP head |
| `lfm2.5-230m` | text | Native-context four-slot CUDA baseline complete; old oversized-context record retained; CPU/Pi results remain separate |
| `lfm2.5-350m` | text | 450 W CUDA Q8_0 baseline complete; CPU/Pi results remain separate |
| `lfm2.5-1.2b-instruct` | text | 450 W baseline complete |
| `lfm2.5-1.2b-thinking` | text, reasoning | 450 W serving baseline complete; future quality work must label thinking mode |
| `lfm2.5-2.6b` | text, tools | 450 W baseline complete |
| `lfm2.5-vl-450m` | text, vision | Four-slot CUDA baseline complete; old one-slot record and tool-call canary failure retained |
| `lfm2.5-vl-3b` | text, tools, vision | 450 W baseline complete |

## Separate CPU and task-specific suites

These models should not be mixed into the CUDA generation ranking:

| Model ID | Suite |
|---|---|
| `lfm2.5-350m` | Oversized-context CPU result retained as a failure record; native-context CUDA baseline is in the primary sweep |
| `lfm2-350m-extract` | Serving smoke complete; extraction accuracy and task throughput pending |
| `lfm2.5-embedding-350m` | Endpoint canary complete; embedding throughput and retrieval quality pending |
| `embeddinggemma-300m` | Embedding latency, throughput, dimension, and retrieval quality |
| `qwen3-embedding-0.6b` | Embedding latency, throughput, dimension, and retrieval quality |

`lfm2.5-encoder-350m` uses Transformers rather than llama.cpp and needs a
masked-language-model suite. It is outside this llama benchmark reset.

## Archived, not scheduled for the first sweep

- Qwen 3.5 9B and Qwen 3.6 35B-A3B now have current one-slot records; other
  Qwen 3.5 and Qwen 3.6 records remain obsolete.
- Gemma 4 26B-A4B now has a current one-slot record; Gemma 4 31B remains
  obsolete.
- DGX Spark results are obsolete. If that host returns, create a new machine
  baseline and rerun from scratch. Never compare its unified-memory GPU numbers
  directly with smarty's discrete RTX PRO 6000 results.
