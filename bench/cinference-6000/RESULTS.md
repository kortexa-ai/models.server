# Qwen3.8-27B on the RTX PRO 6000 Blackwell

Measured on Smarty on 2026-09-21–22 Pacific time (2026-09-22 UTC), at the normal
450 W GPU power limit. Work item: [models.server#27](https://github.com/kortexa-ai/models.server/issues/27).

## Vanilla cold-prefill and cache comparison

The non-abliterated [NVFP4/FP8 `.ninfer` checkpoint](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)
runs in Cinference. It combines the official model with its published NVFP4
quantization; it is not a Huihui or Swift checkpoint.

All native profiles below use eight slots, 524,288 shared KV tokens, a
262,144-token request ceiling, vision enabled, and an 8,192-token prefill chunk.
The requests are cold, non-thinking prose with a 512-token output cap.

| Configuration | 131K TTFT | 260K TTFT | 131K decode | 260K decode | Process VRAM |
|---|---:|---:|---:|---:|---:|
| Stock llama.cpp, q8_0, MTP-3 | 79.43 s | 233.98 s | 66.2 tok/s | 44.4 tok/s | 44.27 GiB |
| Vanilla Cinference, INT8, MTP-3 | 33.68 s | 105.39 s | 112.3 tok/s | 90.6 tok/s | 41.92 GiB |
| Vanilla Cinference, K8V4, MTP-3 | 30.54 s | 92.60 s | 116.3 tok/s | 99.6 tok/s | 37.72 GiB |
| Vanilla Cinference, K8V4, MTP-10 | 30.50 s | 92.60 s | 99.9 tok/s | 85.1 tok/s | 37.86 GiB |
| Vanilla Cinference, K8V4, DFlash2-7 | 30.53 s | 92.73 s | 121.9 tok/s | 124.2 tok/s | 39.77 GiB |

Changing only INT8 to K8V4 at MTP-3 saves 4.20 GiB of process VRAM and reduces
260K prefill by 12%. The native KV payload changes from 17.53 to 13.35 GiB.
DFlash2 adds about 1.91 GiB over MTP-10 or 2.05 GiB over MTP-3 in this setup.
Its draft weights are in the same artifact and load when that backend is selected.

The INT8 chunk sweep measured the following prompt-processing times. These
exclude client overhead, so they are slightly below TTFT.

| Prefill chunk | 131K prefill | 260K prefill | Process VRAM |
|---:|---:|---:|---:|
| 1,024 | 37.08 s | 117.78 s | 41.31 GiB |
| 4,096 | 34.01 s | 106.74 s | 41.34 GiB |
| 8,192 | 33.62 s | 105.29 s | 41.92 GiB |

The 8K step improves the combined prompt time by only 1.3% over 4K, so the
bounded tuner does not extend to 16K/32K. MTP depth mainly affects decode:
MTP-3 and MTP-10 have nearly identical prefill in the matched K8V4 runs.
Longer proposals can lose throughput when acceptance is low.

## Stock and Huihui controls

All values below are cold requests with zero cached prompt tokens. TTFT is
client time to the first streamed token. Decode is the server's decode-phase
rate. Prose requests produce 512 tokens; recall requests stop after their answer.

| Prompt tokens | Stock prose TTFT | Huihui fast prose TTFT | Stock prose decode | Huihui fast prose decode | Huihui fast recall decode |
|---:|---:|---:|---:|---:|---:|
| 8,147 | 3.02 s | 0.99 s | 113.3 tok/s | 128.7 tok/s | 463.7 tok/s |
| 32,775 | 12.76 s | 4.58 s | 104.9 tok/s | 117.3 tok/s | 457.3 tok/s |
| 131,033 | 79.43 s | 31.85 s | 66.2 tok/s | 90.6 tok/s | 357.2 tok/s |
| 259,992 | 233.98 s | 101.58 s | 44.4 tok/s | 80.2 tok/s | 278.8 tok/s |

Recall prompts differ by one token from prose except at 8K (one token fewer).
The fast profile uses one lane, a 262,144-token shared KV pool, K8V4 cache,
MTP-10, a 1,024-token prefill chunk, and no vision residency.

At 260K, Huihui with the stock-sized pool and INT8 cache takes 117.93 s to the
first prose token and decodes at 90.5 tok/s. This profile has eight lanes,
524,288 shared KV tokens, MTP-3, vision enabled, and a 1,024-token prefill chunk.

| Profile | Peak process VRAM | Shared KV tokens | Slots | Cache | Vision |
|---|---:|---:|---:|---|---|
| Stock llama.cpp | 44.27 GiB | 524,288 | 8 | q8_0 K and V | enabled |
| Huihui Cinference, stock-sized | 41.31 GiB | 524,288 | 8 | INT8 K and V | enabled |
| Huihui Cinference, fast | 27.43 GiB | 262,144 | 1 | FP8 K / NVFP4 V | disabled |

These are process values from NVIDIA telemetry, not whole-card totals. The
separate vision service remained resident at about 14.5 GiB. The 27.43 GiB
result changes pool size, slot count and vision as well as cache precision;
it cannot establish the saving from K8V4 alone.

Eight simultaneous cold 8K prose requests produced 4,096 output tokens in
45.01 s on stock and 15.58 s on stock-sized Huihui. Those are end-to-end batch
rates of 91.0 and 262.9 output tok/s, including prompt processing and queueing.
They are not single-user decode rates.

## Coding and short conversation

The coding prompt contains a deterministic snapshot of the pinned runtime's
source and requests a Python JSONL metrics reader. The medium-reasoning runs
used all 2048 output tokens in reasoning and returned no visible code. Their
rates below describe the reasoning phase of a coding request. A separate
reasoning-disabled control emitted 1024 code tokens and also reached its cap;
this is a throughput probe, not a completed or correctness-scored solution.

| Workload | Stock TTFT | Vanilla DFlash2 TTFT | Stock decode | DFlash2 decode |
|---|---:|---:|---:|---:|
| Coding, medium reasoning, 131K | 77.84 s | 30.20 s | 93.2 tok/s | 230.4 tok/s |
| Coding, medium reasoning, 260K | 239.59 s | 91.24 s | 61.9 tok/s | 214.6 tok/s |
| Code output, reasoning off, 131K | 82.96 s | 30.86 s | 97.6 tok/s | 275.8 tok/s |

At 131K with reasoning, the other native profiles decoded at 147.7 tok/s
(INT8 MTP-3), 153.4 tok/s (K8V4 MTP-3), and 202.0 tok/s (K8V4 MTP-10).
DFlash2 won the predeclared selection: lowest cold prefill, then fastest decode
among profiles within 5% of that prefill time. Small prompt-token differences
between engines are retained in [the workload evidence](results-workloads.json).

Five short spoken-style prompts with reasoning disabled had median visible
TTFT of 196 ms on stock and 65 ms on DFlash2. DFlash2 decode ranged from
109.5 to 156.3 tok/s. These measure the LM endpoint only, excluding ASR, TTS,
network routing, and the rest of the voice pipeline.

A fresh prose confirmation measured 30.60 s / 121.8 tok/s at 131K and
92.86 s / 124.2 tok/s at 260K. Changing the 260K prefix produced 93.00 s /
108.8 tok/s, again with zero cached tokens. A 32K medium-thinking probe reached
251.3 tok/s, but emitted only reasoning within its 512-token cap.
Eight simultaneous cold 8K requests produced 4096 output tokens in 14.12 s,
an aggregate end-to-end rate of 290.2 tok/s, not a single-request decode rate.

With exact prefix reuse, a 131K repeat returned its first token in 109 ms:
131029 tokens were reused and only nine were evaluated. The cold seed took
31.03 s. Keep this warm-cache result separate from the cold-prefill figures.

## Shared cache and precision

Cinference's `--kv-capacity` is one shared Main Text KV pool used by active
requests and retained prefixes. `--max-context` is the per-request ceiling.
Eight slots do not allocate eight full-context KV caches. There is additional
per-lane continuation state and speculative/workspace storage. Requests reserve
prompt-plus-output capacity and can queue when the shared pool is full.
See the [pinned serving contract](https://github.com/satellitedown/cinference/blob/b74044fb0a319cd2a737cb7108012e6344b96dac/docs/serving.md#resource-scheduling-and-context-cache).

The stock configuration uses llama.cpp `--kv-unified` with a 524,288-token
shared pool. Its live API and rejected 499,954-token probe establish an actual
262,144-token per-request ceiling. All Cinference profiles use that same
ceiling. CUDA unified memory is a separate feature and was not enabled.

In this Cinference revision, K8V4 means row-scaled FP8 keys and group-16 NVFP4
values. It applies a normalized Hadamard transform to both. INT8 mode uses
group-64 scaling. Thus even its 8-bit cache is not the same codec as llama.cpp
q8_0. See the [cache encoding contract](https://github.com/satellitedown/cinference/blob/b74044fb0a319cd2a737cb7108012e6344b96dac/include/ninfer/ops/kv_cache_append.h).

The vanilla artifact's [published evaluations](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/blob/f0b43ad436b9fa8142c6ed6647c470a6fe409484/README.md#evaluation)
use INT8 KV. They do not validate this K8V4 configuration.

All control recall cases recovered all six markers, including markers near
both ends of 260K prompts. This is a limited correctness check. It does not
measure reasoning accuracy, broad task quality, or image quality. K8V4 can
change answers; these measurements do not establish quality parity with INT8.

## Method and provenance

- Stock: the checked-in Qwen3.8-27B `model.json`, UD-Q4_K_XL weights, q8_0 K/V,
  eight slots, shared 512K KV, MTP-3, vision enabled, CUDA graphs disabled by
  its normal launcher. llama.cpp revision `434ddbbc0e30522e897670681e503b797c12b7c1`.
- Cinference: revision `b74044fb0a319cd2a737cb7108012e6344b96dac`, built for
  `sm_120a` using isolated CUDA 13.4.92. Its CUDA graph reuse remains enabled.
- Huihui: non-Swift artifact revision `12fe3b16d82541dad06288b0da629933efebb6cd`,
  SHA-256 `3876a052290d48171350a299eac3bcdbfeeac69f58efd11747d7f9509f7b54ee`.
- Vanilla: original post-trained weights in mixed NVFP4/FP8 `.ninfer` form,
  artifact revision `f0b43ad436b9fa8142c6ed6647c470a6fe409484`, SHA-256
  `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`.
- GPU UUID `GPU-a71210ca-e14a-755a-88bb-77f53a2102f6`, driver 610.43.02.
  A monitor enforces 10 GiB free VRAM. No benchmark uses the 4090.

The deterministic archive fixtures are identical across engines. Main requests
use temperature zero, seed 42, thinking disabled, and at most 512 output tokens.
Vanilla explicitly sets presence/frequency penalties to zero to match stock.
Huihui used Cinference's non-thinking default presence penalty of 1.5. Its
decode comparison is therefore not a controlled checkpoint-only experiment.
Weight formats and inference kernels also differ between engines.

High speculative acceptance makes marker recall much faster than prose.
The launcher's [linked RTX 5090 raw archive-recall record](https://github.com/satellitedown/fast-long-context-cinference/blob/104b236e557741b990ff084ef6512801689c9613/results/rtx5090-archive-recall.json) identifies
Swift weights, so those figures are not evidence for non-Swift Huihui or for
arbitrary generation. Our own Huihui recall result does exceed 250 tok/s at
260K; its prose result does not.

Native decode timing excludes the first token, which belongs to prefill.
Most measurements are one observation per case. They are performance probes,
not confidence intervals. Thinking and cache-hit requests must be reported
separately from the cold non-thinking figures above.

Compact evidence: [stock](results-stock.json), [Huihui](results-huihui.json),
[vanilla tuning](results-vanilla.json), [coding and conversation](results-workloads.json).
Raw fixtures, output text, logs and half-second GPU samples remain on Smarty
under `/home/francip/src/models.server/bench-results/cinference-27`.
The first native attempts failed to bind during the cross-engine socket
handoff; their errors are retained. The Huihui subdirectory contains the
successful rerun after the bounded socket-state wait was added.
