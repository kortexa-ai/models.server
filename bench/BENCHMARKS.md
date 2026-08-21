# Benchmark Results Log

This log starts on 2026-08-20 with the reproducible benchmark workflow in
[`README.md`](README.md). Results recorded before the reset are preserved in
[`archive/BENCHMARKS-2026-08-20.md`](archive/BENCHMARKS-2026-08-20.md).

Do not copy old throughput numbers into this file. Add a result only when its
run directory contains a manifest, the exact `model.json`, the live process
arguments and safe runtime environment, GPU telemetry, and raw suite output.

## Results

### 2026-08-20 llama.cpp rebaseline on smarty

All results below used the production `llama-server` path on the RTX PRO 6000
Blackwell at a 450 W GPU power cap. CUDA graphs were disabled for CUDA models,
`GGML_CUDA_ENABLE_UNIFIED_MEMORY` was absent from every live process, and the
llama.cpp revision was
`9ee9fc04c136ef2ae729bfc60d18961b23c13ddf` (build 10524). SPEED-Bench used
dataset revision `487aa718444e816458d1a0a52bfce7a454285cf4`.

Numbers are average prompt-processing/decode tokens per second. A dash means
the workload did not fit one configured slot or did not apply. The raw local
records are under `bench-results/20260820-rebaseline/`; each contains the exact
`model.json`, live arguments and environment, executable hash, endpoint
properties, telemetry, canary responses, and raw samples.

| Model | Slots x context | Canaries | Qualitative | 1K / 512 | 8K / 512 | 32K / 512 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 3.8 27B | 2 x 131,072 | 5/5 | 675.91 / 130.82 | 1,608.31 / 124.86 | 1,841.97 / 121.74 | 1,344.60 / 111.44 |
| LFM2.5 VL 3B | 1 x 32,768 | 5/5 | 12,416.02 / 367.26 | 17,915.99 / 365.95 | 19,797.50 / 358.49 | - |
| LFM2.5 2.6B | 1 x 128,000 | 3/3 | 8,222.23 / 369.39 | 18,343.62 / 367.40 | 20,231.20 / 360.68 | 13,959.13 / 342.10 |
| LFM2.5 1.2B Thinking | 1 x 32,768 | 3/3 | 26,596.34 / 753.82 | 40,431.23 / 750.01 | 44,281.69 / 725.21 | - |
| LFM2.5 1.2B Instruct | 1 x 32,768 | 3/3 | 26,044.24 / 754.12 | 40,534.31 / 749.79 | 44,286.17 / 726.02 | - |
| LFM2.5 VL 450M | 4 x 32,768 | 4/5 | 36,370.39 / 1,329.26 | 63,884.66 / 1,313.42 | 77,864.78 / 1,256.12 | - |
| LFM2.5 350M CPU (old oversized context) | 4 x 128,000 | 3/3 | 1,121.77 / 84.37 | 578.15 / 81.19 | 300.78 / 62.42 | 36.58 / 19.46* |
| LFM2.5 350M GPU | 4 x 32,768 | 3/3 | 37,691.80 / 1,331.24 | 65,516.10 / 1,322.71 | 78,755.32 / 1,260.89 | - |
| LFM2.5 230M | 4 x 32,768 | 3/3 | 46,351.55 / 1,585.85 | 79,229.18 / 1,578.04 | 96,064.82 / 1,488.25 | - |
| Gemma 4 E4B | 1 x 131,072 | 5/5 | 3,424.09 / 207.91 | 9,040.24 / 206.36 | 10,082.42 / 198.35 | 6,888.33 / 182.88 |
| Gemma 4 E2B | 1 x 131,072 | 5/5 | 4,709.95 / 267.21 | 13,780.48 / 265.17 | 16,699.43 / 254.73 | 10,578.04 / 244.68 |
| Gemma 4 12B | 1 x 131,072 | 5/5 | 1,746.52 / 128.20 | 3,725.20 / 126.21 | 4,202.36 / 123.28 | 3,015.35 / 120.07 |

Qwen's two-client pass delivered 128.86 aggregate completion tok/s, 81.39
average per-request decode tok/s, and 7.05 s average latency. The current
LFM2.5 230M four-client pass delivered 1,511.38 aggregate completion tok/s,
1,052.56 average per-request decode tok/s, and 0.63 s average latency. The current
LFM2.5 350M GPU four-client pass delivered 1,406.79 aggregate completion tok/s,
928.87 average per-request decode tok/s, and 0.72 s average latency. Its CPU pass
recorded 62.21 average per-request decode tok/s and 10.71 s average latency;
that direct wrapper run predates aggregate-wall-time capture. GPU offload was
about 15.8x faster on qualitative decode, though the retained CPU run used the
old oversized context allocation.

The LFM2.5 350M CPU 32K entry marked `*` is one bounded high-entropy sample,
not the normal 15-sample aggregate. The standard pass was stopped after the
first request exposed a nonlinear collapse; that sample took 270.32 s. The
completed records were kept as evidence of the bad configuration. The model
now uses its native 32K per slot, and run `15` is the current CUDA baseline.

The original 230M run `10` used four oversized 128K slots; run `16` supersedes
it with four native 32K slots. The original VL 450M run `06` used one 32K slot;
run `17` supersedes it with four. Their single-client throughput remained
effectively unchanged. VL 450M's four-client pass delivered 1,396.72 aggregate
completion tok/s, 924.52 average per-request decode tok/s, and 0.72 s average
latency.

LFM2 350M Extract received a preliminary three-sample CPU serving smoke test:
687.40 prompt tok/s, 83.10 decode tok/s, and 2.40 s latency. It passed 3/3 text
canaries, but needs a task-specific structured-extraction quality suite before
models are compared. LFM2.5 Embedding 350M passed its batch-embedding endpoint
canary; generation SPEED-Bench does not apply, and retrieval quality plus
embedding batch-throughput tests remain to be designed.

LFM2.5 VL 450M passed text cold/warm and both 512 px and 1536 px vision
canaries. Its required-tool canary failed with HTTP 400 because llama-server
could not parse the generated grammar. The failure is part of the result.

#### Provenance index

| Run directory | Model config SHA-256 |
|---|---|
| `01-qwen-3.8-27b` | `5cca37848c4512dc2eefc5ca540e805922c3e16af660e6864aef591dd37e9e26` |
| `02-lfm2.5-vl-3b` | `413d5097da33ceb4213d7929732f70c21b34dfa4345823e3a2a726e525d41d00` |
| `03-lfm2.5-2.6b` | `f0f569f4c6f7b7ff9c440a9b0a26f449b37e2ba9042a5a3f5252096449b33c50` |
| `04-lfm2.5-1.2b-thinking` | `4fa1a525acf29b8687c0797bb3a7c636d4430a23333b2be2c3fbd864cd20bf2a` |
| `05-lfm2.5-1.2b-instruct` | `ebde37bb7c793817f4257a70128738207aef18a58aba61fdc06f3ac9a7baa999` |
| `06-lfm2.5-vl-450m` | `bac727f5f8170f059c8f67296ed5bff0283c17f62dd1b1435eb49c68def4fcac` |
| `07-lfm2.5-350m-cpu` | `d54dd72a75c6bd73b2667e9c7bf4e267b80be044bf7db93a1c4805cf299ec5f8` |
| `08-lfm2-350m-extract-cpu-smoke` | `e10548824565a3e3a51d5f4aa79dd05f2e48c09d931427db9e60d15967e961de` |
| `09-lfm2.5-embedding-350m` | `469dcb0c771b37c43c6fcc3733dc9422dfcedb11c93dc66730265da0dfd57eaa` |
| `10-lfm2.5-230m` | `007fc40b7be2c7f9b030524418505365196d4662cd3fc17d8d9a2c4142797b48` |
| `11-gemma-4-e4b` | `aae7ba98a88983a2209b8914fe1a16585485b0c7724255ae93d3e5293d5755f2` |
| `12-gemma-4-e2b` | `359d189c91e46e30e2d06dea8e51f06b94c5d74fed615ecf49091de40f247b64` |
| `13-gemma-4-12b` | `5520690b0e6d13eb085b3e5c7f9811dca5159b1f79afc9e718672204b7895bc2` |
| `14-lfm2.5-350m-gpu` | `e70e52784c7ca3f03ffc8bcc5fc371a58265bae183bde2419b864f4bb349770d` |
| `15-lfm2.5-350m-gpu-native-context` | `6502611b316b7ec6a6c47bc24f5d1877f0ad0e0748cf0e09e7ed3399e74eaab4` |
| `16-lfm2.5-230m-native-context` | `08fc0ccf89bf33f3191327c6ce10e87d211e1e60ed1bccc66032e4befc0ffded` |
| `17-lfm2.5-vl-450m-four-slot` | `f7338481d62344ad48b8b7781bd3d790fca89c3b5cebe092a4b37b3b2dc62e58` |

### 2026-08-21 agentic family rebaseline on smarty

These six results use a common single-request workstation profile: one slot,
Q4-class weights, q8_0 K/V cache, the selected GGUF's maximum usable context,
450 W, and disabled CUDA graphs. `GGML_CUDA_ENABLE_UNIFIED_MEMORY` was absent
from every live process. Gemma 4 12B is the only context exception because its
selected GGUF reports a 131,072-token trained context; the other five use
262,144. All models passed 5/5 text, required-tool, and vision canaries.

The llama.cpp revision was
`9ee9fc04c136ef2ae729bfc60d18961b23c13ddf` (build 10524), and SPEED-Bench
used dataset revision `487aa718444e816458d1a0a52bfce7a454285cf4`.
Raw local records are under `bench-results/20260821-agentic-layer/`.

Numbers are average prompt-processing/decode tokens per second.

| Model | Quant / speculative mode | Slots x context | Qualitative | 1K / 512 | 8K / 512 | 32K / 512 |
|---|---|---:|---:|---:|---:|---:|
| Ornith 1.5 9B | Q4_K_M, MTP-2 | 1 x 262,144 | 1,799.63 / 293.29 | 4,789.12 / 279.37 | 5,804.77 / 274.07 | 4,262.75 / 247.34 |
| Gemma 4 12B | UD-Q4_K_XL, no MTP | 1 x 131,072 | 1,750.43 / 128.39 | 3,744.73 / 127.30 | 4,331.79 / 124.56 | 3,115.97 / 121.47 |
| Ornith 1.5 35B-A3B | Q4_K_M, no MTP | 1 x 262,144 | 1,485.65 / 263.49 | 4,314.67 / 264.06 | 4,837.57 / 258.35 | 3,509.52 / 241.60 |
| Qwen 3.5 9B | UD-Q4_K_XL, no MTP | 1 x 262,144 | 2,138.75 / 194.75 | 6,153.62 / 194.13 | 6,536.62 / 190.37 | 4,791.91 / 180.63 |
| Qwen 3.6 35B-A3B | UD-Q4_K_XL, MTP-3 | 1 x 262,144 | 1,279.06 / 323.95 | 3,782.22 / 313.83 | 4,711.82 / 302.22 | 3,430.62 / 290.79 |
| Gemma 4 26B-A4B | UD-Q4_K_XL, no MTP | 1 x 262,144 | 2,291.07 / 194.26 | 5,644.32 / 192.88 | 6,649.21 / 187.25 | 4,657.99 / 176.63 |

#### Speculative-decoding selections

Ornith 1.5 9B's official checkpoint does not contain its declared MTP head.
The selected protoLabs distilled-head GGUF was tested at depths 2, 3, and 4.
Depth 2 won at 314.14 decode tok/s with 74.5% acceptance, versus 202.51 for
the official no-MTP GGUF: a 55.1% improvement. Depths 3 and 4 reached 296.42
and 283.01 tok/s.

Ornith 1.5 35B-A3B's native MTP head lost its matched smoke: 234.05 tok/s and
32.5% acceptance versus 261.72 tok/s without MTP. MTP remains explicitly off.
Qwen 3.6 35B-A3B's depth-3 head won its match at 357.80 tok/s and 81.0%
acceptance versus 237.99 without MTP, a 50.3% gain. MTP remains on at an
explicit depth 3. Qwen 3.5 9B and both selected Gemma GGUFs have no compatible
llama.cpp draft implementation configured.

#### Provenance index

| Run directory | Model config SHA-256 |
|---|---|
| `01-ornith-1.5-9b-mtp2` | `eff3200e756dab64470b0834cff673fe5c1afbc9372b2ea09cedc8ef549f04da` |
| `02-gemma-4-12b` | `5520690b0e6d13eb085b3e5c7f9811dca5159b1f79afc9e718672204b7895bc2` |
| `03-ornith-1.5-35b-a3b-no-mtp` | `76e829c7f3ac3c9e9ab9cc0ba7de052ec1ee8c43c78853cd5ee72543d8fdf57f` |
| `04-qwen-3.5-9b` | `10f3acb23fc0c9c4bff92dad974ce014c324a98640ceeeaaa0786334ae443118` |
| `05-qwen-3.6-35b-a3b-mtp3` | `087e7fbdc2fb345cd85530320c5fcf267f74a854194be072097e4a4bcd216313` |
| `06-gemma-4-26b-a4b` | `f5780f90d15e518b4dfb7fc26581642646bb09d969c0a933f671759eb1ad33a5` |

### 2026-08-21 Qwen 3.8 vanilla versus uncensored on smarty

This is a paired comparison on llama.cpp revision
`3af988fabcf79fd81f8720505e684d2aa5bfc786` (build 10572), at 450 W, with
CUDA graphs disabled and `GGML_CUDA_ENABLE_UNIFIED_MEMORY` absent. Both models
used two 131,072-token slots, q8_0 K/V cache, MTP depth 3, and identical
runtime arguments other than model identity and weights. Vanilla used
Unsloth `UD-Q4_K_XL`; uncensored used OrcaRouter `Q4_K_M`, so this measures
the selected production artifacts rather than isolating ablation effects.
Both models passed all five text, required-tool, and vision canaries.

Numbers are average prompt-processing/decode tokens per second. The delta is
uncensored relative to vanilla.

| Workload | Vanilla | Uncensored | Decode delta |
|---|---:|---:|---:|
| Qualitative | 675.29 / 124.28 | 670.54 / 123.00 | -1.0% |
| 1K / 512 | 1,578.57 / 121.53 | 1,598.99 / 118.81 | -2.2% |
| 8K / 512 | 1,804.36 / 118.45 | 1,813.83 / 113.95 | -3.8% |
| 32K / 512 | 1,319.67 / 109.54 | 1,329.34 / 105.39 | -3.8% |

The two-client pass delivered 169.78 aggregate completion tok/s for vanilla
and 176.74 for uncensored, a 4.1% uncensored advantage. Average per-request
decode was 116.12 versus 121.60 tok/s. Across the single-request workloads,
uncensored MTP acceptance was 60.4-64.8%, modestly below vanilla's 62.4-66.5%.
The practical conclusion is that the two selected models have comparable
serving speed; uncensored gives up a few percent on long single requests and
recovers it under this small concurrency sample.

Raw local records are under `bench-results/20260821-qwen38-comparison/`.

#### Provenance index

| Run directory | Model config SHA-256 |
|---|---|
| `01-qwen-3.8-27b-mtp3-current` | `1683cc2a11b05f5e341d1775a21a8a3a9f6fe2866c5911599f21487810df7ba3` |
| `02-qwen-3.8-27b-uncensored-mtp3` | `b0096e94d3c15dcfd2318c7e4026092c37bb35c5027d48b0a82e0f4199bb1f54` |
