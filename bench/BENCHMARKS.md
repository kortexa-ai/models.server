# Benchmark Results Log

Each entry tags its machine. Early entries are **sparky** (DGX Spark GB10 Blackwell, 128 GB unified). Later entries are **smarty** (RTX PRO 6000 Blackwell, 96 GB VRAM). MLX entries are **snappy** (Mac Mini M4 Pro, 64 GB unified).
TPS = generation tokens per second (warm, single-request, no-reasoning unless noted).

> **Note to future Claude:** not every bench run lands here — some live only in chat transcripts. If the user asks about speeds and the answer isn't in this file, extensively grep `~/.claude/projects/` on **both smarty and snappy** (over ssh) for unlogged TPS numbers before answering. Then offer to roll any findings into this file.

---

## August 4, 2026 — smarty (Core i9-14900K, CPU only)

### LFM2.5 VL 450M image serving (llama.cpp Q8_0)

Quick CPU-only bring-up of `LiquidAI/LFM2.5-VL-450M-GGUF` with its matching
Q8_0 vision projector. llama.cpp was build 10200 (`5f55650a7`), launched with
`--device none`, q8_0 K/V, 32,768 context, one slot, and `temperature=0`.
The process used about 0.9 GiB RSS and allocated no GPU memory. `smarty` has an
i9-14900K with 8 P-cores, 16 E-cores, and 32 logical CPUs.

The same Statue of Liberty photo was tested at 512x341 and as a 1536x1536
large-image/10-tile stress case. Cold rows processed the image from scratch;
warm rows repeated the identical image and prompt. Image results are one quick
run per condition, while text decode is the mean of two 128-token runs.

| Workload | Prompt / cached tok | Default 4-thread wall | Tuned 8/24 wall | Default prompt tok/s | Tuned prompt tok/s | Default decode tok/s | Tuned decode tok/s |
|----------|---------------------|-----------------------|-----------------|----------------------|--------------------|----------------------|--------------------|
| 512x341 image, cold | 201 / 1 | 0.536s | **0.434s** | 651 | **903** | 89.0 | **95.9** |
| 512x341 image, warm | 4 / 198 | 0.243s | **0.237s** | — | — | 90.2 | **92.9** |
| 1536x1536 image, cold | 2,594 / 1 | 5.007s | **2.742s** | 565 | **1,101** | 73.6 | **80.7** |
| 1536x1536 image, warm | 4 / 2,591 | 0.424s | **0.399s** | — | — | 77.2 | **81.5** |

Text-only decode improved from **87.1 tok/s** with the configured four threads
to **94.5 tok/s** with eight decode threads and 24 batch/image threads. The
tuned command was:

```bash
./run.sh lfm2.5-vl-450m --engine cpu --threads 8 --threads-batch 24
```

**Findings:**

- The tuned hybrid-CPU split cut cold 512px image latency by 19% and cold
  large-image latency by 45%. Large-image prompt throughput nearly doubled
  from 565 to 1,101 tok/s.
- Vision/prompt caching is the bigger win for repeated images: 198 of the small
  prompt's tokens and 2,591 of the large prompt's tokens were reused, reducing
  wall time to 0.24s and 0.40s, respectively.
- Both image sizes correctly identified the Statue of Liberty; this was a
  correctness smoke test, not a quality evaluation.
- Keep the shared four-thread CPU default for the Raspberry Pis. Use the 8/24
  override on `smarty`, where the extra batch threads materially accelerate
  image prefill.
- The temporary server was stopped, port 2052 was left free, and all five
  checked resident service health endpoints still returned HTTP 200.

---

## August 1, 2026 — smarty (RTX PRO 6000 Blackwell)

### Qwen 3.6 27B: current llama.cpp vs vLLM 0.26.0 MTP

Fresh comparison after updating both engines. llama.cpp was build 10200
(`5f55650a7`) with `Qwen3.6-27B-UD-Q4_K_XL.gguf`, q8_0 K/V, 262,144 context,
and GPU-side draft sampling. vLLM was stable 0.26.0 with the official FP8
checkpoint, FP8 KV, FlashInfer 0.6.14, and PyTorch 2.11/CUDA 13. Each single
result is decode throughput for a forced 600-token, temperature-0 `/no_think`
generation. Prose and Python prompts were identical across engines.

| Engine | MTP depth | Prose tok/s | Code tok/s | Mixed mean |
|--------|-----------|-------------|------------|------------|
| llama.cpp | off | 73.43 | 73.44 | 73.44 |
| llama.cpp | 1 | 107.00 | 113.31 | 110.16 |
| llama.cpp | 2 | 119.50 | 144.36 | 131.93 |
| **llama.cpp** | **3** | **115.19** (114.99–115.39, n=3) | **156.26** (152.06–158.39, n=3) | **135.73** |
| llama.cpp | 4 | 106.77 | 157.91 | 132.34 |
| vLLM | off | 50.69 | 50.69 | 50.69 |
| vLLM | 1 | 72.22 | 79.66 | 75.94 |
| vLLM | 2 | 78.13 | 99.96 | 89.05 |
| vLLM | 3 | 80.26 | 119.47 | 99.87 |
| **vLLM** | **4** | **85.91** (85.90–85.92, n=3) | **129.48** (129.37–129.57, n=3) | **107.70** |
| vLLM | 5 | 83.10 | 126.68 | 104.89 |

| Final configuration | llama.cpp | vLLM |
|---------------------|-----------|------|
| Weight / KV quant | UD-Q4_K_XL / q8_0 | official FP8 / fp8 |
| MTP | depth 3 | native `mtp`, depth 4 |
| Single prose / code | 115.19 / 156.26 tok/s | 85.91 / 129.48 tok/s |
| Four-request aggregate | not tested (production has one slot) | **310.44 tok/s**, 104.99 mean per request |
| Added GPU memory over resident baseline | ~28,409 MiB | ~42,411 MiB |
| Free VRAM while loaded | ~32,563 MiB | ~18,537 MiB |

**vLLM cache configuration:**

- Exact `kv_cache_memory_bytes=10,194,124,800` (9.494019 GiB), 183 allocator
  pages. vLLM reports exactly 262,144 aggregate cache tokens and 1.00x
  concurrency at the model's full context.
- `max_num_seqs=4` is a scheduling ceiling, not four preallocated 262K slots.
  One request can consume all 262K tokens; four equally long live requests can
  consume about 65K each. The shared pool therefore avoids reserving four full
  contexts for a mostly single-request service.
- Total GPU use was 78,713 MiB with all resident services still running, so no
  service needed to be stopped. The exact byte pool controls allocation;
  `gpu_memory_utilization=0.60` is only the startup headroom gate.

**Findings:**

- llama.cpp remains the production default: it is 1.34x faster on prose and
  1.21x faster on code while using about 14 GiB less VRAM. vLLM is the useful
  alternate when four-way aggregate throughput matters.
- llama.cpp depth 3 accepted 46.7% of drafted prose tokens and 78.0% of code
  tokens. vLLM depth 4 accepted 39.6% and 72.2%, respectively. Predictable code
  benefits more from deeper speculation.
- vLLM automatic attention selected FlashInfer and passed the four-request
  test. Forcing Triton improved some one-off single-request results, but the
  final four-request test crashed its isolated engine with an illegal CUDA
  memory access; do not use it here. All resident services remained healthy.
- FlashInfer currently falls back to PIECEWISE CUDA graphs with speculative
  decoding, which likely contributes to its single-request deficit.
- The FP8 checkpoint does not publish calibrated q/k/v/prob attention scales;
  vLLM warns that it is using 1.0. Throughput and output validity were checked,
  but quality should be evaluated before making FP8 vLLM the default.

---

## June 26, 2026 — snappy (Mac Mini M4 Pro, 64 GB unified)

### LFM2.5 230M (mlx-lm 0.31.3, 8-bit MLX, prompt/decode concurrency 4)

New tiny edge model on Snappy using `LiquidAI/LFM2.5-230M-MLX-8bit`.
Launcher flags: `python -m mlx_lm server --prompt-concurrency 4 --decode-concurrency 4`.
Numbers are OpenAI-compatible `/v1/chat/completions`, `temperature=0`, wall-clock from client timing and API `usage`.
Completion TPS is completion tokens divided by full request wall time, so cold long-prompt rows include prefill cost.

| Workload | Prompt tok | Completion tok | Cold wall | Cold comp TPS | Warm cached tok | Warm wall | Warm comp TPS |
|----------|------------|----------------|-----------|---------------|-----------------|-----------|---------------|
| Short answer | 46 | 39 | 0.107s | 364.1 | 45 | 0.084s avg | 466.1 avg |
| Long decode | 60 | 256 | 0.609s | 420.7 | 59 | 0.510s avg | 501.6 avg |
| Chat history | 211 | 46 | 0.136s | 338.4 | 210 | 0.097s avg | 476.2 avg |
| Long prompt | 6,361 | 90 | 0.743s | 121.1 | 6,360 | 0.218s avg | 412.8 avg |
| Larger prompt | 15,840 | 96 | 1.720s | 55.8 | 15,839 | 0.293s | 327.4 |
| Near-128K context probe | 120,540 | 16 | 42.632s | 0.4 | 120,539 | 0.341s | 46.9 |

Four concurrent long-ish requests (`prompt-concurrency=4`, `decode-concurrency=4`): 4 requests × ~1,647 prompt tok / 47 completion tok completed in **0.645s batch wall**, aggregate **291.5 completion tok/s** and **10.5K total tok/s**.

**Notes:**
- Prompt cache is very effective on repeated stable prompts: 15.8K prompt cold-to-warm dropped from 1.72s to 0.293s; 120.5K prompt dropped from 42.6s to 0.341s.
- Cold near-128K prefill is usable but slow (~2.8K total tok/s wall-clock including the 16-token answer). Warm repeated-prefix calls are effectively dominated by decode/response overhead.
- The model sometimes ignores terse instructions ("Say ok" became a help-offer), so benchmark prompts that require exact format should be treated as latency probes, not quality probes. Tiny model, tiny opinions, still opinions.

---

## June 3, 2026 — snappy (Mac Mini M4 Pro, 64 GB unified)

### Gemma 4 12B bring-up (mlx-vlm 0.6.1, 4-bit, MTP drafter)

New model. Dense 11.95B, multimodal, 256K ctx. Server-reported `predicted_per_second`, warm, 400-token, `temperature=0`. Drafter `gemma-4-12B-it-assistant-bf16` (~0.85 GB MTP head) confirmed loaded.

| Model | Engine | Quant | Gen TPS |
|-------|--------|-------|---------|
| Gemma 4 12B | mlx-vlm (MTP, block=4) | 4-bit | **28.9** (26.4–33.1, n=5) |
| Gemma 4 12B | mlx-vlm (no drafter) | 4-bit | 21.1 (20.9–21.2, n=5) |

**Notes:**
- Peak memory ~12.2 GB (11 GB model + 0.85 GB drafter + KV).
- **MTP is a net WIN here: 28.9 vs 21.1 = 1.37x.** First MLX model in this log where the drafter pays off. Baseline is rock-flat (20.9–21.2); MTP jitters 26–33 (acceptance-rate variance) but stays well above baseline every run.
- Confirms the large/slow-target rule across engines: spec decoding lost on E2B/E4B (already 70–124 tok/s), wins here where the 12B target is slow enough (~21 tok/s) for the cheap 0.85 GB drafter to come out ahead.
- Decision: keep `mlx.draft_enabled=true` for the 12B.

---

## June 3, 2026 — smarty (RTX PRO 6000 Blackwell)

### Gemma 4 12B (llama.cpp UD-Q4_K_XL, q8_0 KV)

Same new model, llama.cpp side. Server-reported `predicted_per_second`, warm, 400-token, `temperature=0`, single slot.

| Model | Engine | Quant | Gen TPS |
|-------|--------|-------|---------|
| Gemma 4 12B | llama.cpp | UD-Q4_K_XL | **123.8** (123.6–123.9, n=5) |

**Notes:**
- Rock-flat 123.8 — slots between Gemma 4 31B dense (59) and 26B-A4B MoE (164.5); ~2x the 31B, tracking the ~2.6x param ratio.
- ~4.3x faster than snappy's MTP-on 12B (28.9) — RTX 6000 vs M4 Pro.
- No MTP on this side: gemma-4-12b has no `*-MTP-GGUF` (baked draft heads), so llama.cpp runs vanilla. (`common_speculative_init: no implementations specified` in the log is expected.)
- Required a llama.cpp update for Gemma 4 vision: the `gemma4uv` mmproj projector is only in recent builds (smarty's 9467 predated it). Post-update, mmproj loads clean and multimodal works.
- llama.cpp auto-caps the slot to the model's trained context (131072), confirming the GGUF's `n_ctx_train`; `model.json` context (262144, from the HF card) should be lowered to match.

---

## June 1, 2026 — snappy (Mac Mini M4 Pro, 64 GB unified)

### Gemma 4 MLX speculative-decoding bring-up (mlx-vlm 0.6.0, 4-bit, MTP drafter)

First **snappy / MLX** entry in this log. Required patching mlx-vlm to fix a
Gemma 4 MTP rollback crash ([mlx-vlm#1260](https://github.com/Blaizzy/mlx-vlm/issues/1260),
fix PR [#1261](https://github.com/Blaizzy/mlx-vlm/pull/1261)) — see `../README.md`.
Numbers below are server-reported `timings.predicted_per_second`, warm,
single-request, 400-token generations, `temperature=0`.

| Model | Engine | Quant | Drafter | Gen TPS |
|-------|--------|-------|---------|---------|
| Gemma 4 E2B | mlx-vlm (MTP, block=4) | 4-bit | gemma-4-E2B-it-assistant-bf16 | 112.5 (108.7–120.3, n=5) |
| Gemma 4 E2B | mlx-vlm (no drafter) | 4-bit | — | **123.8** (122.8–124.1, n=5) |
| Gemma 4 E4B | mlx-vlm (MTP, block=4) | 4-bit | gemma-4-E4B-it-assistant-bf16 | 66.8 (63.1–72.5, n=5) |
| Gemma 4 E4B | mlx-vlm (no drafter) | 4-bit | — | **70.6** (70.5–70.6, n=5) |

**Notes:**
- MTP confirmed active on the drafter runs (server log: `Drafter ready — speculative decoding enabled.`). Peak memory ~3.8 GB (E2B) / ~5.5 GB (E4B).
- E2B ran on port 2039, E4B on 2038.
- **MTP is a net loss on both small models:** E2B 112.5 vs 123.8 no-drafter (~9% slower), E4B 66.8 vs 70.6 (~5% slower). No-drafter baselines are rock-flat (E2B 122.8–124.1, E4B 70.5–70.6) while MTP runs swing — the signature of low draft acceptance, where verification overhead exceeds the savings. Speculative decoding pays off on large *slow* targets, not 4B-class models where the target is already cheap.
- Decision: `mlx.draft_enabled=false` for E2B/E4B; 26B-A4B/31B left `true` pending an MLX bench (the fix from mlx-vlm#1260 is still required for those to even run MTP).

---

## June 1, 2026 — smarty (RTX PRO 6000 Blackwell)

### Qwen 3.6 35B-A3B MTP speculative decoding (llama.cpp UD-Q4_K_XL, n_max=3, parallel=8)

First bench of the Unsloth MTP build (`Qwen3.6-35B-A3B-MTP-GGUF`) against the non-MTP baseline of **187.1 TPS** from April.

| Workload | Gen TPS | Speedup vs 187.1 | Draft accept rate |
|----------|---------|------------------|-------------------|
| Prose run 1 (no_think, 600 tok) | 265.3 | 1.42x | 52.1% (365/700) |
| Prose run 2 | 264.7 | 1.41x | 51.5% (363/705) |
| Prose run 3 | **275.3** | 1.47x | 55.5% (374/674) |
| Code run 1 (no_think, 600 tok) | 287.8 | 1.54x | 59.7% (384/643) |
| Code run 2 | 268.3 | 1.43x | 53.3% (368/691) |
| Code run 3 | **303.1** | **1.62x** | **65.7%** (396/603) |

**Notes:**
- Net win: **~1.4-1.6x** over the non-MTP baseline. Below the upstream PR's quoted 1.7-2x, but the PR's 72-83% accept rate was on different workloads; we hit 52-66%.
- Code beats prose on both TPS (~285 vs ~268 avg) and accept rate (~60% vs ~52%) — speculation likes predictable token sequences.
- First attempt at this bench (earlier in the same session) hit only ~240 TPS / 43% accept and **segfaulted on the 3rd request**. Two things changed before the successful retry: fresh llama.cpp rebuild AND GPU ECC enabled (`sudo nvidia-smi -e 1`). The crashed run also saw two single-byte file corruptions in the llama.cpp working tree (one bit-flip in `ggml-opencl.cpp`, one in `ggml-metal.metal`), which is the signature of VRAM/DMA corruption with ECC off. Can't prove ECC was the smoking gun without an A/B, but the circumstantial case is strong.
- Blackwell PRO 6000 reports ECC-on with no VRAM tax visible (`memory.total` unchanged at 97887 MiB) — appears to use inline ECC.
- MTP config: `mtp_n_max: 3, mtp_n_min: 0`, prompt cache 8192 MiB, context checkpoints max=32, parallel=8 (8 slots × 65536 ctx). All bench requests served by single slot via LCP cache hits.

### Qwen 3.6 27B dense MTP (llama.cpp UD-Q4_K_XL)

Same session. Non-MTP baseline from April: **68.0 TPS**.

| Workload | Gen TPS | Speedup vs 68.0 | Draft accept rate |
|----------|---------|-----------------|-------------------|
| Prose run 1 | 126.1 | 1.85x | 60.3% (385/639) |
| Prose run 2 | 109.3 | 1.61x | 47.0% (350/744) |
| Prose run 3 | 116.4 | 1.71x | 52.6% (366/696) |
| Code run 1 | 126.9 | 1.87x | 60.6% (386/637) |
| Code run 2 | **130.6** | **1.92x** | 63.4% (392/618) |
| Code run 3 | 130.2 | 1.91x | 63.1% (392/621) |

**Notes:**
- **Dense gets bigger MTP gains than MoE** (1.9x vs 1.6x). Per-token forward-pass cost is higher on dense (full 27B vs 3-4B active on MoE), so each accepted draft saves more wall time. MoE already amortizes the work MTP would otherwise skip.
- Accept rates on code (~62%) close to MoE's (~60%) — speculation quality is more about workload than model architecture.
- No crashes, 6/6 runs clean.

---

## April 22–24, 2026 — smarty (RTX PRO 6000 Blackwell)

### Qwen 3.6 bring-up sweep + Gemma 4 retest (llama.cpp UD-Q4_K_XL, q8_0 KV, n=4)

| Model | Engine | Quant | Gen TPS |
|-------|--------|-------|---------|
| Qwen 3.6 35B-A3B PRISM-NVFP4 | vLLM Marlin | NVFP4 | **195.2** |
| Qwen 3.6 35B-A3B | llama.cpp | UD-Q4_K_XL | **187.1** |
| Qwen 3.5 35B-A3B | llama.cpp | UD-Q4_K_XL | 181.2 |
| Qwen 3.6 35B-A3B PRISM-NVFP4 | vLLM FlashInfer CUTLASS | NVFP4 | 174.6 |
| Qwen 3.6 35B-A3B PRISM-NVFP4 | vLLM 0.19.2 + flashinfer 0.6.8 + torch 2.11 | NVFP4 | 174.4 |
| Gemma 4 26B-A4B | llama.cpp | UD-Q4_K_XL | 164.5 |
| Qwen 3.6 27B (dense) | llama.cpp | UD-Q4_K_XL | 68.0 |
| Gemma 4 31B (dense) | llama.cpp | UD-Q4_K_XL | 59.0 |

**Notes:**
- MoE wins on raw throughput at total-param scale — only ~3-4B active per token vs 27-31B for dense.
- Gemma 4 31B at UD-Q4_K_XL was retested later and hit ~64 tok/s (vs 59 in the sweep, vs 42 at Q8_0 on Apr 5).
- PRISM-NVFP4 vLLM Marlin was the peak; FlashInfer CUTLASS and the bumped vLLM 0.19.2 / flashinfer 0.6.8 / torch 2.11 stack landed lower (~175).

---

## April 5, 2026 — smarty (RTX PRO 6000 Blackwell)

### Gemma 4 family bring-up (llama.cpp Q8_0)

| Model | Type | Active params | Gen TPS | Prompt TPS |
|-------|------|---------------|---------|------------|
| Gemma 4 E2B | dense (PLE) | ~2B | **250** | 2,793 |
| Gemma 4 26B-A4B | MoE | 4B | **171** | 1,387 |
| Gemma 4 31B | dense | 31B | **42** | 646 |

**Notes:**
- E2B on Q8_0 is an absolute screamer — model fits in L2-ish, no routing overhead.
- 26B-A4B / 31B ratio (171 vs 42 ≈ 4x) lines up with active-params (4B vs 31B).
- vLLM with batching hit ~324 tok/s aggregate on 26B-A4B (chat-side note, not formal sweep).
- Gemma 4 E4B never got a clean bench in the transcripts.
- Gemma 4 uses 512-token sliding window attention — llama.cpp can't reuse KV across requests, so prompt processing eats a re-process penalty. Generation TPS is unaffected.

---

## March 21, 2026 — sparky (DGX Spark)

### vLLM Docker (`vllm-node:latest`) — Qwen 3.5-35B-A3B (Intel AutoRound int4)

**BREAKTHROUGH:** After extensive investigation, achieved **50 tok/s** (target was 50-79 tok/s).

| Metric | Value |
|--------|-------|
| Warm gen TPS (avg) | **50.2** |
| TPS range | 46.2 - 51.6 |
| Model size | 19.3 GiB |
| KV cache | 60+ GiB available |
| Context | 32768 |
| Quantization | Intel AutoRound int4 (GPTQ-compatible) |

**Configuration:**
```bash
docker run --rm --name vllm-test-35b \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm-node:latest \
  Intel/Qwen3.5-35B-A3B-int4-AutoRound \
  --port 2242 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.7 \
  --load-format fastsafetensors \
  --kv-cache-dtype fp8
```

**Key findings:**
- vLLM 0.17.2rc1.dev7 has proper SM121 (Blackwell) support
- FlashInfer attention backend works correctly on DGX Spark
- CUDA graphs captured successfully (51 PIECEWISE + 35 FULL)
- FP8 KV cache critical for memory bandwidth on unified memory

**Comparison with previous attempts:**

| Configuration | TPS | Notes |
|--------------|-----|-------|
| vLLM 0.17.1rc1 bare-metal BF16 | 17.4 | Baseline, slow |
| vLLM 0.17.2rc1 Docker + AutoRound int4 | **50.2** | 3x improvement! |
| Target (community reports) | 50-79 | Achieved! |

**Remaining optimization opportunities:**
- Apply MXFP4 patches → could reach 60-70 tok/s
- Use TP=2 (if 2 GPUs available) → could double throughput
- Kill competing processes → could increase to 0.85 memory utilization

---

### vLLM Docker (`vllm-node:latest`) — Qwen 3.5 Full Sweep

**Date:** March 22, 2026

Complete benchmark of all Qwen 3.5 sizes in both BF16 and int4:

| Model | BF16 TPS | int4 TPS | Speedup | BF16 Mem | int4 Mem |
|-------|----------|----------|---------|----------|----------|
| Qwen3.5-4B | 20.6 | **43.9** | 2.1x | 8.6 GiB | 3.7 GiB |
| Qwen3.5-9B | 12.3 | **32.7** | 2.7x | 17.7 GiB | 8.1 GiB |
| Qwen3.5-27B | 2.5 | **12.6** | 5.0x | ~55 GiB | 17.6 GiB |
| Qwen3.5-35B-A3B | 17.4 | **50.2** | 2.9x | ~67 GiB | 19.3 GiB |

**Key findings:**
1. **int4 quantization helps all models** - 2-5x speedup depending on size
2. **27B shows biggest gain (5x)** - most memory-bound model benefits most
3. **35B MoE int4 is fastest** - despite being "largest", MoE only activates 3B params
4. **BF16 for large models is unusable** - 27B crawls at 2.5 tok/s

**Recommendation:** Always use int4 quantization on DGX Spark. The 35B-A3B int4 at 50 tok/s is the sweet spot.

**Updated run.sh scripts (March 22):**
All Qwen 3.5 `run.sh` scripts now auto-detect DGX Spark and use vLLM Docker with int4:
- `GPU_MEM_FRAC` env var controls memory allocation (default varies by model size)
- On Spark: uses `vllm/vllm-openai:vllm-node` with AutoRound int4 models
- On other Linux: uses llama.cpp with GGUF
- On macOS: uses mlx-vlm

**Comparison with llama.cpp (4B):**

| Engine | Quant | TPS |
|--------|-------|-----|
| vLLM Docker | AutoRound int4 | **43.9** |
| llama.cpp | Q4_K_M GGUF | 38.2 |

vLLM Docker int4 beats llama.cpp GGUF for the 4B model.

---

### Nemotron 3 Nano 30B NVFP4 — Not Working

**Status:** CUDA graph capture hangs / CUTLASS TMA errors

```
[ERROR] Assertion failed: Failed to initialize cutlass TMA WS grouped gemm
```

The `vllm-node:latest` image has CUTLASS SM120 kernel issues with Nemotron models.
Needs investigation or different Docker image.

---

## March 12, 2026

### Nemotron-3-Super 120B-A12B on DGX Spark

Architecture: NemotronHForCausalLM (Mamba-2 hybrid + LatentMoE, 120B total / 12B active).

#### llama.cpp — Q4_K_M (GGUF, unsloth, 16k context, Q4_0 KV cache)

| Metric | Value |
|--------|-------|
| Cold gen TPS | 10.9 |
| Warm gen TPS | 11.7–12.6 |
| Prompt TPS | 41–46 |
| Model size | ~82.5 GB |

#### vLLM 0.17.1rc1 Bare-Metal — NVFP4 (Marlin backend, FP8 KV cache, 32k context)

| Metric | Value |
|--------|-------|
| Warm gen TPS | 12.3–12.7 |
| GPU memory | 86.8 GB |

Notes:
- NVFP4 is trained-in quantization (not post-training), weights ~69.5 GB
- Requires Marlin env vars: `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_TEST_FORCE_FP8_MARLIN=1`, `VLLM_MARLIN_USE_ATOMIC_ADD=1`
- Uses `super_v3` reasoning parser via plugin file (not built into vLLM 0.17.1)
- Model card mandates `temperature=1.0, top_p=0.95` for all tasks
- TRT-LLM OOM'd even at batch_size=1 / 8k context — 69.5 GB model + KV cache exceeds 128 GB budget

---

## March 11, 2026

### vLLM 0.17.1rc1 Bare-Metal — Qwen 3.5 Full Sweep (BF16)

| Model | TTFT (warm) | Gen TPS | Context | Notes |
|-------|-------------|---------|---------|-------|
| Qwen3.5-0.8B | 206ms | 98.1 | 32768 | |
| Qwen3.5-4B | ~15s | 22.8 | 32768 | High TTFT under investigation |
| Qwen3.5-9B | 4.5s | 13.5 | 32768 | |
| Qwen3.5-35B-A3B | 4.3s | 17.4 | 32768 | MoE, only 3B active — faster than 9B dense |
| Qwen3.5-27B | 4.1s | 3.8 | 32768 | --enforce-eager |

### TRT-LLM PyTorch Backend — Qwen 3 (BF16, Docker Sourcebuild)

Image: `local/trtllm-main:main-transformers--5.3.0-sourcebuild-120-real`

Note: These are Qwen **3** models (not 3.5 — TRT-LLM doesn't support Qwen 3.5 yet).

| Model | TTFT (warm) | Gen TPS | TPS (overall) |
|-------|-------------|---------|---------------|
| Qwen3-0.6B | 180ms | 56.2 | 55.2 |
| Qwen3-4B | 730ms | 22.6 | 21.5 |
| Qwen3-8B | 1304ms | 14.0 | 13.6 |

---

## March 9, 2026

### vLLM Docker (`vllm/vllm-openai:cu130-nightly`) — Qwen 3.5 (BF16)

| Model | Context | Cold TPS | Warm TPS | Notes |
|-------|---------|----------|----------|-------|
| Qwen3.5-0.8B | 32768 | 2.69 | 103.77 | gpu-memory-utilization=0.25 |
| Qwen3.5-9B | 262144 | — | 12.03 | Avg of 3 runs (13.01, 13.01, 10.08) |

### vLLM Bare-Metal — Qwen 3.5 (BF16)

| Model | Context | Cold TPS | Warm TPS |
|-------|---------|----------|----------|
| Qwen3.5-0.8B | 32768 | 4.01 | 103.84 |
| Qwen3.5-9B | 262144 | 3.40 | 12.14 |

### SGLang Docker — Qwen 3.5-0.8B (BF16)

| Image | Cold TPS | Warm TPS |
|-------|----------|----------|
| scitrera/dgx-spark-sglang:0.5.9-t5 | — | 86.97 |
| scitrera/dgx-spark-sglang:0.5.9-dev1-329817e2-t5 | — | 93.11 |
| Same dev image, Q4_K_M comparison | 27.18 | 85.03 |

### SGLang Docker — Qwen 3.5-9B (BF16, 262144 ctx)

| Engine | Cold TPS | Warm TPS |
|--------|----------|----------|
| SGLang (scitrera dev) | 10.32 | 12.69 |

### llama.cpp — Qwen 3.5-0.8B

| Quant | Cold TPS | Warm TPS |
|-------|----------|----------|
| Q8_0 | 70.27 | 70.60 |
| Q4_K_M | 72.17 | 69.38 |

Server-side decode: ~72 tok/s (Q8_0), ~71 tok/s (Q4_K_M).

### llama.cpp — Qwen 3.5-9B (Q4_K_M, 262144 ctx)

| Metric | Value |
|--------|-------|
| Cold TPS | 24.39 |
| Warm TPS | 24.83 |

Server-side decode: ~25 tok/s.
Startup memory: model 4861 MiB, context 2354 MiB, compute 808 MiB → ~8023 MiB.

### 4B Quantization Sweep (vLLM Bare-Metal + llama.cpp)

| Engine / Checkpoint | Quant | Warm TPS | Valid Output |
|---------------------|-------|----------|--------------|
| llama.cpp Q4_K_M | GGUF 4-bit | 38.21 | Yes |
| vLLM AxionML NVFP4 | modelopt_fp4 | 38.45 | No (corrupted) |
| llama.cpp Q8_0 | GGUF 8-bit | 35.96 | Yes |
| vLLM olka-fi MXFP4 | compressed-tensors | 33.82 | Yes |
| vLLM QuantTrio AWQ | awq_marlin | 33.15 | Yes |

### 0.8B Engine Comparison Summary

| Engine | Warm TPS |
|--------|----------|
| vLLM (Docker or bare-metal) | ~104 |
| SGLang (Docker dev) | ~93 |
| SGLang (Docker stable) | ~87 |
| llama.cpp Q8_0 | ~71 |
| llama.cpp Q4_K_M | ~69 |

### 9B Engine Comparison Summary

| Engine | Warm TPS | Context |
|--------|----------|---------|
| llama.cpp Q4_K_M | 24.83 | 262144 |
| SGLang Docker | 12.69 | 262144 |
| vLLM bare-metal | 12.14 | 262144 |
| vLLM Docker | 12.03 | 262144 |

Note: llama.cpp wins at 9B because the GGUF Q4_K_M quantization is much more memory-efficient than BF16, allowing more room for compute. The BF16 engines (vLLM, SGLang) are constrained by the larger model footprint at this size.

---

## March 8, 2026

### SGLang Docker — Qwen 3.5-0.8B First Bring-Up

- `scitrera/dgx-spark-sglang:0.5.9-t5` successfully booted `Qwen/Qwen3.5-0.8B`
- Confirmed: weight load, KV cache allocation, CUDA graph capture, API up on port 2131
- Smoke test passed: `Reply with exactly: ok` → `ok`

### vLLM Docker — Qwen 3.5-0.8B First Bring-Up

- `vllm/vllm-openai:cu130-nightly` booted `Qwen/Qwen3.5-0.8B`
- Required `--gpu-memory-utilization 0.25` (default 0.9 fails)
- Resolved as `Qwen3_5ForConditionalGeneration`
- API up on port 2211, chat completions working
