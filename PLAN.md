# LFM2.5 1.2B Thinking and 2.6B bring-up

## Goal

Add both text models with official Q8_0 GGUF checkpoints on `smarty` CUDA,
official 8-bit MLX checkpoints on `snappy`, maximum supported context, and one
request slot. Verify and benchmark each backend, then leave both services
uninstalled, stopped, and without downloaded model artifacts on either host.

## Work plan

1. Add model configs, service units, inventory docs, and minimal MLX subfolder
   support for the 2.6B checkpoint's multi-precision repository.
2. Validate parsers, launchers, launchd plists, and systemd units.
3. Temporarily install, smoke-test, and benchmark both MLX services on `snappy`.
4. Commit and sync through Git, then repeat on `smarty` with full CUDA offload
   while preserving resident production services and at least 10 GiB VRAM.
5. Record results; uninstall both services, remove only artifacts downloaded by
   this run, and verify ports, processes, services, caches, and resident health.

## Status

- Completed 2026-08-04. Both models returned the correct arithmetic canary on
  MLX and CUDA; llama.cpp reported one 32,768-token slot for 1.2B Thinking and
  one 128,000-token slot for 2.6B.
- Warm 256-token decode measured 147.9 / 68.5 wall tok/s on `snappy` MLX and
  749.6 / 330.8 server tok/s on `smarty` CUDA for 1.2B / 2.6B, respectively.
  Matched 4.1K-token cold and cached-prompt results are in
  `bench/BENCHMARKS.md`.
- The launchd plists use model-named wrapper scripts, so macOS Login Items do
  not show two more generic `run.sh` entries.
- No production service was stopped. Both new services are uninstalled and
  off on both hosts; ports 2026 and 2027 are free, all four downloaded model
  caches were removed, all resident `smarty` endpoints are HTTP 200, and both
  repositories are clean at the same commit.

---

# LFM2.5 VL 450M GPU benchmark on smarty

## Goal

Measure the same text, 512px photo, and 1536px tiled-photo workloads on
`smarty` with full GPU offload, without stopping or disturbing resident
production services.

## Live baseline — 2026-08-04

- RTX PRO 6000: 64,798 MiB used, 32,453 MiB free, 0% utilization, 32 C.
- Running GPU set: image base, ComfyUI, Vision, Qwen 3.6 27B, Gemma 4 E2B,
  ASR, TTS, and Qwen3 Embedding 0.6B.
- Repository is clean on `main` at `830e373`; port 2052 is free and no LFM2.5
  VL benchmark process exists.
- No service will be stopped. Keep at least 10 GiB free and begin with one
  short image request before running the matched benchmark.

## Work plan

1. Verify every resident GPU endpoint and capture the exact GPU baseline.
2. Start one isolated llama.cpp GPU server and run a small canary.
3. Measure matched text, 512px, and 1536px cold/warm requests.
4. Stop the benchmark server, recheck every baseline service, and log results.

## Status

- Completed 2026-08-04 without production downtime. Full GPU offload delivered
  1,310.64 text tok/s, 0.084s / 0.026s cold/warm 512px images, and 0.159s /
  0.062s cold/warm 1536px tiled images.
- Peak sampled use left 30,955 MiB VRAM free. The benchmark server was stopped,
  port 2052 is free, all eight resident endpoints are HTTP 200, and final GPU
  use returned to within 90 MiB of baseline.
- Results are recorded in `bench/BENCHMARKS.md`.

---

# LFM2.5 VL 450M deployment and Pi benchmark

## Goal

Keep the model available on `snappy` as a low-latency camera/photo service,
then measure its CPU-only image performance on one 8 GB Raspberry Pi 5 with
a matched Q8_0 model and projector.

## Work plan

1. Inspect the existing service and both Pis; use the quieter healthy Pi.
2. Install, start, and exercise the managed MLX service on `snappy`.
3. Update llama.cpp on the selected Pi only if needed, copy the non-repository
   Q8_0 artifacts, and benchmark text plus small and tiled image inputs.
4. Record results, stop the temporary Pi server, and verify affected services.

## Status

- Completed 2026-08-04: the managed MLX service is installed, enabled at
  login, running, and photo-tested on `snappy` at port 2052 with a 32K
  effective context.
- `happyhippo` was idle and selected for the Pi test. llama.cpp was updated to
  build 10267, and the verified 459.7 MiB Q8_0 model/projector pair remains on
  the Pi for later use.
- The Pi delivered 26.71 text tok/s, a 4.07s cold / 0.60s warm 512px image,
  and a 60.48s cold / 1.01s warm 1536px tiled image. All image responses were
  correct in the smoke test.
- Results are logged in `bench/BENCHMARKS.md`. The temporary Pi server and test
  images were removed, port 2052 is free there, and the `snappy` service is
  healthy.

---

# Qwen 3.6 27B engine and memory tuning

## Goal

Configure Qwen 3.6 27B for production with a fixed vLLM KV-cache byte
budget, native 262,144-token maximum context, up to four concurrent
requests sharing the cache, native MTP speculative decoding, and deliberate
VRAM headroom alongside the other services on `smarty`. Measure the latest
stable vLLM against the current llama.cpp build before choosing the production
engine and settings.

## Live baseline — 2026-08-01 13:35 PDT

- Host: `smarty`, Linux, RTX PRO 6000 Blackwell, 97,887 MiB VRAM.
- GPU: 36,280 MiB used; 61,607 MiB free.
- Qwen 3.6 27B and Gemma 4 31B are not installed or running.
- Healthy GPU services to preserve: Gemma 4 E2B, Qwen3 Embedding 0.6B,
  ASR, TTS, Vision, and LegoLM.
- Vision uses 14,846 MiB and may be stopped temporarily if the user-authorized
  benchmark cannot retain safe headroom. If stopped, restore it and verify
  `http://127.0.0.1:4001/health` returns HTTP 200.
- Repository: clean `main` at `f6ede81`.
- llama.cpp: build 10200 (`5f55650a7`).
- vLLM before upgrade: `0.19.2rc1.dev131+gac58e2a17.cu130`.
- vLLM after upgrade: stable `0.26.0` with FlashInfer `0.6.14`.

## Work plan

1. Inspect current launch/config support and select a reproducible latest
   stable vLLM build.
2. Add minimal configuration plumbing for exact KV bytes and native Qwen MTP.
3. Cache required model files without allocating GPU memory.
4. Measure current llama.cpp with matched prose/code workloads, MTP depths,
   GPU memory, and acceptance rate.
5. Measure latest vLLM with the same workloads, first without and then with
   native MTP; determine the smallest exact KV budget that admits a 262,144
   token request while leaving comfortable production headroom.
6. Select the production engine/configuration, document results, validate
   parsers/launch commands, and restore every temporarily stopped service.

## Safety bounds

- Keep at least 10 GiB VRAM free during canaries; prefer at least 15 GiB in
  the final production state.
- Stop only `vision.server` if the initial canary cannot preserve the bound.
- Run one model/engine candidate at a time and begin with a short batch-1
  request.
- Use exact `kv_cache_bytes`; do not rely on GPU-utilization percentages.
- Do not leave a benchmark server or model process running after measurement.

## Status

- Baseline captured.
- Exact-KV and native-MTP launcher plumbing implemented.
- Dummy-weight allocator canary confirmed that `9,804,185,600` bytes
  (9.130859 GiB) produces exactly 262,144 aggregate KV tokens and 1.00x
  full-context concurrency with MTP depth 2. `max_num_seqs=4` shares this pool
  dynamically. The pool is 176 allocator pages: 164 full-attention pages for
  262,144 tokens plus 12 small recurrent-state reservations (three hybrid
  groups times four schedulable sequences). Four-way scheduling therefore
  costs only 478.125 MiB more than a one-sequence pool, not four full contexts.
- The 60% GPU-utilization value is only vLLM's startup admission guard; vLLM
  explicitly confirms that the exact KV byte value overrides it.
- Both model artifacts are cached locally. The final vLLM configuration uses
  MTP depth 4 and a 183-page, `10,194,124,800`-byte (9.494019 GiB) FP8 KV
  pool. vLLM reports exactly 262,144 aggregate tokens and 1.00x full-context
  concurrency. Four requests share the pool dynamically.
- Current llama.cpp build 10200 is fastest at MTP depth 3: 115.19 tok/s prose
  and 156.26 tok/s code (three 600-token runs). Its measured footprint is
  about 28,409 MiB above the resident-service baseline.
- vLLM 0.26.0 with automatic FlashInfer attention is fastest at MTP depth 4:
  85.91 tok/s prose and 129.48 tok/s code (three 600-token runs), plus 310.44
  aggregate tok/s for four simultaneous 600-token requests. It uses about
  42,411 MiB above baseline and leaves 18,537 MiB free with every resident
  service running.
- Forcing Triton attention was rejected: single requests worked, but the
  four-request test caused an illegal CUDA memory access. GPU state recovered
  when the isolated benchmark process exited and every resident health check
  remained HTTP 200.
- Production remains on llama.cpp for the common single-request workload;
  the exact-cache vLLM configuration is available through `--engine vllm` for
  bursty concurrent workloads.
- Completed 2026-08-01: configuration and launcher syntax pass, the vLLM
  environment has no dependency conflicts, no benchmark process remains, GPU
  use returned to the 36,302 MiB resident baseline, and all six preserved GPU
  services are running with HTTP 200 health checks.
