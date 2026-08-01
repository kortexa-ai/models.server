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
