# Cinference on the RTX PRO 6000

This bounded experiment compares non-Swift Huihui Qwen3.8-27B NVFP4 with the
stock Qwen3.8 configuration. It does not install a production service.

The launcher is `satellitedown/fast-long-context-cinference` at
`104b236e557741b990ff084ef6512801689c9613`; its manifest pins the native runtime
and the model file checksum. Clone it under `.engines/cinference` on Smarty.
Run its `scripts/install.sh` and `scripts/download_models.py` before stopping
production. The build needs FFmpeg development libraries and libcurl.

After explicit authorization for downtime and an exclusive GPU work claim:

```bash
mkdir -p bench-results/cinference-27
bash bench/cinference-6000/run-block.sh > bench-results/cinference-27/block.log 2>&1
```

The existing LegoLM GPU-block script owns restoration in its EXIT trap. The
wrapper maps its historical Qwen entry to Bonsai only when the live snapshot
shows Bonsai as production. It stops safe-list services until 80 GiB is free.
The benchmark pins the 6000 UUID, keeps the existing 450 W cap, and terminates
its own server if free VRAM falls below 10 GiB. No 4090 process is changed.

Profiles:

- `published`: 262144 shared KV tokens, one lane, K8V4, MTP-10, text only.
- `stock-like`: 524288 shared KV tokens, eight lanes, INT8, MTP-3, vision
  enabled. Cinference CUDA graphs and prefix retention remain enabled.
- `stock`: the current Qwen3.8 `run.sh` configuration, on a private port.
  The model snapshot and live process capture determine the actual settings.

All profiles use the same deterministic archive prompts. Each context has a
six-marker recall request and a prose request capped at 512 output tokens.
Prompt cache reuse is recorded; sizes are measured with the model tokenizer.
The main table uses greedy, non-thinking requests. A separate medium-thinking
request records reasoning-mode throughput. The larger profiles additionally
probe a 500K prompt and eight simultaneous 8K requests.

These are bounded performance probes, not broad model-quality evaluations.
Recall speed can be inflated by high speculative acceptance. The 500K probe
exceeds the model's native 262K context and does not qualify extended-context
quality. Compare actual API usage, server timings, output lengths, and marker
recovery alongside speed. Native timing definitions can differ by engine.

Raw output, server logs, and half-second GPU telemetry stay in ignored
`bench-results/cinference-27`. Process VRAM is separate from total-card VRAM.
Each profile directory is new and cannot overwrite a previous run.
