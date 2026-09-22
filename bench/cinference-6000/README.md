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
exercise eight simultaneous 8K requests. The stock baseline also probes a 500K
request; its actual individual-request ceiling is 262K despite the 512K shared
KV pool. Cinference comparisons use that same 262K request ceiling.

The vanilla campaign uses `neroued/Qwen3.8-27B-nvfp4-NInfer` at
`f0b43ad436b9fa8142c6ed6647c470a6fe409484`, with model SHA-256
`74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`.
Download it into `.engines/cinference/models/Qwen3.8-27B-nvfp4-NInfer`, verify
that digest, and write it to `verified.sha256` in the same directory. This is
the original post-trained model in mixed NVFP4/FP8 form, without abliteration.

After the control campaign has produced its prompt fixtures, run:

```bash
bash bench/cinference-6000/run-block.sh --vanilla-tune \
  > bench-results/cinference-27/vanilla-block.log 2>&1
```

The bounded tuner compares 1024, 4096 and 8192 prefill chunks at fixed INT8 KV,
eight slots and MTP-3. It selects the lowest summed 131K/260K prompt-processing
time, extending to 16K/32K only while gains exceed 5% with ample headroom. It then
compares K8V4 at the same MTP-3 settings to isolate cache precision, then MTP-10
and DFlash2-7 with K8V4 and production-sized capacity. The winner
has the lowest 260K cold prefill time, using prose decode to break ties within
5%, and is confirmed with a
second cold trial, medium thinking, eight clients, and separate warm-prefix
requests. The published recall workload alone does not select the winner.
Vanilla requests explicitly use zero presence/frequency penalties to match
stock. The Huihui control used Cinference's default non-thinking presence
penalty of 1.5; its response style and decode figures are not a controlled
checkpoint-only comparison.

After tuning, `run-block.sh --workload-check` compares the stock launcher and
all four tuned vanilla configurations on five short, reasoning-disabled
conversation prompts and 131K code generation. Stock and the selected coding
profile also process a 260K code prompt. The code context is a deterministic
snapshot of the pinned public runtime's source files; the task produces a
Python metrics reader, with the configured medium reasoning and a 2,048-token
output cap. Stock and the selected profile also run 131K coding with reasoning
disabled and a 1,024-token cap. These are generation throughput probes, not
coding pass-rate tests or full voice-pipeline latency. Reasoning token counts
and finish reasons are retained alongside visible output.
The source inventory, prompt bytes and all answers remain in the raw run.

These are bounded performance probes, not broad model-quality evaluations.
Recall speed can be inflated by high speculative acceptance. The 500K probe
exceeds the model's native 262K context and does not qualify extended-context
quality. Compare actual API usage, server timings, output lengths, and marker
recovery alongside speed. Native timing definitions can differ by engine.

Raw output, server logs, and half-second GPU telemetry stay in ignored
`bench-results/cinference-27`. Process VRAM is separate from total-card VRAM.
Each profile directory is new and cannot overwrite a previous run.
