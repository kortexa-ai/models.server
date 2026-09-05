# Snappy MLX upgrade read

## Package outcome

Latest stable resolver check on 2026-09-05:

- mlx-lm remained 0.31.3 (already latest available).
- mlx-vlm upgraded from 0.6.15 to 0.6.17.
- mlx and mlx-metal remained 0.32.1; all other packages were unchanged.

Used uv pip install --python .venv-mlx/bin/python --upgrade-package mlx-lm
--upgrade-package mlx-vlm mlx-lm mlx-vlm. A broad --upgrade dry run would have
changed 17 packages; the narrow operation changed only the requested vision package.
Inventories are in before.json and after.json. Package metadata was also checked
against https://pypi.org/project/mlx-lm/ and https://pypi.org/project/mlx-vlm/.

No model service or MLX model process was active at the mutation baseline.
Imports of both libraries passed, Metal was available, uv pip check passed,
and both server help commands executed. Vision inference was not benchmarked.
Rollback is uv pip install --python .venv-mlx/bin/python mlx-vlm==0.6.15.
The old package was checked with an offline dry run.

## Execution controls inspected

These are capabilities of the installed versions, not a claim that each was
introduced by this upgrade.

mlx-lm server:

- --prompt-concurrency and --decode-concurrency for batchable requests.
- --prefill-step-size for prefill chunks.
- --prompt-cache-size for retained cache entry count.
- --prompt-cache-bytes for cache memory budgeting.
- Sampling, output-token budget, chat template, and template arguments.

Its server CLI has no --kv-bits setting. The single-generation path does not
forward KV quantization options, and the server's BatchGenerator construction
does not configure them. Lower-level generation APIs contain KV quantization
support, but that is not equivalent to an exposed server control.

mlx-vlm server 0.6.17:

- --kv-bits, --kv-key-bits, --kv-value-bits.
- --kv-quant-scheme and separate key/value schemes (uniform or turboquant).
- --kv-group-size and --quantized-kv-start.
- --max-kv-size (tokens), --max-num-seqs (concurrent sequence backpressure).
- --vision-cache-size, --prefill-step-size, --max-tokens.
- Draft-model controls and thinking budgets.

The existing run-mlx.sh already forwards the common VLM KV controls from model
configuration. It does not map every separate key/value override into model.json;
extra CLI arguments can be forwarded. No launcher or roster changes were made.
Do not switch the text-only 230M into a vision backend merely because flags exist:
model compatibility, correctness, batching, and cache behavior need validation.

## Matched text smoke benchmark

Replayed the exact payloads from hamster-orchestra d4cc839, experiment 5
calibration-verbatim.json: 25 balanced, repeated, very short direction-copying
requests per concurrency, temperature zero, eight output tokens maximum.
Stock LiquidAI/LFM2.5-230M-MLX-8bit through models.server, native KV cache,
server prompt/decode concurrency four, same ascending client-concurrency order.

| Clients | Before requests/s | After requests/s | After correctness |
| --- | --- | --- | --- |
| 1 | 31.4 | 31.3 | 25/25 |
| 4 | 79.4 | 97.8 | 25/25 |
| 8 | 59.2 | 79.1 | 25/25 |

The text backend and its dependencies did not change. These short cache-friendly
batches are affected by warmup, scheduling, and run-to-run variation. They do not
establish an upgrade speedup or a throughput ceiling. Raw post-check payloads,
responses, timing, and expected answers are in benchmark.json.

Transient PID 22860 used loopback port 2045 and a ten-minute watchdog. It was
stopped after the 75 requests; the port was confirmed closed. The separate
hamster observatory static replay viewer remains available without inference.

## Scope

Snappy only. No package upgrade, sync, restart, or GPU operation on Smarty:
the user explicitly reserved it for other work. No managed service changed.
Only evidence documents are delivered through Git; no shared launcher code changed.
