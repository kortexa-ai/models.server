# Qwen on DGX Spark

This note tracks the current single-user DGX Spark setup for the Qwen 3.5
family.

## Bottom Line

- Keep the Linux default on `llama-server` for now.
- Keep the `vLLM` work isolated under [`../vllm-spark/`](../vllm-spark/).
- Do not upgrade NVIDIA drivers on DGX Spark.

For single-user chat on this box, `llama-server` is clearly better on startup
time, steady response speed, and shared-GPU friendliness.

## Current Spark Presets

These are the Linux defaults now baked into the Qwen launchers:

| Model | Quant | Context | Notes |
| --- | --- | ---: | --- |
| `0.8B` | `Q8_0` | `32768` | cheap chat / utility model |
| `2B` | `Q8_0` | `32768` | small chat / reasoning |
| `4B` | `Q8_0` | `32768` | best small-model default |
| `9B` | `Q4_K_M` | `262144` | full context, still very shareable |
| `27B` | `Q4_K_M` | `262144` | full context, dense model; also used for `qwen-3.5-27b-opus` |
| `35B-A3B` | `Q4_K_M` | `163840` | "164k" target implemented as `163840` |

## Measured Llama Footprints

The table below comes from direct `llama-server` startup probes on the Spark
with:

- `--parallel 1`
- `--flash-attn on`
- `--cache-type-k q4_0 --cache-type-v q4_0`
- `--no-mmap`
- `--no-warmup`
- text-only probing via `--no-mmproj`

Totals are the sum of the model, KV, recurrent-state, and compute buffers that
`llama-server` reports at startup.

| Model | Quant | Context | Observed GPU footprint |
| --- | --- | ---: | ---: |
| `0.8B` | `Q8_0` | `32768` | `1378 MiB` (`1.35 GiB`) |
| `2B` | `Q8_0` | `32768` | `2525 MiB` (`2.47 GiB`) |
| `4B` | `Q8_0` | `32768` | `5093 MiB` (`4.97 GiB`) |
| `9B` | `Q4_K_M` | `262144` | `8024 MiB` (`7.84 GiB`) |
| `27B` | `Q4_K_M` | `262144` | `20854 MiB` (`20.37 GiB`) |
| `35B-A3B` | `Q4_K_M` | `163840` | `21949 MiB` (`21.43 GiB`) |

Raw logs:

- `0.8B`: [`vllm-spark/bench-logs/preset-probes/0.8b-preset.log`](../vllm-spark/bench-logs/preset-probes/0.8b-preset.log)
- `2B`: [`vllm-spark/bench-logs/preset-probes/2b-preset.log`](../vllm-spark/bench-logs/preset-probes/2b-preset.log)
- `4B`: [`vllm-spark/bench-logs/preset-probes/4b-preset.log`](../vllm-spark/bench-logs/preset-probes/4b-preset.log)
- `9B`: [`vllm-spark/bench-logs/preset-probes/9b-preset.log`](../vllm-spark/bench-logs/preset-probes/9b-preset.log)
- `27B`: [`vllm-spark/bench-logs/preset-probes/27b-preset.log`](../vllm-spark/bench-logs/preset-probes/27b-preset.log)
- `35B-A3B`: [`vllm-spark/bench-logs/preset-probes/35b-a3b-preset.log`](../vllm-spark/bench-logs/preset-probes/35b-a3b-preset.log)

## Long-Context Scaling

The long-context numbers are better than the raw parameter counts suggest,
because cache growth is driven by the model's attention geometry rather than
just by total parameters.

| Model | `8k` | `64k` | current target |
| --- | ---: | ---: | ---: |
| `9B` | `5477 MiB` | `5981 MiB` | `8024 MiB` at `262144` |
| `27B` | `16061 MiB` | `17069 MiB` | `20854 MiB` at `262144` |
| `35B-A3B` | `21071 MiB` | `21386 MiB` | `21949 MiB` at `163840` |

Key takeaway:

- `35B-A3B` is not the worst long-context memory hog here.
- `27B` grows faster with context because it has a steeper KV-cache slope.
- `9B` is the sweet spot if you want full context while leaving plenty of room
  for other apps.

Reference logs:

- `9B` 8k/64k: [`vllm-spark/bench-logs/ctx-probes/9b-ctx8192.log`](../vllm-spark/bench-logs/ctx-probes/9b-ctx8192.log), [`vllm-spark/bench-logs/ctx-probes/9b-ctx65536.log`](../vllm-spark/bench-logs/ctx-probes/9b-ctx65536.log)
- `27B` 8k/64k: [`vllm-spark/bench-logs/ctx-probes/27b-ctx8192.log`](../vllm-spark/bench-logs/ctx-probes/27b-ctx8192.log), [`vllm-spark/bench-logs/ctx-probes/27b-ctx65536.log`](../vllm-spark/bench-logs/ctx-probes/27b-ctx65536.log)
- `35B-A3B` 8k/64k: [`vllm-spark/bench-logs/ctx-probes/35b-a3b-ctx8192.log`](../vllm-spark/bench-logs/ctx-probes/35b-a3b-ctx8192.log), [`vllm-spark/bench-logs/ctx-probes/35b-a3b-ctx65536.log`](../vllm-spark/bench-logs/ctx-probes/35b-a3b-ctx65536.log)

## vLLM vs llama-server

This comparison used `8k` context and three short follow-up prompts. The
`vLLM` runs were text-only (`LANGUAGE_MODEL_ONLY=1`) and the `llama-server`
runs were text-only (`--no-mmproj`) with `q4_0` KV cache.

| Model | Engine | Startup | GPU footprint | Completion tok/s by turn |
| --- | --- | ---: | ---: | --- |
| `2B` | `vLLM` | `115.224s` | `20588 MiB` | `1.60`, `5.84`, `39.04` |
| `2B` | `llama-server` | `32.029s` | `2444 MiB` | `47.48`, `45.83`, `44.37` |
| `4B` | `vLLM` | `157.307s` | `25018 MiB` | `1.75`, `5.02`, `19.15` |
| `4B` | `llama-server` | `73.074s` | `4877 MiB` | `23.49`, `23.04`, `22.74` |
| `27B` | `vLLM` | `184.339s` | `60458 MiB` | `1.07`, `2.20`, `4.40` |
| `27B` | `llama-server` | `23.074s` | `16061 MiB` | `8.26`, `8.21`, `8.19` |

Artifacts:

- results JSON: [`vllm-spark/bench-results-small-models.json`](../vllm-spark/bench-results-small-models.json)
- `2B` logs: [`vllm-spark/bench-logs/vllm-2b.log`](../vllm-spark/bench-logs/vllm-2b.log), [`vllm-spark/bench-logs/llama-2b.log`](../vllm-spark/bench-logs/llama-2b.log)
- `4B` logs: [`vllm-spark/bench-logs/vllm-4b.log`](../vllm-spark/bench-logs/vllm-4b.log), [`vllm-spark/bench-logs/llama-4b.log`](../vllm-spark/bench-logs/llama-4b.log)
- `27B` logs: [`vllm-spark/bench-logs/vllm-27b.log`](../vllm-spark/bench-logs/vllm-27b.log), [`vllm-spark/bench-logs/llama-27b.log`](../vllm-spark/bench-logs/llama-27b.log)

### Takeaways

- `llama-server` wins clearly for single-user Spark chat workloads.
- `vLLM` was using CUDA correctly, but its startup cost and reserved GPU memory
  were far higher on this machine.
- The small Qwen models (`0.8B`, `2B`, `4B`) are cheap enough at `Q8_0` that
  there is little reason to force them down to smaller quants on Spark.
- `9B` is the best "large context without crowding the box" option.
- `27B` and `35B-A3B` are both workable, but `35B-A3B` deserves more respect
  than raw parameter count would suggest because its context growth is gentler.

## vLLM Folder Layout

All Spark `vLLM` experimentation now lives under [`../vllm-spark/`](../vllm-spark/):

- `Dockerfile.vllm-ngc`
- `docker-build.sh`
- `run-vllm-docker.sh`
- `systemd/kortexa-ai-llm-qwen-3.5-27b-vllm.service`
- `benchmark_vllm_vs_llama.py`
- `bench-results-small-models.json`
- `bench-logs/`
