# Reproducible Model Benchmarks

This directory contains the benchmark plan and tools. The scripts benchmark an
already-running endpoint. They never start, stop, or restart a model service.

Raw runs go under the gitignored `bench-results/` directory. Curated summaries
belong in [`BENCHMARKS.md`](BENCHMARKS.md). The pre-reset log and its bespoke
Qwen harnesses are under [`archive/`](archive/).

## Why the old log was reset

Some historical direct `run.sh` launches may have inherited
`GGML_CUDA_ENABLE_UNIFIED_MEMORY`. On a discrete NVIDIA GPU, that changes
llama.cpp allocations to CUDA managed memory and can severely reduce
throughput. The old logs did not capture the process environment, exact live
arguments, GPU power cap, or CUDA graph state, so their numbers cannot be
audited after the fact.

The new harness refuses to benchmark a live process that contains
`GGML_CUDA_ENABLE_UNIFIED_MEMORY`. Presence enables the llama.cpp behavior even
if the value is `0`.

## Standard tool

Use llama.cpp's upstream
[`SPEED-Bench` client](https://github.com/ggml-org/llama.cpp/tree/master/tools/server/bench/speed-bench)
as the standard text-generation workload. It targets an already-running
OpenAI-compatible llama-server, uses NVIDIA's public SPEED-Bench dataset, and
records prompt throughput, decode throughput, request latency, and speculative
draft acceptance. Fixed-input variants cover 1K through 32K prompts.

We do not vendor the client or dataset. The wrapper uses the client in the
local `~/src/llama.cpp` checkout and records that checkout's exact Git commit.
This keeps the benchmark implementation auditable without maintaining a fork.

`llama-bench` remains useful for isolated kernels, but it excludes
tokenization and sampling and does not measure the production server path. It
is therefore a diagnostic tool, not the primary score.

## One-time setup

```bash
./bench/setup.sh
```

This creates `.venv-bench/` and installs the upstream SPEED-Bench client's
dependencies. It does not download a model, start a server, or run a benchmark.
Dependency versions are captured in every run manifest.

Validate the local tools without inference:

```bash
./bench/test-tools.sh
```

## Dry run first

```bash
./bench/run-suite.sh lfm2.5-vl-3b
```

Without `--execute`, the command checks the model and prints the planned test
matrix. It sends no inference requests and creates no result directory.

To run later, after inspecting the live service and obtaining any required GPU
downtime authorization:

```bash
./bench/run-suite.sh lfm2.5-vl-3b --execute
```

Useful overrides:

```bash
./bench/run-suite.sh qwen-3.8-27b --execute --suite smoke
./bench/run-suite.sh qwen-3.8-27b --execute --suite standard
./bench/run-suite.sh qwen-3.8-27b --execute --suite smoke --workload mixed-chat
./bench/run-suite.sh lfm2.5-vl-3b --execute --suite smoke --workload vision-pipeline
./bench/run-suite.sh qwen-3.8-27b --execute --output bench-results/manual-name
```

The default `standard` suite runs capability canaries, the qualitative
SPEED-Bench split, fixed 1K and 8K input splits, a 32K split when the live
context permits it, and a live-slot concurrency pass when the model has
more than one slot. The `smoke` suite uses one sample from three categories.

The optional `mixed-chat` workload runs four independent Qwen-style sessions
with approximately 512, 8K, 32K, and 131K words of existing history. It sends
two rounds per session with prompt caching enabled, retaining the first model
response in the second request. This exposes latency, throughput, and cache
reuse under uneven simultaneous KV demand.

The optional `vision-pipeline` workload sends 16 distinct deterministic 1024px
images as independent sessions. Prompt caching is disabled, and each response
is bounded to 64 tokens. Run it once for each deliberately prepared live slot
count; the client concurrency follows `/props.total_slots`, so comparisons of
1, 2, 4, and 8 slots measure the server configuration rather than a client
queue setting.

## Required run record

Each run directory contains:

- `manifest.json`: timestamps, host and OS, repository commits, executable
  version and SHA-256, live PID, exact command line, filtered GGML/CUDA
  environment, GPU residency, CUDA graph state, unified-memory state, GPU UUID,
  and GPU power cap. CPU-only processes using `--device none` record CUDA
  inference as false and graph state as `not_applicable`, even if the CUDA build
  creates a small driver context visible to `nvidia-smi`.
- `model.json`: byte-for-byte snapshot of the model configuration.
- `effective-config.env`: values produced by `scripts/parse-config.py`.
- `props.json`: the live llama-server `/props` response.
- `python-packages.txt`: exact benchmark-client dependency versions.
- `speed-bench-dataset-revision.txt`: pinned Hugging Face dataset commit.
- `gpu-before.csv`, `gpu-telemetry.csv`, and `gpu-after.csv`.
- `canaries.json`: text, tool-call, vision, or embedding capability probes.
- `speed-*.json` and matching console logs from SPEED-Bench.
- `workload-*.json` and matching console logs for an optional focused workload.

Keep raw results local because they can be large. Add a concise summary to
`BENCHMARKS.md` with the run directory name, model ID, model-config SHA-256,
llama.cpp commit, power cap, graph state, context, slots, and the result table.

## Test battery

1. **Preflight and provenance**: health, port owner, live PID, executable,
   source commits, exact configuration, command line, safe environment, driver,
   GPU identity, power cap, clocks, graph state, and unified-memory refusal.
2. **Capability canaries**: deterministic text, required tool call, 512px and
   1536px image requests for VLMs, or a small embedding batch for embedding
   models. These are compatibility gates, not quality scores.
3. **Single-user workload mix**: SPEED-Bench qualitative categories with
   deterministic sampling and fixed output length.
4. **Fixed-shape throughput**: 1K/512 and 8K/512 input/output runs; add
   32K/512 only when it fits in one configured slot.
5. **Configured concurrency**: repeat a bounded qualitative workload at the
   live slot count. Report both aggregate throughput and per-request latency.
6. **Cache behavior**: the capability probe repeats an identical text prefix;
   retain both cold and warm server timings.
7. **Stability**: after the first clean sweep, promote important production
   models to a separate 15-minute soak. Do not mix soak results with the short
   throughput table.

Use 450 W as the normal comparison baseline unless the experiment says
otherwise. A 600 W result must be a separate run, ideally paired with the same
model, configuration, workload, and ambient service baseline at 450 W. Never
combine samples across power caps or CUDA graph states.

## Result acceptance rules

- The endpoint must be an existing managed or deliberately prepared service;
  the harness does not manipulate it.
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY` must be absent from the live process.
- The power limit must remain constant for the run.
- CUDA graph state must come from the live process environment, not an
  assumption based on source configuration.
- Warm-ups do not count as measured samples.
- Compare only matching model, quant, KV type, total KV allocation, per-request
  context limit, unified-KV state, slot count, speculative mode/depth, graph
  state, power cap, workload, and client version.
- Record failures. A crash, invalid tool call, or failed vision request is a
  result, not a sample to quietly throw into the swamp.
