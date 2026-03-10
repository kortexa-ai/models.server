# SGLang Spark Comparison Notes

This document tracks the `SGLang + NVFP4` vs `llama.cpp` comparison work for
the Qwen 3.5 family on the Spark.

## Goal

Build a repeatable path to compare:

- `SGLang` serving either standard `Qwen/*` or `AxionML/*-NVFP4`
- current `llama.cpp` serving `unsloth/*-GGUF`

Primary metrics:

- startup time
- GPU memory footprint after load
- single-request TPS, cold and warm
- multi-request concurrency throughput
- five-message chat history request behavior
- reasoning enabled vs disabled

## Harness

Files:

- `sglang.spark/run.sh`
- `sglang.spark/run-docker.sh`
- `sglang.spark/resolve_model.py`
- `sglang.spark/models.json`
- `sglang.spark/benchmark_sglang_vs_llama.py`
- `sglang.spark/README.md`

Kick off a model download/start with:

```bash
./sglang.spark/run.sh 0.8b
```

Known presets:

| Model | Standard SGLang | Axion NVFP4 | Current llama.cpp baseline | Status |
| --- | --- | --- | --- | --- |
| `0.8b` | `Qwen/Qwen3.5-0.8B` | `AxionML/Qwen3.5-0.8B-NVFP4` | `unsloth/Qwen3.5-0.8B-GGUF` | standard verified; nvfp4 blocked on loader |
| `2b` | `Qwen/Qwen3.5-2B` | `AxionML/Qwen3.5-2B-NVFP4` | `unsloth/Qwen3.5-2B-GGUF` | pending |
| `4b` | `Qwen/Qwen3.5-4B` | `AxionML/Qwen3.5-4B-NVFP4` | `unsloth/Qwen3.5-4B-GGUF` | pending |
| `9b` | `Qwen/Qwen3.5-9B` | `AxionML/Qwen3.5-9B-NVFP4` | `unsloth/Qwen3.5-9B-GGUF` | pending |
| `27b` | `Qwen/Qwen3.5-27B` | `AxionML/Qwen3.5-27B-NVFP4` | `unsloth/Qwen3.5-27B-GGUF` | pending |

## Bring-Up Notes

### March 8, 2026

- `scitrera/dgx-spark-sglang:0.5.9-t5` successfully booted
  `Qwen/Qwen3.5-0.8B` via:

```bash
./sglang.spark/run-docker.sh --profile standard 0.8b
```

- Confirmed milestones for the Docker `0.8b` standard run:
  - weight load succeeded
  - KV cache allocation succeeded
  - CUDA graph capture succeeded
  - API came up on `http://127.0.0.1:2131`
  - `POST /v1/chat/completions` returned a valid response

- Smoke test request:

```json
{
  "model": "qwen-3.5-0.8b",
  "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
  "max_tokens": 16,
  "temperature": 0,
  "stream": false,
  "chat_template_kwargs": {"enable_thinking": false},
  "separate_reasoning": true
}
```

- Smoke test response summary:
  - reply: `ok`
  - prompt tokens: `17`
  - completion tokens: `2`

- Practical conclusion:
  - Docker is now the trusted path for the upcoming `0.8b` comparisons.
  - Bare-metal `sglang==0.5.9` remains useful only as a bring-up attempt log,
    not as the benchmark path we should trust on this machine.

### March 9, 2026

- Warm single-request no-reasoning spot check on
  `scitrera/dgx-spark-sglang:0.5.9-t5` with `Qwen/Qwen3.5-0.8B`:
  - run 1: `227` output tokens in `2.609s` -> `87.0 tok/s`
  - run 2: `227` output tokens in `2.611s` -> `86.95 tok/s`
  - warm average: `86.97 tok/s`

- Warm single-request no-reasoning spot check on
  `scitrera/dgx-spark-sglang:0.5.9-dev1-329817e2-t5` with `Qwen/Qwen3.5-0.8B`:
  - run 1: `227` output tokens in `2.441s` -> `92.99 tok/s`
  - run 2: `227` output tokens in `2.435s` -> `93.23 tok/s`
  - warm average: `93.11 tok/s`

- Fresh-process `llama.cpp` no-reasoning baseline on `unsloth/Qwen3.5-0.8B-GGUF:Q8_0`
  with `llama-server`:
  - cold run: `213` output tokens in `3.031s` -> `70.27 tok/s`
  - warm run 1: `213` output tokens in `3.024s` -> `70.43 tok/s`
  - warm run 2: `213` output tokens in `3.010s` -> `70.77 tok/s`
  - warm average: `70.60 tok/s`
  - `llama.cpp` server-side decode timing on those same requests was about
    `72.0 tok/s`; the lower API number is mostly prompt and request overhead
  - loaded GPU memory reported by `llama-server` at startup:
    `CUDA0 model buffer 763.78 MiB`, `KV 108.00 MiB`, `RS 19.27 MiB`,
    `compute 487.00 MiB` -> about `1378 MiB` total

- Dev-tag versus stable-image delta for the same standard `0.8b` no-reasoning
  spot check:
  - `93.11 tok/s` vs `86.97 tok/s`
  - about `+7.1%` on the dev tag

- Standard SGLang dev-tag versus current `llama.cpp Q8_0` on the same
  no-reasoning spot check:
  - `93.11 tok/s` vs `70.60 tok/s`
  - about `+31.9%` for SGLang on warm single-request end-to-end output TPS

- Same-session `standard SGLang` versus `llama.cpp Q4_K_M` no-reasoning
  single-request spot check on `0.8b`:
  - SGLang image:
    `scitrera/dgx-spark-sglang:0.5.9-dev1-329817e2-t5`
  - llama.cpp model:
    `unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M`
  - SGLang cold run:
    `206` output tokens in `7.579s` -> `27.18 tok/s`
  - SGLang warm runs:
    `227` output tokens in `2.702s` -> `84.01 tok/s`
    `227` output tokens in `2.651s` -> `85.64 tok/s`
    `227` output tokens in `2.711s` -> `83.72 tok/s`
    `227` output tokens in `2.651s` -> `85.64 tok/s`
    `227` output tokens in `2.635s` -> `86.14 tok/s`
  - SGLang warm average:
    `85.03 tok/s`
  - llama.cpp `Q4_K_M` cold run:
    `177` output tokens in `2.452s` -> `72.17 tok/s`
  - llama.cpp `Q4_K_M` warm runs:
    `177` output tokens in `2.474s` -> `71.55 tok/s`
    `177` output tokens in `2.587s` -> `68.41 tok/s`
    `177` output tokens in `2.596s` -> `68.18 tok/s`
  - llama.cpp `Q4_K_M` warm average:
    `69.38 tok/s`
  - warm delta:
    `85.03 tok/s` vs `69.38 tok/s`
    about `+22.6%` for SGLang on end-to-end API output TPS
  - llama.cpp server-side decode timing for those warm `Q4_K_M` runs was
    about `73.51`, `70.49`, and `69.67 tok/s`, so its API measurement is close
    to the raw decode number

- Same-session `standard SGLang` versus `llama.cpp Q4_K_M` no-reasoning
  single-request spot check on `9b` at full `262144` context:
  - SGLang image:
    `scitrera/dgx-spark-sglang:0.5.9-dev1-329817e2-t5`
  - llama.cpp model:
    `unsloth/Qwen3.5-9B-GGUF:Q4_K_M`
  - SGLang cold run:
    `256` output tokens in `24.800s` -> `10.32 tok/s`
  - SGLang warm runs:
    `256` output tokens in `20.174s` -> `12.69 tok/s`
    `256` output tokens in `20.178s` -> `12.69 tok/s`
    `256` output tokens in `20.179s` -> `12.69 tok/s`
  - SGLang warm average:
    `12.69 tok/s`
  - llama.cpp `Q4_K_M` cold run:
    `256` output tokens in `10.496s` -> `24.39 tok/s`
  - llama.cpp `Q4_K_M` warm runs:
    `256` output tokens in `10.337s` -> `24.77 tok/s`
    `256` output tokens in `10.274s` -> `24.92 tok/s`
    `256` output tokens in `10.319s` -> `24.81 tok/s`
  - llama.cpp `Q4_K_M` warm average:
    `24.83 tok/s`
  - warm delta:
    `24.83 tok/s` vs `12.69 tok/s`
    about `+95.7%` for llama.cpp, or about `1.96x` faster than standard SGLang
  - llama.cpp server-side decode timing for those warm runs was about
    `25.06`, `25.15`, and `25.03 tok/s`
  - llama.cpp startup memory breakdown at `262144` context:
    `CUDA0 model 4861 MiB`, `context 2354 MiB`, `compute 808 MiB` ->
    about `8023 MiB` on GPU

- Tried an alternative FP4-family checkpoint:
  `olka-fi/Qwen3.5-9B-MXFP4`
  - first attempt with `--quantization mxfp4` failed immediately because the
    model config declares `quant_method: compressed-tensors`, so SGLang raised:

```text
ValueError: Quantization method specified in the model config (compressed-tensors)
does not match the quantization method specified in the `quantization`
argument (mxfp4).
```

  - second attempt with no quantization override got further and confirmed the
    repo metadata is:
    - `quant_method: compressed-tensors`
    - `format: mxfp4-pack-quantized`
  - but model loading still failed before ready state with:

```text
ImportError: Other method (CompressedTensorsW4A16Sparse24) is not supported now
```

  - practical conclusion:
    - this `olka-fi` MXFP4 repo is visible to SGLang and gets past initial
      repo/config recognition
    - but the current Spark SGLang image still cannot load the actual
      compressed-tensors scheme used by this checkpoint, so it is not a usable
      benchmark fallback today

- Separate bring-up check outside the SGLang path:
  `vllm/vllm-openai:cu130-nightly`
  - image booted successfully with:

```bash
docker run --rm --name vllm-qwen35-08b --gpus all --network host --ipc host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --entrypoint vllm vllm/vllm-openai:cu130-nightly \
  serve Qwen/Qwen3.5-0.8B \
  --host 0.0.0.0 \
  --port 2211 \
  --served-model-name qwen-3.5-0.8b \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.25
```

  - default `--gpu-memory-utilization 0.9` failed startup on this machine
    before model load, but reducing it to `0.25` fixed bring-up
  - nightly image identified itself as
    `v0.17.0rc1.dev164+gfff3711a2`
  - vLLM resolved the model as `Qwen3_5ForConditionalGeneration`
  - `/v1/models` came up successfully and a real `POST /v1/chat/completions`
    request completed successfully on port `2211`

- `vLLM` no-reasoning spot check for `0.8b` on
  `vllm/vllm-openai:cu130-nightly`:
  - command used the standard `Qwen/Qwen3.5-0.8B` checkpoint with:
    - `--dtype bfloat16`
    - `--max-model-len 32768`
    - `--gpu-memory-utilization 0.25`
  - benchmark requests explicitly set:
    `chat_template_kwargs: {"enable_thinking": false}`
  - first benchmark pass after startup was a slow outlier:
    `206` tokens in `76.518s` -> `2.69 tok/s`
  - warm runs:
    `206` tokens in `2.024s` -> `101.80 tok/s`
    `206` tokens in `1.948s` -> `105.73 tok/s`
  - warm average:
    `103.77 tok/s`
  - versus earlier warm baselines:
    - `vLLM 103.77 tok/s`
    - `SGLang 85.03 tok/s`
    - `llama.cpp Q4_K_M 69.38 tok/s`
  - practical read:
    - `vLLM` is about `+22.0%` versus `SGLang`
    - `vLLM` is about `+49.6%` versus `llama.cpp Q4_K_M`

- `vLLM` no-reasoning spot check for `9b` on
  `vllm/vllm-openai:cu130-nightly`:
  - command used the standard `Qwen/Qwen3.5-9B` checkpoint with:
    - `--dtype bfloat16`
    - `--max-model-len 262144`
    - `--gpu-memory-utilization 0.25`
  - benchmark requests explicitly set:
    `chat_template_kwargs: {"enable_thinking": false}`
  - measured runs:
    `256` tokens in `19.673s` -> `13.01 tok/s`
    `256` tokens in `19.680s` -> `13.01 tok/s`
    `256` tokens in `25.404s` -> `10.08 tok/s`
  - average across those no-thinking runs:
    `12.03 tok/s`
  - versus earlier warm baselines:
    - `vLLM 12.03 tok/s`
    - `SGLang 12.69 tok/s`
    - `llama.cpp Q4_K_M 24.83 tok/s`
  - practical read:
    - `vLLM` is about `-5.2%` versus `SGLang`
    - `vLLM` is about `-51.5%` versus `llama.cpp Q4_K_M`

- `AxionML/Qwen3.5-0.8B-NVFP4` is currently blocked in Docker on this machine.
  The exact same `AssertionError` during weight loading reproduced on:
  - `scitrera/dgx-spark-sglang:0.5.9-t5`
  - `scitrera/dgx-spark-sglang:0.5.9-dev1-329817e2-t5`

- Both `scitrera` images reached `ModelOptModelLoader`, detected the
  checkpoint as `nvfp4`, and then failed while loading weights with:

```text
Parameter model.layers.0.linear_attn.in_proj_a.input_scale not found in params_dict
...
File ".../sglang/srt/layers/linear.py", line 418, in weight_loader
    assert param_data.shape == loaded_weight.shape
AssertionError
```

- `lmsysorg/sglang:spark` was also tested, but it fails earlier because that
  image ships `sglang 0.5.4.post2` and `transformers 4.57.1`, which do not
  recognize the `qwen3_5` architecture at all.

- Practical conclusion:
  - standard `0.8b` SGLang is benchmarkable today
  - Axion `0.8b` NVFP4 is not yet benchmarkable in the tested Spark Docker
    images because the checkpoint loader is failing before the server becomes ready

## Current Installation Status

Direct install is partially working:

- created `sglang.spark/.venv` with Python `3.12`
- installed `sglang==0.5.9`
- installed `sgl-kernel==0.3.21`
- reinstalled `torch==2.9.1+cu130` after the `sglang` install so CUDA stays enabled
- installed CUDA 12 compatibility libs for the current `sgl_kernel` wheel
- upgraded compatibility CuDNN to `nvidia-cudnn-cu12==9.16.0.29`
- verified `torch.cuda.is_available() == True`
- defaulted SGLang launches to `--attention-backend triton` because Qwen 3.5
  on Blackwell asserts unless the backend is explicitly `triton` or `trtllm_mha`
- defaulted `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` so Triton uses CUDA
  13's assembler instead of the bundled Triton `ptxas 12.8`, which does not
  recognize `sm_121a`
- patched `GemmaRMSNorm` to use native PyTorch on GB10 because the packaged
  `sgl_kernel` CUDA op throws `no kernel image is available for execution on the device`

Important caveat:

- This native fallback means bare-metal SGLang numbers on this machine are now
  a workaround path, not a clean representation of the intended optimized kernel
  stack. The NVIDIA Docker route is still the likely path for trustworthy final
  performance comparisons.

Caveat:

- the local PyTorch build warns that GB10 reports compute capability `12.1`
  while that wheel advertises support through `12.0`

So the bare-metal path is promising enough to keep, but it is not fully
trusted yet. If real server startup or generation fails, the next fallback is
the NVIDIA DGX Spark `SGLang` Docker images.

## Important Baseline Note

The current repo defaults are not all `Q4`:

- `0.8b`, `2b`, and `4b` currently default to `Q8_0`
- `9b` and `27b` currently default to `Q4_K_M`

If the desired comparison is specifically `NVFP4 vs Q4`, the benchmark runner
should be launched with `--llama-quant Q4_K_M` for the smaller models instead
of using their current default.

## Scenario Design

### 1. Single request, cold

Measure the first request after the server is ready. This captures:

- first real prefill path
- first decode path
- any runtime cache misses that still happen after startup

### 2. Single request, warm

Repeat the same request immediately. This captures:

- warm kernels
- any prefix/radix cache benefits
- steady-state single-user latency

### 3. Concurrent requests

Send the same prompt with multiple simultaneous clients. This is where we can
see whether `SGLang` gains more from scheduler behavior than `llama.cpp`.

### 4. Five-message history request

Keep this scenario. It is not a fundamentally different decode regime, but it
is still useful because it stresses:

- prompt prefill cost
- tokenizer throughput
- cache reuse on repeated chat prefixes

One history scenario should be enough. We probably do not need multiple chat
history variants unless the first results are surprising.

## Reasoning Control

`SGLang` supports Qwen reasoning control at request time. The intended request
shape is:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true
  },
  "separate_reasoning": true
}
```

For the no-reasoning comparison, set `enable_thinking` to `false`.

Because Qwen 3.5 reasoning can be token-hungry, the harness defaults should
stay around:

- no reasoning: `256` max tokens
- reasoning on: `8192` max tokens

If the reasoning traces are still being cut off, increase the reasoning budget
further before drawing speed conclusions. On March 8, 2026 the Dockerized
`0.8b` standard model exhausted a `4096` token reasoning budget without
reaching the visible answer, so `2048` is clearly too small for this machine's
Qwen 3.5 thinking runs.

## Comparison Order

1. `SGLang standard` vs current `llama.cpp` baseline
2. `SGLang NVFP4` vs the same `llama.cpp` baseline
3. For the smaller models, rerun with `--llama-quant Q4_K_M` if we want the
   stricter `NVFP4 vs Q4` apples-to-apples view

This lets us separate:

- raw harness or scheduler differences
- incremental speedup from NVIDIA's NVFP4 path on GB10

## Planned Commands

Direct install:

```bash
./sglang.spark/setup.sh
```

Start a model:

```bash
./sglang.spark/run.sh 0.8b
./sglang.spark/run.sh --profile standard 0.8b
./sglang.spark/run.sh --profile nvfp4 0.8b
./sglang.spark/run-docker.sh --profile standard 0.8b
```

Benchmark standard first:

```bash
python3 ./sglang.spark/benchmark_sglang_vs_llama.py 0.8b --sglang-profile standard --reasoning both
```

Then benchmark NVFP4:

```bash
python3 ./sglang.spark/benchmark_sglang_vs_llama.py 0.8b --sglang-profile nvfp4 --reasoning both
```

Strict `NVFP4 vs Q4_K_M` for the smaller models:

```bash
python3 ./sglang.spark/benchmark_sglang_vs_llama.py 0.8b --sglang-profile nvfp4 --reasoning both --llama-quant Q4_K_M
```

## March 9, 2026 Bare-Metal Reset

A new clean harness now lives in `bare.spark/`. The goal is to mirror the
working Docker environments more faithfully on bare metal instead of continuing
to patch the older mixed `cu12/cu13` `sglang.spark/.venv`.

### New Bare-Metal Layout

- `bare.spark/setup-vllm.sh`
- `bare.spark/setup-sglang.sh`
- `bare.spark/run-vllm.sh`
- `bare.spark/run-sglang.sh`
- `bare.spark/doctor.sh`

The new layout uses separate envs:

- `bare.spark/.venv-vllm`
- `bare.spark/.venv-sglang`

This keeps `vllm` and `SGLang` from silently downgrading each other's `torch`,
`triton`, or `transformers` pins.

### Bare-Metal vLLM Result

The fresh bare-metal `vllm` bring-up for `Qwen/Qwen3.5-0.8B` is working.

Environment:

- Python `3.12.3`
- `torch 2.10.0+cu130`
- `vllm 0.17.0rc1.dev182+g55d27cca5.cu130`
- `transformers 4.57.6`
- `triton 3.6.0`
- `flashinfer-python 0.6.4`

Important launcher fix:

- bare-metal `vllm` initially failed because Docker had left a root-owned
  `~/.cache/vllm/torch_compile_cache`
- `bare.spark/run-vllm.sh` now forces:
  - `VLLM_CACHE_ROOT=bare.spark/.cache/vllm`
  - `VLLM_CONFIG_ROOT=bare.spark/.config/vllm`
  - `TORCHINDUCTOR_CACHE_DIR=bare.spark/.cache/vllm/torch_compile_cache`
- the launcher also now forces:
  - `HF_HOME=bare.spark/.cache/huggingface`
  - `HF_HUB_CACHE=bare.spark/.cache/huggingface/hub`
  - `TRANSFORMERS_CACHE=bare.spark/.cache/huggingface/transformers`
  so root-owned Hugging Face cache leftovers from Docker do not spam
  permission warnings during bare-metal runs

Proof points:

- `/health` returned `200`
- `/v1/models` returned `qwen-3.5-0.8b`
- a real `POST /v1/chat/completions` request completed successfully on the
  bare-metal server

Working command:

```bash
./bare.spark/run-vllm.sh 0.8b
```

Bare-metal no-reasoning TPS spot checks:

- `Qwen/Qwen3.5-0.8B` at `32768` context:
  - cold run:
    `206` output tokens in `51.432s` -> `4.01 tok/s`
  - warm runs:
    `206` output tokens in `1.977s` -> `104.18 tok/s`
    `206` output tokens in `1.990s` -> `103.50 tok/s`
  - warm average:
    `103.84 tok/s`
- `Qwen/Qwen3.5-9B` at `262144` context:
  - cold run:
    `256` output tokens in `75.372s` -> `3.40 tok/s`
  - warm runs:
    `256` output tokens in `19.962s` -> `12.82 tok/s`
    `256` output tokens in `22.341s` -> `11.46 tok/s`
  - warm average:
    `12.14 tok/s`

Request settings for both spot checks:

- `temperature=0`
- `max_tokens=256`
- `chat_template_kwargs: {"enable_thinking": false}`
- `separate_reasoning: true`

### Bare-Metal SGLang Status

The fresh `SGLang` env is much cleaner than the old attempt, but `0.8b`
standard is still blocked at the native kernel layer.

Current environment:

- Python `3.12.3`
- `torch 2.10.0+cu130`
- source-installed `sglang` from `sgl-project/sglang`
- `sgl-kernel 0.3.21`
- `transformers 5.3.0`
- `triton 3.6.0`
- `flashinfer-python 0.6.5`

What we learned:

1. Public `sglang==0.5.9` is not a good base for this machine.
   It hard-pins `torch==2.9.1`, `transformers==4.57.1`,
   `flashinfer_python==0.6.3`, and `cuda-python==12.9`.
2. Installing the current `sglang` code with `--no-deps` plus a
   container-shaped runtime stack works better.
3. The prebuilt `sgl-kernel` wheel still does not work cleanly on this stack:
   - first it wanted CUDA 12 runtime libs
   - after adding those, it failed with a Torch ABI symbol mismatch
4. Rebuilding `sgl-kernel` from source is the right next direction, but that
   native build is not done yet.

Current source-build blockers seen so far:

- missing `libnuma-dev` on the host
- then a CMake 4 compatibility failure inside a pulled dependency
  (`dlpack`) that suggests using
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`

So the current honest status is:

- `vllm` bare metal: proven for `Qwen/Qwen3.5-0.8B`
- `SGLang` bare metal: env is rebuilt and much closer to the working container,
  but still blocked on native `sgl-kernel` buildability for GB10

## March 9, 2026 4B Quant Candidate Sweep

Target model size for the next comparison pass: `4B`.

Rationale:

- it is large enough for quantization differences to show up more clearly than
  `0.8B`
- it is still small enough that bring-up failures do not waste as much time as
  `9B` or `27B`
- current `llama.cpp` comparison on this machine is already anchored around
  `Qwen3.5-4B` `GGUF`

### Candidate Repos

- `AxionML/Qwen3.5-4B-NVFP4`
  - target format: `NVFP4`
  - repo metadata: `safetensors`, `qwen3_5`, `NVFP4`, `sglang`
  - best-case target for Blackwell GB10 if it loads cleanly
  - first try in `SGLang`, then consider `vLLM` if the format is recognized

- `osoleve/Qwen3.5-4B-Base-Text-NVFP4`
  - target format: `NVFP4`
  - repo metadata: `qwen3_5_text`, `nvfp4`, `modelopt`
  - README includes `vLLM` usage with `--quantization modelopt`
  - not apples-to-apples with `Qwen/Qwen3.5-4B` because it is the `Base-Text`
    branch, but it may be the cleanest `vLLM` FP4 bring-up candidate

- `olka-fi/Qwen3.5-4B-MXFP4`
  - target format: `MXFP4`
  - repo metadata: `mxfp4`, `compressed-tensors`
  - important risk:
    the `9B` sibling previously failed in `SGLang` with unsupported
    `CompressedTensorsW4A16Sparse24`, so this is worth one quick smoke test but
    should not be the main line of attack

- `QuantTrio/Qwen3.5-4B-AWQ`
  - target format: `AWQ 4-bit`
  - repo metadata: `AWQ`, `4-bit`, `vLLM`
  - README explicitly says it is compatible with `vLLM` and `SGLang`
  - strongest likely-to-work 4-bit fallback if FP4-family checkpoints keep
    failing

- `Intel/Qwen3.5-4B-int4-AutoRound`
  - target format: `int4` / `AutoRound`
  - repo metadata: `4-bit`, `auto-round`
  - README includes direct `vLLM` usage
  - strong practical fallback for `vLLM` if `AWQ` underperforms or fails

- `unsloth/Qwen3.5-4B-GGUF`
  - target format: `GGUF`
  - current stable `llama.cpp` source for `Q4_K_M`
  - available files include `Q4_0`, `Q4_1`, `Q4_K_M`, `Q4_K_S`, `IQ4_NL`,
    and `IQ4_XS`
  - keep `Q4_K_M` as the main llama baseline; test `IQ4_XS` only if the goal
    shifts from fair comparison to absolute speed chasing

- `lovedheart/Qwen3.5-4B-FP8`
  - target format: `FP8`
  - not a 4-bit model, but worth trying if the goal becomes "fastest workable
    non-BF16 path" rather than strictly "smallest quant"

### Recommended Order

1. `AxionML/Qwen3.5-4B-NVFP4`
   - highest upside for GB10 if it works
2. `olka-fi/Qwen3.5-4B-MXFP4`
   - one quick smoke test only; fail fast if compressed-tensors blocks again
3. `QuantTrio/Qwen3.5-4B-AWQ`
   - most promising 4-bit fallback with good odds of loading
4. `Intel/Qwen3.5-4B-int4-AutoRound`
   - second strong `vLLM`-friendly 4-bit fallback
5. `unsloth/Qwen3.5-4B-GGUF:Q4_K_M`
   - current llama baseline to compare against everything else
6. `lovedheart/Qwen3.5-4B-FP8`
   - fallback if the question changes from "4-bit-ish" to "fastest practical"

### Formats To Deprioritize

- `bitsandbytes` 4-bit
  - easy to find, but usually not the speed-first choice
- `MLX` `NVFP4` / `MXFP4` repos
  - relevant for Apple / MLX stacks, not for our `vLLM` / `SGLang` / `llama`
    comparison on GB10

### First 4B Results

Environment notes that mattered:

- `bare.spark/setup-vllm.sh` now installs `ninja`
- `bare.spark/run-vllm.sh` now prepends the venv `bin/` directory to `PATH`
  so FlashInfer JIT subprocesses can actually find `ninja`
- these two harness fixes were required to get `AxionML/Qwen3.5-4B-NVFP4`
  past the FP4 kernel build stage on bare-metal `vLLM`

No-thinking spot checks used:

- `temperature=0`
- `max_tokens=256`
- `chat_template_kwargs: {"enable_thinking": false}` for `vLLM`
- `separate_reasoning: true` for `vLLM`

Current `4B` comparison snapshot:

- `llama.cpp` `unsloth/Qwen3.5-4B-GGUF:Q8_0`
  - existing systemd service on port `2029`
  - warm runs:
    `35.89 tok/s`, `36.04 tok/s`
  - warm average:
    `35.96 tok/s`

- `llama.cpp` `unsloth/Qwen3.5-4B-GGUF:Q4_K_M`
  - ad hoc run on port `2139`
  - warm runs:
    `38.06 tok/s`, `38.35 tok/s`
  - warm average:
    `38.21 tok/s`
  - startup memory summary:
    `CUDA0 model 2603.50 MiB`, `KV 288.00 MiB`, `RS 50.25 MiB`,
    `compute 490.00 MiB`, plus `223.30 MiB` vision warmup compute buffer

- `vLLM` `AxionML/Qwen3.5-4B-NVFP4` with `--quantization modelopt_fp4`
  - bare-metal bring-up now succeeds on port `2239`
  - model load reported:
    `3.85 GiB` model memory
  - warm runs:
    `38.48 tok/s`, `38.42 tok/s`
  - warm average:
    `38.45 tok/s`
  - critical caveat:
    output quality is broken; even a trivial `Reply with exactly: ok` prompt
    produced corrupted text like
    `mutmut styr...`
  - practical status:
    interesting speed, not yet a valid serving result

- `vLLM` `QuantTrio/Qwen3.5-4B-AWQ` with `--quantization awq_marlin`
  - bare-metal bring-up succeeds on port `2239`
  - model load reported:
    `5.58 GiB` model memory
  - warm runs:
    `33.14 tok/s`, `33.16 tok/s`
  - warm average:
    `33.15 tok/s`
  - simple sanity prompt:
    `Reply with exactly: ok` -> `ok`
  - practical status:
    valid text output, but slower than both `llama.cpp Q8_0` and
    `llama.cpp Q4_K_M` on this spot check

- `vLLM` `Intel/Qwen3.5-4B-int4-AutoRound`
  - failed before model load with tokenizer packaging issues:

```text
ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.
```

  - `--trust-remote-code` did not help because the tokenizer path still did
    not resolve in this nightly build

Practical ranking from this first `4B` sweep:

1. `llama.cpp Q4_K_M`
   - `38.21 tok/s`
   - currently the fastest valid result
2. `vLLM Axion NVFP4`
   - `38.45 tok/s`
   - nominally fastest, but invalid due to corrupted output
3. `llama.cpp Q8_0`
   - `35.96 tok/s`
4. `vLLM AWQ Marlin`
   - `33.15 tok/s`

Next likely follow-ups:

- try `AxionML/Qwen3.5-4B-NVFP4` with a different backend mode or eager mode to
  see if the corruption is a compiled-kernel issue rather than bad weights
- smoke-test `olka-fi/Qwen3.5-4B-MXFP4` once, but fail fast if
  `compressed-tensors` blocks again
- try a text-only ModelOpt FP4 repo such as
  `osoleve/Qwen3.5-4B-Base-Text-NVFP4`

### Follow-Up: `olka-fi` and `osoleve`

- `vLLM` `olka-fi/Qwen3.5-4B-MXFP4` with `--quantization compressed-tensors`
  - bare-metal bring-up succeeds on port `2239`
  - model load reported:
    `5.5 GiB` model memory
  - warm runs:
    `33.83 tok/s`, `33.81 tok/s`
  - warm average:
    `33.82 tok/s`
  - simple sanity prompt:
    `Reply with exactly: ok` -> `ok`
  - important caveat:
    `vLLM` warned that this path is not using native FP4 compute and is instead
    using weight-only FP4 compression via Marlin, so this should not be treated
    as the Blackwell-native FP4 fast path
  - practical status:
    valid, but no faster than `AWQ Marlin`, and still slower than
    `llama.cpp Q4_K_M`

- `vLLM` `osoleve/Qwen3.5-4B-Base-Text-NVFP4` with `--quantization modelopt`
  - failed before model load with:

```text
The checkpoint you are trying to load has model type `qwen3_5_text`
but Transformers does not recognize this architecture.
```

  - practical status:
    blocked by `transformers` support in the current `vLLM` env, not by an
    obvious quantization-kernel failure

### Bare-Metal `vLLM` Upgrade Follow-Up

- upgraded `bare.spark/.venv-vllm` from `transformers 4.57.6` to
  `transformers 5.3.0`
- the `vLLM` CLI and imports still worked after the upgrade
- this upgrade fixed the original `qwen3_5_text` architecture-recognition
  blocker for `osoleve/Qwen3.5-4B-Base-Text-NVFP4`

What changed for `osoleve` after the upgrade:

- before upgrade:
  - failed immediately because `transformers` did not recognize
    `qwen3_5_text`
- after upgrade:
  - model resolves as `Qwen3_5ForCausalLM`
  - `--language-model-only` is accepted by the CLI
  - but bring-up still fails later inside `vLLM`'s multimodal Qwen 3.5 path
    with a config-type mismatch:

```text
Expected type: Qwen3_5Config
but found type: Qwen3_5TextConfig
```

Practical conclusion:

- upgrading to `transformers 5.3.0` was necessary and helped
- but `osoleve` is still blocked by a `vLLM` integration issue around
  `Qwen3_5TextConfig`, not by plain `transformers` age anymore

Updated practical ranking after the `olka-fi` / `osoleve` pass:

1. `llama.cpp Q4_K_M`
   - `38.21 tok/s`
   - fastest valid result so far
2. `vLLM Axion NVFP4`
   - `38.45 tok/s`
   - nominally fastest, but invalid due to corrupted output
3. `llama.cpp Q8_0`
   - `35.96 tok/s`
4. `vLLM olka-fi MXFP4`
   - `33.82 tok/s`
   - valid, but effectively in the same band as `AWQ Marlin`
5. `vLLM AWQ Marlin`
   - `33.15 tok/s`

### Axion `4B NVFP4` Recheck After `transformers 5.3.0`

- rechecked `AxionML/Qwen3.5-4B-NVFP4` in bare-metal `vLLM` after the
  `transformers 5.3.0` upgrade
- the model still boots successfully with:
  - `--quantization modelopt_fp4`
  - native `NvFp4LinearBackend.FLASHINFER_CUTLASS`
- model load still looks healthy:
  - `Qwen3_5ForConditionalGeneration`
  - about `3.85 GiB` model memory

Default mode sanity check:

- prompt:
  - `Reply with exactly: ok`
- result:

```text
mutmut styr尔ulanmut profilrijmutrijdatamutuyeuyeaptmut
```

Eager mode sanity check:

- command:
  - `PORT=2239 ./bare.spark/run-vllm.sh AxionML/Qwen3.5-4B-NVFP4 --quantization modelopt_fp4 --enforce-eager`
- prompt:
  - `Reply with exactly: ok`
- result:

```text
mut树皮达尔mutmutmutmutmutmutmut九大mutmutmutmut
```

Practical conclusion:

- upgrading to `transformers 5.3.0` does not fix the corrupted-decoding issue
  for Axion `4B NVFP4`
- `--enforce-eager` also does not fix it
- current status:
  - model boots
  - NVFP4 kernels initialize
  - output is still invalid, so the result remains non-benchmarkable

## TensorRT-LLM Added

- added a Docker harness in [`trtllm.spark/`](./trtllm.spark)
- main launcher:
  - [`trtllm.spark/run-docker.sh`](./trtllm.spark/run-docker.sh)
- default image:
  - `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6`
- default config file:
  - [`trtllm.spark/extra-llm-api-config.yml`](./trtllm.spark/extra-llm-api-config.yml)
  - passed by `--extra_llm_api_options`
  - enables iteration perf stats for the PyTorch backend metrics path

Current bring-up assumptions:

- use the official `trtllm-serve serve ...` OpenAI-compatible server
- start with Docker instead of bare metal
- reuse the existing `bare.spark` model resolver for short names like `0.8b`,
  `4b`, and `9b`

Practical notes from the official docs:

- TensorRT-LLM explicitly documents Blackwell support and `Qwen3`
- the public support docs do **not** explicitly list `Qwen 3.5`
- recent docs also expose `--reasoning_parser qwen3`, `--max_seq_len`,
  `--kv_cache_free_gpu_memory_fraction`, and `--tp_size`
- NVIDIA's install/release docs still make the NGC container the safer path on
  SBSA / DGX Spark than a fresh bare-metal PyTorch workflow

Status:

- harness added
- first NGC image pull started
- no TensorRT-LLM benchmark numbers recorded yet

### TensorRT-LLM `0.8B` First Bring-Up Attempt

- first live run used:
  - `./trtllm.spark/run-docker.sh 0.8b`
- the large `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6` image pull
  completed enough to reach actual model bring-up
- first real failure was **not** model load itself; it was our optional
  `--extra_llm_api_options` file:

```text
ValueError: LLM got invalid argument: pytorch_backend_config
```

Practical consequence:

- disable the default extra YAML config for now
- retry the plain server path before drawing any conclusion about
  `Qwen/Qwen3.5-0.8B` support

### TensorRT-LLM `0.8B` Plain Retry

- plain retry got past the extra-config issue and reached actual model startup
- next real blocker in the stock `1.3.0rc6` image:

```text
The checkpoint you are trying to load has model type `qwen3_5`
but Transformers does not recognize this architecture.
```

- the container itself reports:
  - `transformers 4.57.1`
- practical next move:
  - build a tiny local derivative image from the official TensorRT-LLM base
  - upgrade only `transformers`, but stay in a TensorRT-LLM-friendly range
  - try `4.57.6` first, because our `vLLM` tests already proved that version
    recognizes `Qwen 3.5`
  - retry `0.8b`, then continue with `4b` and `9b` if the upgraded image works

### TensorRT-LLM `0.8B` Compatibility Attempts

We tried three official-base image variants for `Qwen/Qwen3.5-0.8B`:

1. stock `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6`
   - reports `transformers 4.57.1`
   - fails because `qwen3_5` is not recognized

2. local derivative with `transformers 4.57.6`
   - still fails inside TensorRT-LLM worker init with the same
     `qwen3_5` config-recognition error
   - practical finding:
     `4.57.6` was not enough inside TensorRT-LLM's own runtime path

3. local derivative with `transformers 5.3.0`
   - `qwen3_5` becomes recognizable
   - but the stock TensorRT-LLM package imports legacy symbols that no longer
     exist in `transformers 5.x`
   - first blocker:

```text
ImportError: cannot import name 'AutoModelForVision2Seq'
```

   - after patching that compatibility point in a local derivative image, the
     next blocker appears immediately:

```text
ImportError: cannot import name 'get_parameter_device'
```

Practical conclusion for this pass:

- `TensorRT-LLM 1.3.0rc6` is blocked for `Qwen 3.5` on this machine with the
  official release line
- the blocker is not just one missing flag; it is a broader compatibility gap
  between current TensorRT-LLM imports and `transformers` versions that know
  `qwen3_5`
- because `0.8b` never reached ready state, `4b` and `9b` were **not**
  benchmarked

### TensorRT-LLM Full Source Build from `main`

Goal: build TensorRT-LLM from latest `main` with `transformers==5.3.0`
compatibility patches, compiled natively for Blackwell `120-real`.

#### Build Harness

- Dockerfile: `trtllm.spark/Dockerfile.main-source`
- Build script: `trtllm.spark/build-main-image.sh`
- Command:
  ```bash
  FULL_SOURCE_BUILD=1 CUDA_ARCHITECTURES=120-real JOB_COUNT=4 ./trtllm.spark/build-main-image.sh
  ```
- Target image tag:
  `local/trtllm-main:main-transformers--5.3.0-sourcebuild-120-real`

#### `transformers 5.x` Compatibility Patches

The Dockerfile applies a Python patch step (step 9/11) that fixes ~12
compatibility issues between TRT-LLM `main` and `transformers 5.3.0`:

- `AutoModelForVision2Seq` → `AutoModelForImageTextToText` fallback
  (`gpt/convert.py`, `tools/multimodal_builder.py`)
- `get_parameter_device` / `get_parameter_dtype` removed from
  `transformers.modeling_utils` — replaced with local shims
  (`modeling_clip.py`, `modeling_siglip.py`, `transformer_wan.py`)
- `load_sharded_checkpoint` removed — replaced with fallback impl
  (`modeling_llama.py`)
- `AutoConfig.register` now raises `ValueError` on duplicate registration
  in `transformers 5.x` — wrapped in try/except
  (`modeling_exaone4.py`, `modeling_exaone_moe.py`, `modeling_nemotron_h.py`,
  `modeling_vila.py`, `modeling_qwen3_5_moe.py`)
- `suffix_automaton` native binding optional import
  (`suffix_automaton.py`)
- `record_global_timer` optional import with fallback
  (`autotuner.py`)
- `requirements.txt` pinned to `transformers==5.3.0`

#### Build Attempt 1: Precompiled (no source build)

- image: `local/trtllm-main:main-transformers--5.3.0`
- built successfully in ~30 minutes
- reused the precompiled CUDA binaries from the base `1.3.0rc6` image
- not yet tested at runtime — the precompiled binaries may not have
  Blackwell (`sm_120`) kernels

#### Build Attempt 2: Full Source Build (killed)

- launched: `FULL_SOURCE_BUILD=1 CUDA_ARCHITECTURES=120-real JOB_COUNT=4`
- ran for ~1.5 hours before being killed (Codex process termination)
- was actively compiling CUDA kernels at the time of kill
- no image produced

#### Build Attempt 3: Full Source Build (hash mismatch failure)

- restarted the same command after killing Codex
- docker buildx cache reused all steps up to the `build_wheel.py` stage
- CUDA kernel compilation progressed normally to **66%** (~2 hours of nvcc)
- failed at the `deep_ep` (DeepEP) module with an nvshmem hash mismatch:

```text
CMake Error at nvshmem_project-stamp/verify-nvshmem_project.cmake:29 (message):
  error: SHA256 hash of
    /opt/TensorRT-LLM/cpp/tensorrt_llm/deep_ep/nvshmem_src_3.2.5-1.txz
  does not match expected value
    expected: 'eb2c8fb3b7084c2db86bd9fd905387909f1dfd483e7b45f7b3c3d5fcf5374b5a'
      actual: 'd0284e8894c2d8d4555f60889f8461e5278587fb51908391a16eee084a9ee3e8'
```

- root cause: the Dockerfile uses `GIT_LFS_SKIP_SMUDGE=1` and only pulls
  LFS files for `internal_cutlass_kernels/**` — the `deep_ep/nvshmem_src_3.2.5-1.txz`
  file remains as a Git LFS pointer, so the hash check fails
- the actual nvcc compilation error was at 63% but gmake kept running
  unfinished parallel jobs until 66% before propagating the failure
- no recent commits on TRT-LLM `main` address this issue

#### Build Attempt 4: Disable DeepEP

- fix: pass `-DBUILD_DEEP_EP=OFF` to avoid the nvshmem download entirely
- rationale: DeepEP is for multi-node EP (expert parallelism) via NVSHMEM,
  not needed for single-GPU inference on Spark
- if this still fails, fallback plan is to add `deep_ep/nvshmem_src*` to
  the LFS pull include pattern

#### Build Attempt 4 Result: New LFS Failure at 77%

- disabling DeepEP (`-DBUILD_DEEP_EP=OFF`) successfully got past the 63%
  nvshmem hash mismatch
- however, a new failure appeared at 77%:

```text
/opt/TensorRT-LLM/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/cubin/xqa_kernel_cubin.cpp:1:1:
error: 'version' does not name a type
    1 | version https://git-lfs.github.com/spec/v1
```

- same root cause: `xqa_kernel_cubin.cpp` is a Git LFS pointer file, not
  actual C++ source
- the selective LFS pull (`--include="cpp/tensorrt_llm/kernels/internal_cutlass_kernels/**"`)
  missed this and other cubin files
- note: `BUILD_DEEP_EP=OFF` was passed via `-D` but the cmake configure line
  shows it was accepted (`"-DBUILD_DEEP_EP=OFF"` present in the cmake args)

#### Build Attempt 5: Full LFS Pull + DeepEP Disabled

- fix: changed `git lfs pull --include="..."` to `git lfs pull` (pull ALL
  LFS-tracked files)
- this ensures all cubin files, nvshmem archives, and any other LFS-tracked
  build inputs are real files, not pointers
- trade-off: larger download during the clone step, but eliminates the
  whack-a-mole of discovering which files are LFS-tracked
- keeping `-DBUILD_DEEP_EP=OFF` since we still don't need multi-node NVSHMEM

#### Build Attempt 5 Result: deep_ep Target Reference Failure

- full LFS pull fixed BOTH previous issues:
  - `xqa_kernel_cubin.cpp` compiled successfully (no longer an LFS pointer)
  - no nvshmem hash mismatch (file is real now)
- CUDA compilation reached 76%+ with no kernel build errors
- failed at the very end with:

```text
gmake[3]: *** No rule to make target 'deep_ep', needed by 'CMakeFiles/build_wheel_targets'.  Stop.
```

- root cause: `build_wheel.py` hardcodes `deep_ep` in the
  `BUILD_WHEEL_TARGETS` list (line 623), and our `-D BUILD_DEEP_EP=OFF`
  disabled the cmake target but didn't remove it from the targets list
- the cmake configure line shows both `"-DBUILD_DEEP_EP=OFF"` and
  `WHEEL_TARGETS="...;deep_ep;..."` — contradictory
- since full LFS pull fixed the nvshmem hash issue, we no longer need
  `-DBUILD_DEEP_EP=OFF` at all

#### Build Attempt 6: Full LFS Pull, No DeepEP Override

- removed `-D "BUILD_DEEP_EP=OFF"` from the Dockerfile
- kept the full `git lfs pull` (no `--include` filter)
- this lets `build_wheel.py` use its default targets including `deep_ep`,
  and the nvshmem archive is a real file thanks to full LFS pull

#### Build Attempt 6 Result: Compilation SUCCESS, Wheel Packaging Failure

- **all CUDA kernels compiled successfully** — reached `[100%] Built target build_wheel_targets`
- failed in the wheel packaging step afterward:

```text
pip download tensorrt_llm==None --dest=/tmp/... --extra-index-url=https://pypi.nvidia.com
distutils.errors.DistutilsSetupError: Failed to download the automatically resolved wheel
```

- root cause: `--depth 1` shallow clone has no git tags, so the build
  script resolves the version as `None` and then tries to download
  `tensorrt_llm==None` from PyPI, which obviously doesn't exist
- the wheel build is not needed for the Docker image — we just need the
  editable install (`pip install -e .`) which happens separately via
  `--install`

#### Build Attempt 7: Skip Wheel Build

- added `--skip_building_wheel` to `build_wheel.py` invocation
- this skips the broken `python -m build ... --wheel` step
- the `--install` flag still runs `pip install -e .[devel]` after
  compilation, which is all we need for the Docker image

#### Build Attempt 7 Result: Editable Install Version Failure

- compilation: SUCCESS (100%)
- wheel build: SKIPPED (as intended)
- failed on the `--install` step which runs `pip install -e .[devel]`:

```text
ERROR: Invalid requirement: 'tensorrt_llm==None'
distutils.errors.DistutilsSetupError: Failed to download the automatically resolved wheel
```

- same version-is-None issue as attempt 6, just triggered by the editable
  install path instead of the wheel build path
- root cause: setup.py resolves version from git tags, shallow clone has
  no tags, version becomes `None`

#### Build Attempt 8: No --install, Manual pip install --no-deps

- removed `--install` from `build_wheel.py` to avoid its broken
  `pip install -e .[devel]` which triggers version resolution
- instead, after `build_wheel.py` completes, run:
  `TRTLLM_USE_PRECOMPILED=1 python3 -m pip install --no-deps -e .`
- `TRTLLM_USE_PRECOMPILED=1` tells setup.py to use the already-compiled
  build artifacts instead of trying to download a wheel
- `--no-deps` avoids any dependency resolution that might fail
