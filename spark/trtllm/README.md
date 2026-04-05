# TensorRT-LLM Spark

Docker harness for trying NVIDIA TensorRT-LLM on this DGX Spark machine with
the same short model-key workflow we already use for `vLLM` and `SGLang`.

This path intentionally starts with the official NGC container rather than a
bare-metal install. NVIDIA's current docs position the container as the clean
quick-start path for Blackwell, and recent release notes still call out SBSA
bare-metal issues for the PyTorch workflow.

## Default Image

```bash
nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6
```

## Start a Server

```bash
./trtllm.spark/run-docker.sh 0.8b
./trtllm.spark/run-docker.sh 4b
./trtllm.spark/run-docker.sh Qwen/Qwen3.5-4B
```

If the stock image does not recognize `qwen3_5`, build the local image with a
newer `transformers` once and then reuse it:

```bash
./trtllm.spark/build-image.sh
IMAGE=local/trtllm-qwen35:transformers-4.57.6 ./trtllm.spark/run-docker.sh 0.8b
```

If you want the latest TensorRT-LLM Python code from upstream `main` while
still reusing the NGC runtime binaries, build the source-overlay image:

```bash
./trtllm.spark/build-main-image.sh
IMAGE=local/trtllm-main:main-transformers--5.3.0 BACKEND=_autodeploy ./trtllm.spark/run-docker.sh 0.8b
```

For the full native source build on this machine, the helper now defaults to a
safer `JOB_COUNT=4`. You can still override it if you want:

```bash
FULL_SOURCE_BUILD=1 ./trtllm.spark/build-main-image.sh
FULL_SOURCE_BUILD=1 JOB_COUNT=6 ./trtllm.spark/build-main-image.sh
```

Useful overrides:

```bash
PORT=2251 ./trtllm.spark/run-docker.sh 4b
KV_CACHE_FREE_GPU_MEMORY_FRACTION=0.8 ./trtllm.spark/run-docker.sh 4b
REASONING_PARSER=qwen3 ./trtllm.spark/run-docker.sh Qwen/Qwen3.5-4B
TRUST_REMOTE_CODE=1 ./trtllm.spark/run-docker.sh some/custom-model
```

## Notes

- The launcher uses `trtllm-serve serve ...`, which exposes an OpenAI-compatible
  API on `/v1/chat/completions`.
- The default backend is `pytorch`, because that is TensorRT-LLM's current
  default for `trtllm-serve` and is the lowest-friction way to try direct HF
  checkpoints.
- A small YAML file is included for future metrics tuning, but it is **not**
  enabled by default because the current `1.3.0rc6` PyTorch path rejected the
  `pytorch_backend_config` key during live bring-up on this machine.
- The known preset keys reuse `bare.spark/models.json`, so the short names stay
  aligned with the rest of the repo.
- TensorRT-LLM's public support matrix explicitly lists `Qwen3`, but not
  `Qwen 3.5`, so `Qwen 3.5` should be treated as an experiment until we prove a
  real server bring-up on this machine.
- The stock `1.3.0rc6` image currently reports `transformers 4.57.1`, which was
  too old to recognize `qwen3_5` during our first live `Qwen 3.5` attempt.
- The current local derivative target is `transformers 4.57.6`, because it
  recognized `Qwen 3.5` in our `vLLM` bring-up while still staying in the
  `4.57.x` family that TensorRT-LLM expects.
- There is now a separate `main` source-overlay build path for trying upstream
  TensorRT-LLM Python changes without waiting for a newer NGC release tag.
