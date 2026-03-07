# DGX Spark vLLM Notes

This folder contains the experimental DGX Spark `vLLM` path for Qwen 3.5:

- Dockerfile based on `nvcr.io/nvidia/vllm:26.02-py3`
- build/run scripts
- a Docker-oriented systemd unit
- the `vLLM vs llama-server` benchmark harness plus raw logs

For the current single-user setup, this path is still experimental. The main
recommendation and the latest benchmark summary live in
[`../qwen-3.5-27b/QWEN_SPARK.md`](../qwen-3.5-27b/QWEN_SPARK.md).

## Build

```bash
./docker-build.sh
```

## Run

```bash
./run-vllm-docker.sh
```

The container serves an OpenAI-compatible API on `http://0.0.0.0:2026/v1`.
Model weights are not baked into the image:

- `~/.cache/huggingface` is mounted into the container so model repos are reused
- `~/.cache/vllm` is mounted so `torch.compile` artifacts survive rebuilds and
  restarts

## Useful Overrides

```bash
# Refresh the NVIDIA base image and rebuild
./docker-build.sh --pull

# Lower memory usage / shorter context
MAX_MODEL_LEN=32768 ./run-vllm-docker.sh

# Text-only mode (skips the vision encoder and frees memory)
LANGUAGE_MODEL_ONLY=1 ./run-vllm-docker.sh

# Use the 35B A3B MoE model with the same image
PORT=2027 \
MODEL_REPO=Qwen/Qwen3.5-35B-A3B \
SERVED_MODEL_NAME=unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M \
LANGUAGE_MODEL_ONLY=1 \
./run-vllm-docker.sh
```

## Files

- `Dockerfile.vllm-ngc`
- `docker-build.sh`
- `run-vllm-docker.sh`
- `benchmark_vllm_vs_llama.py`
- `bench-results-small-models.json`
- `bench-logs/`

## Notes

- Do not upgrade NVIDIA drivers on DGX Spark for this setup.
- The default `SERVED_MODEL_NAME` intentionally matches the legacy Linux value
  from `model.json`, so existing API routing keeps working.

---

© 2025 kortexa.ai
