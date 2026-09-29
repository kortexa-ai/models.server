# Qwen3.8-Flash-Next NVFP4 on two DGX Sparks

This recipe serves NVIDIA's `nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint with
vLLM tensor parallelism across `static` (rank 0) and `shock` (rank 1). The
service listens on `static:2066`. The model supports 262,144 tokens natively;
this config enables the recipe's YaRN extension to 1,000,000 tokens. It uses
MTP-3 speculative decoding and FP8 KV cache.

The checkpoint is about 124 GB per node. Keep a local copy on each Spark at
`~/models/Qwen3.8-Flash-Next-NVFP4`; the PLE (n-gram embedding) table must be
local for its random reads. The canonical source copy is in Smarty's model
vault at `~/storage/models/vault/huggingface/nvidia/Qwen3.8-Flash-Next-NVFP4`.

## Runtime

The serving container uses the current vLLM nightly's ARM64 manifest,
`sha256:c3bede3517c5c982e217c785fd10a8b8ef26e1641c169f5a8634c5b7aa71ca03`.
Its image ID is pinned in `model.json` and checked before launch. Stock Qwen4Exp,
QSA, PLE, and ModelOpt modules import cleanly on a DGX Spark. Older overlays
from earlier Qwen recipes are not mounted because they are incompatible with
this current runtime.

## Install and start

On both nodes, fast-forward `~/src/models.server` through Git to the same
`origin/main` commit and stage the local checkpoint copy. Pull the pinned
Docker image:

```sh
docker pull vllm/vllm-openai@sha256:c3bede3517c5c982e217c785fd10a8b8ef26e1641c169f5a8634c5b7aa71ca03
docker tag vllm/vllm-openai@sha256:c3bede3517c5c982e217c785fd10a8b8ef26e1641c169f5a8634c5b7aa71ca03 \
  vllm/vllm-openai:qwen38fn-arm64-c3bede35
```

Install and start the managed service on `shock` first, then on `static`:

```sh
# On shock (rank 1 / worker)
ktxsvc install qwen-3.8-flash-next

# On static (rank 0 / API head)
ktxsvc install qwen-3.8-flash-next
```

The model config sets `max_model_len` to 1,000,000 with the documented YaRN
scaling settings. This limit counts input plus generated tokens. Check service
logs with `ktxsvc status qwen-3.8-flash-next`.
The OpenAI-compatible endpoint is `http://192.168.2.101:2066/v1`.

## Benchmarking

For a maximum-context test, use a prompt close to 999,000 tokens and leave at
least 1,000 tokens for the response. Record the server-reported prompt and
completion token counts, time to first token, and decode rate separately.
Always include context length and whether thinking is enabled with a speed
result. The default config disables thinking to match the published speed
lane; clients may enable it through Qwen's chat-template options.

The original two-Spark speed lane reported roughly 53.7 tokens/s median over
40 shorter prompts with this checkpoint, MTP-3, six sequences, a 262,144-token
configured context, and thinking disabled. It does not measure the 1M YaRN
extension. Treat it as a reference, not a prediction for this pair or a
999K-token prompt. Long prefill is a different workload.
