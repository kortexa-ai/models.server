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

## Runtime compatibility files

The serving container is pinned to the official vLLM Qwen3.8 image's ARM64
manifest, `vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`. The
GB10/SM121 QSA path and this checkpoint's MTP metadata need five compatibility
overlays. The recipe pins their Apache-2.0 source repository to commit
`6ad1c8f15cbab1ababd2048e8e5f94094dbfc4a0`:

```sh
git clone https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark.git \
  ~/src/vendor/qwen38-flash-next-dgx-spark
git -C ~/src/vendor/qwen38-flash-next-dgx-spark checkout --detach \
  6ad1c8f15cbab1ababd2048e8e5f94094dbfc4a0
```

The launcher verifies this exact commit before it starts. It mounts only the
overlays listed in `model.json`; it does not execute the upstream launcher.

## Install and start

On both nodes, fast-forward `~/src/models.server` through Git to the same
`origin/main` commit, stage the local checkpoint copy, and clone the pinned
compatibility repository above. Pull the pinned Docker image:

```sh
docker pull vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e
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
