# Qwen3.8 Flash Next Fast

This is a single-Spark, text-only TensorFold experiment. It uses the pinned
[MiaAI Lab TensorFold recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/4cd99569f00f84b397cee77900bec9cdf43287bd)
and the `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` checkpoint. It serves on
`static` at port 2067, so the existing two-Spark NVFP4 service remains
independently addressable on port 2066.

The recipe defaults to five concurrent 262,144-token windows, int8 KV cache,
PLE tables on SSD, six MTP drafts at 0.60 confidence, and thinking enabled.
Its expected startup budget is tight: about 102.6 GiB of 103.64 GiB, with
roughly 9.7 GiB free. Check the upstream recipe's memory notes before changing
the stream count or context.

TensorFold v0.3.6.2 does not accept image, audio, or video requests. This model
is therefore advertised as text-only. The upstream implementation explicitly
rejects media in
[`messages.py`](https://github.com/ashhart/TensorFold/blob/v0.3.6.2/src/tensorfold/server/messages.py).

The existing NVIDIA NVFP4 checkpoint is not interchangeable with the MLX
affine 4-bit checkpoint. TensorFold v0.3.6.2's Qwen3.8 family supports MLX
affine 4-bit/group-32 and EXL3 formats; its quantization guide says NVFP is
unsupported. Keep the NVFP4 service config and weights for rollback.

Install and run this model with `ktxsvc install qwen-3.8-flash-next-fast` and
`ktxsvc start qwen-3.8-flash-next-fast` on `static`. The runner checks the
recipe checkout against the pinned commit before launch.

To switch back, stop and uninstall the fast service, then install the original
pair service on `shock` followed by `static`:

```sh
ktxsvc stop qwen-3.8-flash-next-fast
ktxsvc uninstall qwen-3.8-flash-next-fast
# On shock, then on static:
ktxsvc install qwen-3.8-flash-next
```

Uninstalling a service removes its systemd registration. It leaves the model
config and weights in place.
