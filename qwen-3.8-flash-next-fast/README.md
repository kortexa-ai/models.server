# Qwen3.8 Flash Next Fast

This is the single-Spark TensorFold recipe on `static`, served at port 2067.
It uses [MiaAI Lab's recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e)
pinned to commit `856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e`, with TensorFold
v0.3.6.3 and the `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` checkpoint.

This revision includes MiaAI's CUDA vision-tower patch and supports image and
video input. Send image/video content as OpenAI-style `image_url` or `video_url`
parts using data URLs. Remote URL fetching is disabled. Audio input is not
supported. Vision is enabled by the service config.

The recipe defaults to five concurrent 262,144-token windows, int8 KV cache,
PLE tables on SSD, six MTP drafts at 0.60 confidence, and thinking enabled.
The vision-enabled recipe uses 2,048-row prompt chunks to reserve scratch space;
its published long-prefill runs are a few percent slower than text-only mode.
Check upstream's memory notes before changing stream count or context.

The existing two-Spark NVFP4 service remains independently addressable on port
2066. TensorFold cannot load that NVFP4 checkpoint; this service uses the
separate MLX 4-bit checkpoint.

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
