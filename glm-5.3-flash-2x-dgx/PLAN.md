# GLM TP2 context and vLLM upgrade plan

## Capacity on the pinned runtime

Use a 524288-token input-plus-output limit, 6 GiB FP8 KV per rank and eight
shared sequences. Keep the registered weights, runtime, DFlash2, image budget,
batch budget, RoCE interface and model port 2070. Record validation and delivery
in [models.server #59](https://github.com/kortexa-ai/models.server/issues/59).

The context ceiling and shared KV pool are separate limits. Confirm the pool
capacity from the engine's startup log, but do not divide its token estimate
by a smaller context size to predict concurrency. Hybrid per-request state
and block allocation can limit admission first. Test a request close to the full window
and eight independent requests near 64K each. Measure generation, prefill,
preemption and host memory on both ranks. Count output and draft lookahead in
the budget. Prefix sharing can improve capacity, but must not be required for
these tests.

A bounded sixteen-sequence experiment can use about 32K per request to keep
the total history comparable. First test short histories with enough output
to overlap decode, and verify that cache capacity permits the intended active
count. Check scheduler policy, draft lengths, CUDA graph coverage and memory.
Skip a scheduler-limit increase if cache admission is already the bottleneck.
Do not infer sixteen active sequences from
sixteen submitted clients. Record peak RUNNING/WAITING, total throughput,
per-request throughput and memory. Retain eight as the normal setting unless
a later decision adopts sixteen.

## Later upgrade: shard KV with DCP

The image is based on vLLM `487ecf187d3dfe74d2cf6119a92881dba403c219`
(2026-08-25), plus the pinned Spark overlays. TP2 partitions model weights;
the main MLA KV is replicated across Static and Shock. DCP can distribute
token history across the same ranks. Small recurrent and tail state remains
replicated. Savings must be measured for the complete hybrid cache layout.

Upstream changes to evaluate:

- [#59211](https://github.com/vllm-project/vllm/pull/59211), merged 2026-10-05:
  GLM kpool indexer DCP support, including replicated tail handling and global
  top-k merging. Its tests used H100, BF16 KV and no speculative decoding.
- [#60032](https://github.com/vllm-project/vllm/pull/60032), merged 2026-10-08:
  warm up the kpool DCP top-k merge kernels before serving.
- [#57169](https://github.com/vllm-project/vllm/pull/57169), merged 2026-10-09:
  generic packed KV layout and circular tail buffers. Review its effect on
  the recipe's source overlays and speculative ring handling.
- [#54305](https://github.com/vllm-project/vllm/issues/54305) and
  [#54472](https://github.com/vllm-project/vllm/pull/54472): direct DCP A2A
  output-layout handling. Recheck the fix and backend selection when building;
  the report documents a generic A2A fallback.

Release v0.31.0 predates the GLM DCP merge. Select and pin a reviewed later
commit or a tested backport. Do not update the running container in place.
An upgrade needs a new immutable image and compatible Spark overlays.

## Upgrade experiment

1. Inventory source mounts, Python hooks, native kernels, mixed-weight loading,
   DFlash2 scheduling and RoCEnante integration. Check which patches are already
   upstream before porting the remainder. Run the recipe's source-drift and CPU
   tests against the selected vLLM source.
2. Build one ARM64/GB10 image. Transfer it directly between the Sparks and pin
   the same digest on both. Reuse the existing verified weights.
3. Establish TP2/DCP1 correctness and performance on the new image first.
   Then test TP2/DCP2 with `--decode-context-parallel-size 2` and
   `--cp-kv-cache-interleave-size 4`. Verify FP8 KV, DFlash2 and the selected
   sparse MLA backend together on GB10.
4. Measure actual KV capacity, per-group block demand, recurrent/speculative
   state overhead, host headroom, cold/warm prefill, decode rate, prefix-cache
   reuse and concurrent requests. Test retrieval at several
   depths, tool/result continuation with empty output, streamed tool calls,
   eight-image history, and long-context speculative decoding.
5. Replay the saved OMP conversation as a bounded regression probe. Record
   the reconstructed prompt and tool schemas: it is not an exact replay when
   the original system prompt or schemas are unavailable. Check repeated calls
   and malformed tool output as well as HTTP success.
6. Select a larger served window only after the measured capacity and accuracy
   gates pass. Sharding the main cache does not guarantee twice the total
   capacity or unchanged speed; cross-rank communication has a cost.

## Deployment and recovery

Keep each experiment in Git with its complete manifest and image pins. Sync
both clean checkouts, run explicit setup, stop Static then Shock with `ktxsvc`,
and start Shock then Static. Preserve the preceding image, source checkouts,
weights and stopped containers. Use a focused Git revert and the same managed
lifecycle if a gate fails. Verify the restored endpoint and a real request.
Keep benchmark artifacts and changing execution status in the owning issue,
not this durable plan.
