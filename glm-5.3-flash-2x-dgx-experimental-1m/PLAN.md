# 1M context and latency experiment

Owning issue: https://github.com/kortexa-ai/models.server/issues/61
Prior pinned-main comparison: models.server #60.

1. Preserve the two exact recipes in BASELINES.md. Verify their manifests,
   runtime fingerprints, images, setup receipts and stopped containers. Keep
   this test in its own manifest and managed service on port 2072.
2. Reuse the pinned image and DCP2 implementation. Change the request ceiling
   to 1048576 and the cache budget to 6 GiB/rank. Keep all other serving knobs
   fixed. Replay the observed allocator on CPU before using the GPUs.
3. Inspect both services, GPU ownership, ports and outstanding requests. Stop
   the active API, then worker. Set up both hosts and start the new worker,
   then API. Keep only the chosen recipe enabled at boot.
4. Pass short streaming and tool canaries. Measure cold and immediate warm
   requests at 2K, 128K, near 512K, 768K and near 1M. Reserve output space below
   the context ceiling. Request 1024 output tokens for sustained decode timing.
   Forced output lengths are throughput probes, not repetition-quality tests.
5. Record client time to first token and server prefill/decode durations
   separately. Report actual input/output counts, uncached input rate, prefix
   hits, queue time, preemptions, DFlash acceptance and both hosts' memory.
   Attribute server metric deltas only when exactly one request completed.
   A cold case requires zero prefix hits; a warm case must prove reuse.
6. Stop the exact owned services if memory approaches exhaustion or a request
   stops making progress. Do not weaken retrieval, stream or token-accounting
   checks to pass a larger context. Preserve logs for any failed attempt.
7. Pass eight-image and tool checks after the long workload. If 1M cannot serve
   reliably, restore and verify the 512K experimental recipe. A working but
   slow result is useful evidence; promotion and client routing remain separate.

Store raw measurements outside Git under `~/Desktop/ai reports/`. Keep the
issue and immutable evidence current, without embedding live status here.
