# models.server #39 — Qwen3.8-Flash-Next on the two DGX Sparks

Issue: <https://github.com/kortexa-ai/models.server/issues/39>

## Decisions

- Use NVIDIA's NVFP4 checkpoint already stored in Smarty's model vault. Stage
  an independent local copy on each Spark because the PLE table needs local
  random reads.
- Serve with vLLM TP=2 over the private RoCE network. `static` is rank 0 and
  serves port 2066; `shock` is rank 1.
- Use the published YaRN settings to extend the 262,144-token native context
  to 1,000,000 tokens. Start with six sequences, MTP-3, FP8 KV, and 0.70 GPU
  memory utilization.
- Pin the vLLM container image and the Apache-2.0 compatibility overlay
  repository revision in `qwen-3.8-flash-next/model.json`.
- Keep the service available on the private LAN only through `static`'s
  configured interfaces. Leave other model services alone.

## Validation and delivery

- Pending: config and shell validation, remote checkout synchronization,
  checkpoint transfer, container startup, maximum-context request, speed
  measurement, commit/push, and per-node service verification.
