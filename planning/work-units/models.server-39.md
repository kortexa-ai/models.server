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
- Pin a vLLM nightly ARM64 image by digest in `qwen-3.8-flash-next/model.json`.
- Use stock Qwen4Exp runtime modules; older compatibility overlays fail import
  against the current image and are not mounted.
- Keep the service available on the private LAN only through `static`'s
  configured interfaces. Leave other model services alone.

## Validation and delivery

- Config and shell validation passed; the service unit passed
  `systemd-analyze verify` on both new machines.
- Repo changes are committed and pushed on `main`.
- Both Sparks have the same clean repo revision, RoCE links, and local image
  tag. The tag on each host resolves to image ID
  `sha256:29339ec3eddf9131b1fbea8e49fb88dd7021755beaa0a17669b2efcff901ad0a`.
- The source ARM64 vLLM image was pulled by manifest digest on `static` and
  mirrored to `shock` over RoCE.
- Checkpoint transfer from Smarty to `static` is still running; afterward the
  deployment watcher will verify the files, mirror to `shock` over RoCE, start
  both services, and run a near-maximum-context speed test.
- Pending: successful two-node startup, measured maximum-context request and
  speed, and service health verification.
