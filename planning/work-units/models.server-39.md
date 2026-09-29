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
- The checkpoint was copied directly from Smarty to `static` with resumable
  `rsync`, then mirrored from `static` to `shock` over RoCE. Both file-name and
  size manifests match the source. The direct copy sustained about 40–50 MB/s;
  the RoCE mirror sustained about 350 MB/s.
- The vLLM runner enables long context and prefix caching for the Qwen MTP
  draft config. The final runner is committed on `main`; both Sparks are
  synced to the same revision and their `ktxsvc` services are active.
- `GET http://192.168.2.101:2066/health` returns 200. `static` is rank 0/API;
  `shock` is rank 1 over RoCE.
- At idle after startup, `free -h` reported about 24 GiB available on `static`
  and 26 GiB on `shock`. vLLM reported 74.27 GiB of weights plus non-Torch
  memory, 1.3 GiB peak activation, 0.29 GiB CUDA graph, and 9.61 GiB KV cache
  per rank at the configured 0.70 GPU-memory target.
- Near-maximum-context test: 998,800 prompt tokens plus 128 output tokens;
  494.603 s time to first token (2,019.4 prompt tokens/s), 68.66 output
  tokens/s, and 496.467 s total. This is a synthetic repeated-text stress
  prompt; it validates throughput and the long-context path, not retrieval
  quality. A separate 32,767-token prompt plus 64 output tokens took 11.146 s
  to first token (2,939.72 prompt tokens/s) and decoded at 42.63 tokens/s.
- The model is usable for shorter contexts; a full 1M-token request has an
  approximately eight-minute first-token delay at this configuration.
