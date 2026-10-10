# GLM-5.3 Flash NVFP4 + DFlash2 on two DGX Sparks

Static is rank 0 and serves `glm-5.3-flash-2x-dgx` at
`http://192.168.2.101:2070/v1`. Shock is rank 1 and has no separate public API.
Port 2070 follows the per-model sequence in the root inventory. Port 29669 is
the private distributed bootstrap port on the direct RoCE network.

This recipe pins Tony's TP2 launcher at `d061f26` and Knapcio's runtime/profile
at `770d115`. `model.json` records full revisions, exact image ID, source
checkpoint revisions, complete served artifact hashes, node GPU UUIDs and all
local profile overrides. Routed experts retain NVIDIA NVFP4. Selected dense
weights use the upstream `lossless8` mixed FP8 conversion; the name does not
mean mathematically lossless. The FP8 DFlash2 drafter is a local derivative.

The served window is **262144 tokens**, including output, from a checkpoint
trained for 1048576. Eight sequences share 4 GiB FP8 KV per rank. This is not
eight independent 262K caches. Batch budget is **6919**, including draft
lookahead; the working trial inherited this value from the pinned profile.
All seven draft-length graph families remain available. Default effort is low;
clients can request another supported effort. Vision is limited to two images,
with a 1 GiB processor cache. Each container has a 112 GiB memory limit and no
extra swap allowance. Startup requires 108 GiB available host memory.

## Explicit setup on each node

Setup uses Git for both source checkouts, verifies every serving artifact with
SHA-256 and records file identity for fast startup checks. It never starts a
service. Run it again after a manifest or launcher change.

The validated October 9 artifacts already exist on both hosts. Adopt them
through local hardlinks (no network transfer or extra weight copy):

```bash
./scripts/setup-vllm-glm53-tp2.sh glm-5.3-flash-2x-dgx \
  --adopt-target /home/francip/models/glm53-tp2-20261009/glm-quant-mix/lossless8 \
  --adopt-draft /home/francip/models/glm53-tp2-20261009/draft-fp8
./run.sh glm-5.3-flash-2x-dgx --dry-run
```

On a clean machine, prepare the pinned image and artifact first. Build
`Dockerfile.roce` from the pinned Knapcio checkout with the manifest's base
image, and transfer that exact saved image directly between Sparks. A rebuild
can have a different image ID and requires explicit recipe validation and a
pin update. Download the exact `source_model` and `source_draft` revisions with
`hf download`; use the pinned `scripts/build_lossless8.sh` and
`scripts/drafter_fp8.py` CPU conversions, then supply their output directories
to setup. Setup rejects outputs that differ from the registered hashes. Keep
source weights in the vault; do not relay large transfers through Snappy.
Do not redistribute the converted drafter; retain its upstream license.

## Managed pair lifecycle

Inspect `ktxsvc list`, container ownership, memory and both GPUs first. Do not
run any other model on either Spark alongside this pair. Stop the manual trial
before installation. Install the worker first, then the API rank:

```bash
ssh shock 'ktxsvc install models/glm-5.3-flash-2x-dgx'
ssh static 'ktxsvc install models/glm-5.3-flash-2x-dgx'
```

`ktxsvc install` enables and starts the unit. The same unit selects its rank
from the hostname and pins that host's GPU UUID. The attached Docker process
keeps systemd tied to the actual rank. Services retry after failure, including
a delayed peer boot. Neither service starts or stops processes over SSH.
For maintenance, stop Static then Shock; start Shock then Static. Always
coordinate both ranks: a single surviving rank cannot serve requests.

```bash
ssh static 'ktxsvc stop models/glm-5.3-flash-2x-dgx'
ssh shock 'ktxsvc stop models/glm-5.3-flash-2x-dgx'
ssh shock 'ktxsvc start models/glm-5.3-flash-2x-dgx'
ssh static 'ktxsvc start models/glm-5.3-flash-2x-dgx'
curl -f http://192.168.2.101:2070/health
curl -f http://192.168.2.101:2070/v1/models
```

Source mounts are revision-specific, immutable Git checkouts. Runtime invokes
only the upstream command builder; it never invokes upstream source rsync or
cluster stop/start. Containers carry model, rank and recipe labels, survive a
stop for inspection, and can restart when their fingerprint matches. A changed
recipe creates a new container only after the old GPU workload stops. No
unrelated container is removed. Startup never downloads, converts or builds.

The public API discovers the managed service after this repository is synced
to Smarty; allow its catalog cache to refresh. Clients use
`kortexa.ai/glm-5.3-flash-2x-dgx` or explicitly select `kortexa-dual` at the
direct endpoint. The fallback is a selectable provider, not an automatic retry
chain. Both routes depend on the same two-node model.

## Recovery and validation limits

Uninstall the old Static `qwen-3.8-flash-next-fast` and Shock
`glm-5.3-flash-exl3` service registrations after this replacement is prepared.
Their recipes and weights remain available. To revert the cutover, stop both
new services and start the preserved trial worker `glm53-tp2-exp-20261009-r1`
on Shock, then `glm53-tp2-exp-20261009-r0` on Static. That experiment serves
`glm-5.3-flash` on port 28053 and does not match the new client route; restore
the saved private client configs when using that rollback.

The original trial passed direct streaming, tool/result continuation, empty
tool output, two concurrent requests and 27K prompt retrieval. It did not prove
eight-request saturation, full-window reliability or freedom from agent loops.
The recorded OMP failure can be reconstructed from its JSONL conversation, but
its exact system prompt and tool schemas were not saved. A bounded continuation
is useful evidence, not a replay of the complete task. See issue #57 for the
registered-service checks and replay result.
