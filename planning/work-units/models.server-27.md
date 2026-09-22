# Huihui Cinference benchmark on Smarty

Work item: https://github.com/kortexa-ai/models.server/issues/27

Measure the pinned non-Swift Huihui Qwen3.8-27B NVFP4 artifact on the RTX PRO
6000 at its normal 450 W power limit. Preserve the published 256K/K8V4/MTP-10
profile and compare a 512K shared-pool, eight-slot, INT8 KV, MTP-3, vision-enabled
profile with stock Qwen3.8 in models.server. Report actual prompt sizes, prefill,
decode, first-token latency, process VRAM, and total-card VRAM separately.

Use a short correctness canary before increasing context. Cold long-context
requests cover approximately 8K, 32K, 128K, and 260K tokens; a larger request
checks the configured 512K ceiling when supported. Include recall and prose
workloads so speculative acceptance on repetitive text is visible.

Franci authorized temporary production service downtime. Pin the 6000 UUID,
retain 10 GiB free, use the process-owned GPU block restoration trap, and leave
4090 services alone. Resolve the production 27B from the live service snapshot;
the historical block helper needs its Qwen entry mapped to Bonsai when Bonsai
is the running service. Restore and health-check exactly the stopped set.

Keep external runtime/model files under ignored `.engines/cinference`, raw
records under ignored `bench-results/cinference-27`, and curated results and
reproduction code under `bench/cinference-6000`. Git is the transfer mechanism
for tracked scripts and results. Delivery includes focused commits, push, and
matching local/Smarty revisions; it does not replace a production model.
