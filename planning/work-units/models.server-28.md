# Qwen3.8 fast model registration

Work item: https://github.com/kortexa-ai/models.server/issues/28

Register vanilla and Huihui models with pinned NVFP4 artifacts and the tested
Cinference runtime. Both use DFlash2-7, K8V4, 512K shared KV, eight slots,
262K per request, 8192-token prefill chunks, and vision on the RTX PRO 6000.
Prepare and smoke-test launchers without installing services or changing the default.
Validate artifact provenance/checksums, GPU pinning, API aliases, tool and image
requests, model inventory, and normal service restoration. The companion runbook
work is https://github.com/kortexa-ai/legolm/issues/143; benchmark evidence is #27.
