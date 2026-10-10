# GLM TP2 512K context and concurrency

Work order: https://github.com/kortexa-ai/models.server/issues/59

Raise the pinned recipe to a 524288-token served window and 6 GiB FP8 KV per
rank. Keep eight normal sequences. Add the recipe-local DCP upgrade plan.

Verify manifest/lifecycle checks, sync through Git, run explicit setup on both
Sparks and restart the managed pair in dependency order. Test a near-window
request and eight distinct 64K histories, including generation and lookahead
room. A bounded sixteen-sequence/32K experiment is optional if the current
scheduler and memory allow it without runtime changes; restore the normal
eight-sequence setting after measurement.

Record exact commits, cache capacity, peak active requests, memory, timing,
correctness, service start times and rollback evidence in the issue. Preserve
the 896ebb98a8f30bb93d2a6996c5839d608c71f6ad configuration and containers as the
initial rollback target. The future DCP upgrade is documentation only here.
