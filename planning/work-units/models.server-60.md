# GLM pinned-main Spark experiment

Owning issue: https://github.com/kortexa-ai/models.server/issues/60

The completed capacity baseline is models.server #59. The experiment has its
own recipe and [plan](../../glm-5.3-flash-2x-dgx-experimental/PLAN.md).
It reuses the verified weights while preserving the stable recipe as a
reproducible rollback. The new recipe is the working version; promotion is a
later decision, after speed, cache sharding and correctness measurements.

Keep manifests, build inputs and benchmark programs in Git. Keep timing logs,
private OMP replay payloads, live status and raw results in the issue and
external audit directory. No harness default change is part of this work.
