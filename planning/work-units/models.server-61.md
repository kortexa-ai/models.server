# GLM 1M context and latency experiment

Owning issue: https://github.com/kortexa-ai/models.server/issues/61

This follows the completed pinned-main comparison in models.server #60.
Preserve both the production recipe and the 512K DCP2 recipe as recorded in
[BASELINES.md](../../glm-5.3-flash-2x-dgx-experimental-1m/BASELINES.md).
The separate [1M plan](../../glm-5.3-flash-2x-dgx-experimental-1m/PLAN.md) measures
cold prefill, prefix reuse and sustained decode before any promotion decision.
Client defaults and public routing are outside this work unit.
