# Bounded VRAM cache

Owner: https://github.com/kortexa-ai/models.server/issues/52

Pin and deploy only the verified public engine release. Keep production memory floors explicit, account for cache capacity before serving, compare the production binary to the tested artifact, and verify all affected service health. The user authorizes Shingi downtime and control of both GPUs.

Develop Shingi on a local branch. Validate CPU tests and frozen GPU controls, then publish one consolidated runtime improvement on public main. Keep intermediate branches local. Store full reports under `~/Desktop/ai reports/shingi-vram-cache-2026-10-08/`.
