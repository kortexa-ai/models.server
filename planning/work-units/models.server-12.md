# Issue #12: Replace Gemma 4 E2B with LFM2.5 8B-A1B

Parent issue: https://github.com/kortexa-ai/models.server/issues/12

## Decision

Promote the existing Q8_0 LFM2.5 8B-A1B configuration with one 128K q8_0 KV
slot to the RTX 4090. Retire Gemma 4 E2B only after LFM remains healthy beside
the resident ASR and TTS services.

## Validation record

- The live one-slot 128K service holds 10,208 MiB on the RTX 4090. Together
  with ASR and TTS the GPU uses 19,994 MiB and leaves 2,563 MiB free; all three
  health endpoints remain HTTP 200 and a real completion decoded at 231 tok/s.
- Gemma was then stopped, disabled, and uninstalled, releasing its 6000
  allocation.
- The API Radio fallback, Radio dependency documentation, 6000 borrowing
  inventory, and active OMP, Pi, Prime Agent, and Hermes host configurations
  now select LFM2.5 8B-A1B. Live canaries passed on Smarty and Snappy; Scrappy
  OMP also passed. Scrappy's native Hermes YAML validates, but its existing
  uv-managed Python 3.11 base is missing, so that launcher cannot run a canary.
- Final production health checks passed with LFM, ASR, and TTS resident on the
  4090 and the remaining production stack resident on the 6000.
