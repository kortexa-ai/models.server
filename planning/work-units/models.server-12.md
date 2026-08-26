# Issue #12: Replace Gemma 4 E2B with LFM2.5 8B-A1B

Parent issue: https://github.com/kortexa-ai/models.server/issues/12

## Decision

Promote the existing Q8_0 LFM2.5 8B-A1B configuration with one 128K q8_0 KV
slot to the RTX 4090. Retire Gemma 4 E2B only after LFM remains healthy beside
the resident ASR and TTS services.

## Validation record

- Pending live fit gate and downstream consumer inventory.
