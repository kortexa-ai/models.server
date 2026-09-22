# Primary model switching guide

Work item: https://github.com/kortexa-ai/models.server/issues/29

Document the canonical rollout and rollback checklist in PRIMARY_MODEL_SWITCH.md.
Read actual model/provider configuration on Snappy, Smarty, Scrappy/WSL2 and
Moodymoose/Pi 140. Include explicit model pins, auxiliary/profile settings,
API catalog/routing/radio/Realtime, Hermes-LiveKit, remote desktop clients, and
both phone configurations. Preserve credentials and selected cloud roles.
Validate paths/field names against current local code and sanitized config
projections. This work does not change a primary/default selection or take the
6000 back from the acknowledged Shingi handoff. The listener-address correction
found during inventory is tracked separately in models.server#30.
