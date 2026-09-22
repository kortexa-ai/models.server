# Cinference harness tool qualification

Issue: https://github.com/kortexa-ai/models.server/issues/31

## Scope

Qualify ordinary tool loops in Hermes, OMP and pi.dev before switching the main
model. Preserve caller-side argument validation and distinguish automatic tool
selection from guarantees the runtime does not implement. Both registered
aliases need real harness tests; a direct weather-tool canary alone is not
qualification. Coordinate a later GPU window through the 6000 operating guide.

The initial documentation change links the API adapter in api.server#95 and
separates engine limitations into models.server#32 (strict schemas/JSON) and
#33 (required/named/single-call enforcement). It changes no harness settings,
model defaults, services or GPU allocations. The API's CPU protocol fixtures
prove proxy behavior, not model generation through the deployed gateway.

## Acceptance

Record installed client/runtime revisions, exact non-secret compatibility
settings, streaming calls and tool-result replay, invalid-argument repair,
multiple calls, reasoning on/off, cancellation and long-context continuation.
Mark absent harness installations explicitly. Do not silently remove a
guarantee from a workflow that requires it; document the unsupported boundary.
Update PRIMARY_MODEL_SWITCH.md with verified client results.
