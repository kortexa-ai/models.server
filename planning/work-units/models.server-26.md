# Standalone Parakeet Redux

Work order: https://github.com/kortexa-ai/models.server/issues/26

Add the publisher's ternary Parakeet Redux checkpoint as a standalone Photon
model on port 2063, with CPU serving on macOS and Linux. Keep runtime packages
in `parakeet-redux/.venv/`. The launcher uses the standard model configuration
and service lifecycle. Disable the installer's automatic startup and crash
restart settings before leaving the model available on demand.

Expose file transcription, model metadata, health, and optional timestamps.
Validate request errors, file cleanup, lifecycle, configuration dispatch, and
actual audio transcription. Synchronize through Git to Smarty, install the
on-demand definition, verify an actual request, then stop Redux. Leave the
existing Qwen/Nemotron ASR service unchanged.

API catalog compatibility belongs to api.server#94: honor the declared model
type so an unproxied standalone ASR model is not advertised as a chat model.
