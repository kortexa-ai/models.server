# Parallel Shingi production deployment

Work order: https://github.com/kortexa-ai/models.server/issues/50
Engine: https://github.com/kortexa-ai/shingi-27b/issues/2
Application comparison: https://github.com/kortexa-ai/mappity/issues/3

Pin the validated public package, retain the model, calibration and Prism revision,
and build through setup-shingi.sh. Four sequences share one model allocation and
one 16K-token context pool. Raise the pre-load floor to 11,536 MiB based on the
9,769 MiB measured peak; retain the 1,024 MiB serving floor for colocated voice services.

Validate config, launcher and build behavior against the real package before the
Shingi-only stop/start. Keep the previous package and models.server commit available
for rollback. Verify the installed package stamp, native binary, four-slot metadata,
concurrent decision results and all existing 4090 service health after restart.
