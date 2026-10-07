# Strata GB10 reproduction documentation

Issue: https://github.com/kortexa-ai/models.server/issues/45
Evidence: https://github.com/kortexa-ai/models.server/issues/44

Add a README-only `qwen-3.8-flash-next-strata` model-zoo entry and an inventory
link. Record the pinned ARM port, ggml, model and draft revisions; isolated
build/download/packing commands; tested launch configuration; grouped-MTP
workaround; measured results and context/image limitations. Keep TensorFold on
Static as the selected recipe. No runtime or managed-service change is part of
this work.

Validation passed: all four shell blocks parse with `bash -n`; embedded Python
and the smoke-request JSON parse; the generated config matches the saved
four-slot 262K configuration apart from the local log path; all four checkpoint
hashes and source/model/draft pins match the evidence; local Markdown links
resolve; and `git diff --check` passes. No model was restarted or downloaded.
