# Scrappy native Windows Hermes registration

Issue: https://github.com/kortexa-ai/models.server/issues/48
Parent: https://github.com/kortexa-ai/models.server/issues/47

Scrappy has native Windows Hermes at `C:\src\hermes-agent`. Its configured
`HERMES_HOME` is `C:\Users\francip\AppData\Local\hermes`, rather than the
default `.hermes` directory. The Windows virtual environment and its installed
dependencies remain in `C:\src\hermes-agent\venv`.

Add `glm-5.3-flash-exl3` under the canonical `kortexa.ai` provider, with the
served 262144-token context, text input, reasoning and ordinary tools. Preserve
the legacy `kortexa-ai` provider, its model entries and the global Qwen default.
The legacy environment credential returns HTTP 401. The new provider uses the
working credential already configured in Scrappy's WSL Pi provider, with an
authenticated catalog readback. The Windows environment and legacy provider's
credential reference stay unchanged. Private credentials remain outside Git.

Exact configuration backups are inside the native Hermes home at
`provider-backups/glm53-windows-20261007T143218.240559Z/` and
`provider-backups/glm53-windows-auth-20261007T144710.679765Z/`. Writes are atomic
and guarded against concurrent edits. Removing the new provider and capability
override from the parsed readback reproduces the original configuration.

The uv executable launcher embeds the generic CPython 3.11 junction path.
The concrete CPython 3.11.15 executable runs, while the junction cannot be
traversed from this remote Windows session. This matches the reported
Windows remote-session issue:
https://github.com/NousResearch/hermes-agent/issues/61557

Invoke the existing installation through the concrete native Python executable
for remote validation:

```powershell
$env:PYTHONPATH = 'C:\src\hermes-agent;C:\src\hermes-agent\venv\Lib\site-packages'
$env:HERMES_HOME = 'C:\Users\francip\AppData\Local\hermes'
& 'C:\Users\francip\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe' `
  -m hermes_cli.main --provider custom:kortexa.ai --model glm-5.3-flash-exl3 `
  -t todo --ignore-rules -z 'Reply exactly with GLM_OK.'
```

This process uses the installed Windows source and virtual-environment
dependencies. It does not require a package reinstall or a Windows policy
change. A `pyvenv.cfg` pointer change alone did not fix the embedded launcher
path and was restored byte-for-byte. The remote launcher limitation remains.

Accept the echo canary only when its fresh usage record names
`glm-5.3-flash-exl3` and provider `custom`; an answer after fallback does not
verify GLM. The validated invocation returns `GLM_OK` in one API call, with
that exact model and provider in the usage record. Arithmetic canaries require
Hermes's terminal tool, so use an echo
when deliberately limiting available tools. Sanitized registration, readback,
launcher and client evidence is retained on Snappy under
`~/Desktop/ai reports/glm53-harness-registration-20261007/`.
