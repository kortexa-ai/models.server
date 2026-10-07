# Scrappy Windows Hermes provider parity

Issue: https://github.com/kortexa-ai/models.server/issues/49
Parent: https://github.com/kortexa-ai/models.server/issues/48

Scrappy has one native Windows Hermes profile. Its installation is
`C:\src\hermes-agent`, its virtual environment is
`C:\src\hermes-agent\venv`, and its configuration is
`C:\Users\francip\AppData\Local\hermes\config.yaml`. SSH enters WSL2;
the Windows Hermes home must be inspected separately.

Match the non-authentication provider definitions and capability overrides
to Snappy's default profile. Snappy's Mira profile has the same definitions.
The `kortexa.ai` provider contains seven models: Qwen 3.8 27B, Bonsai 2 27B,
Qwen 3.8 Flash Next Fast, Claude Fable 5, GPT 5.6 Luna, GPT 5.6 Sol, and
GLM 5.3 Flash EXL3. The `kortexa-static` provider contains only
`qwen-3.8-flash-next-fast` at `http://192.168.2.101:2067/v1`.

Copy exact model IDs, context limits, vision/tool/reasoning overrides, provider
URLs, default entries and explicit discovery settings. Preserve Scrappy's
working public-provider credential, selected main model, legacy provider,
auxiliary assignments and unrelated private settings. Do not create a Mira
profile or copy OAuth stores. Configured model entries can remain inactive;
this does not start model services.

Back up the exact native configuration under its Hermes home at
`provider-backups/snappy-provider-parity-20261007T154707.045597Z/` before a
guarded atomic write. Parsed readback matches Snappy for both providers and
their overrides. Restoring only the affected sections reproduces the original
configuration. A repeated preview produces no changes.

Validate fresh native Windows Hermes calls through the public and Static
providers with the requested model and provider in fresh usage records.
The public canary returns `GLM_OK` from `glm-5.3-flash-exl3`; the direct Static
canary returns `STATIC_OK` from `qwen-3.8-flash-next-fast`. Each uses one API
call and reports provider `custom`, with its intended URL resolved before
generation. Both catalogs return HTTP 200. The public catalog omits the
currently inactive `qwen-3.8-27b`, which remains configured as on Snappy.
Use the existing concrete CPython 3.11.15 executable and installed venv
dependencies for the remote launch; the uv junction limitation remains.
No Windows Hermes process was running during this update.
The primary-model guide records the installation, single-profile scope,
provider rosters and launch recipe. Documentation synchronization on Smarty
requires no model or API restart.

Sanitized configuration receipts, parity checks and native request evidence
are retained on Snappy at
`~/Desktop/ai reports/scrappy-hermes-provider-parity-20261007/`.
