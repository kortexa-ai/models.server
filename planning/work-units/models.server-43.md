# Public harness providers

Work unit: https://github.com/kortexa-ai/models.server/issues/43
Proxy dependency: https://github.com/kortexa-ai/api.server/issues/102

Installed Hermes profiles, OMP and pi use `kortexa.ai` at the public API for
Kortexa models, with `kortexa-static` as a selectable direct Flash fallback.
The old Qwen 3.8 27B definition remains selectable. Bonsai 2 27B and Flash Next
Fast are included. LFM provider entries are removed and their auxiliary-role
selections move to Bonsai. Main defaults use Codex `gpt-6-luna` with each
harness's native OAuth provider spelling.

Private configuration is backed up on the owning machine. Provider credentials
travel only through the protected process/SSH channels and private files.
Repository files use Git. Acceptance requires fresh model requests and native
provider/default resolution on every installed harness; absent installations
are recorded without creating empty configuration directories.
