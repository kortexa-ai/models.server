# GLM-5.3 Flash EXL3 harness registration

Issue: https://github.com/kortexa-ai/models.server/issues/47
Public discovery dependency: https://github.com/kortexa-ai/api.server/issues/104

Add `glm-5.3-flash-exl3` to the existing `kortexa.ai` provider in installed
Hermes, OMP and pi.dev configurations on Snappy, Smarty and Scrappy.
Private configuration stays outside Git; exact files are backed up on their
owning machines before atomic writes, with concurrent-change checks.

Snappy has Hermes default and Mira profiles, OMP and pi.dev. Smarty and
Scrappy's WSL account have OMP and pi.dev. Hermes is absent on those two
Linux accounts. Scrappy also has native Windows Hermes at
`C:\src\hermes-agent`, with `HERMES_HOME` set to
`C:\Users\francip\AppData\Local\hermes`. Its registration and remote
launch recipe are recorded in
https://github.com/kortexa-ai/models.server/issues/48 and
`planning/work-units/models.server-48.md`.

The entry advertises the validated 262144-token served window, text input,
reasoning and ordinary tools. OMP and Pi use Chat Completions, a 16384-token
output ceiling and the explicit chat-template thinking switch. Registration
preserves provider credentials, existing models, defaults and role choices.
Semantic readback confirms that only the new model and Hermes capability
override change. A repeated preview produces no changes.

Smarty's model manifests synchronize through Git. API discovery includes
Shock through the separately owned dependency. Harness loading and fresh
public inference readbacks are recorded in the issues. Sanitized evidence
and the bounded private-file registration script are retained on Snappy at
`~/Desktop/ai reports/glm53-harness-registration-20261007/`.
