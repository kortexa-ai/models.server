# Shock harness provider

Owner: https://github.com/kortexa-ai/models.server/issues/53

Add `kortexa-shock` as a manually selectable local provider for
`glm-5.3-flash-exl3` at `http://192.168.2.102:2069/v1`. Preserve every selected
default, auxiliary role, credential, and unrelated setting. Keep the existing
`kortexa.ai` GLM registrations at the served 262144-token context limit.

Scope: Snappy Hermes default and Mira, OMP and pi; Smarty OMP and pi; Scrappy
native Windows Hermes and WSL2 OMP and pi. Smarty has no Hermes installation.
Back up exact affected private files on their owning hosts and compare all
pre-existing configuration values after each write. Match the existing Static
provider's authentication convention and GLM's public model metadata.

Validation used fresh installed-client requests for all nine configurations
through both providers: 18 checks, with a read-tool call and continuation on
both routes from each machine's pi client. Hermes checks captured the actual
request URL and the served model in addition to the usage record, including
the concrete native Windows interpreter on Scrappy.

Every pre-existing configuration value was preserved. OMP/pi settings files
were not edited, and post-request checks compared file hashes with the applied
baseline. Backups are under each Unix home's
`.local/share/kortexa/provider-migration/shock-20261009T015407.081293Z/`;
native Windows Hermes uses its own home under `provider-backups/` with the
same timestamp. Secret-free evidence is in
`~/Desktop/ai reports/shock-harness-providers-20261008/verification.json` on Snappy.

The installed Snappy gateway model picker reads the profile config on each
invocation; both profile loaders resolved the new provider with the original
default intact. This provider-only addition required no gateway or model
server restart.
