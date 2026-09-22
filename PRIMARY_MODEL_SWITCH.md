# Switching the primary model on Smarty

This is the canonical checklist for changing the **local primary language
model** served by Smarty. Audit date: **2026-09-22**. Recheck live state at each
switch; a configured provider, a running model, and a client's selected default
are three separate settings.

This guide does not authorize a switch. The fast Qwen models are registered and
tested, but are not installed or enabled as services. Bonsai remains the
installed production 27B at the audit date. Coordinate the release of any
active experiment before a later rollout.

## 1. Record the target and rollback before changing anything

Use one change record with these values. Do not replace strings across whole
home directories: that can alter fallback roles, old transcripts, or credentials.

| Field | Example: vanilla fast Qwen |
|---|---|
| Public model ID and backend alias | `qwen-3.8-27b-fast` |
| Managed service | `models/qwen-3.8-27b-fast` |
| Model manifest | `qwen-3.8-27b-fast/model.json` |
| Direct OpenAI base URL | `http://192.168.2.3:2064/v1` |
| Health URL | `http://192.168.2.3:2064/health` |
| Public gateway base URL | `https://api.kortexa.ai/v1` (unchanged) |
| Gateway provider | `kortexa-ai` |
| Per-request context, including output | `262144` tokens |
| Shared KV pool / admitted slots | `524288` tokens / `8` slots |
| Normal reasoning | explicit `medium` for coding; `none` for non-thinking voice |
| Backend default output allowance | `32768` tokens; clients can set a smaller bound |
| GPU | RTX PRO 6000 UUID `GPU-a71210ca-e14a-755a-88bb-77f53a2102f6` |
| Expected model-process memory | about `39.8 GiB`, including DFlash2 and vision |
| Rollback | record the actual old model ID, service, port, Git SHA, and config backups |

Candidate ports: stock Qwen `2053`, Bonsai `2062`, vanilla fast Qwen `2064`,
abliterated fast Qwen `2065`. **Smarty is `192.168.2.3`; Snappy is
`192.168.2.6`.** Do not copy the older stock manifest's Snappy host metadata into
a Linux-only listener. Fast manifests now use Smarty's verified address.

Choose whether the change updates the local `kortexa-primary` role only or also
changes selected harness defaults. A Hermes or OMP session using a cloud model
must not be silently moved to the local model. Preserve the independent small
fallback at `2059` and vision-language helper at `2055` unless they are explicitly
part of the change.

Back up only affected private configuration files with restrictive permissions.
Retain each machine's own credentials, profile identity and unrelated settings.
Tracked code/configuration moves through Git. Private home-directory configs
need separate, targeted edits on each machine; they are not synchronized by a
`models.server` pull. Never commit `.env`, auth files, or whole private configs.

## 2. Four-machine inventory

All paths below are relative to the named account's home. Inspect any active
`HERMES_HOME`, profile, command-line override, and project-level configuration
before assuming these defaults are authoritative. Check private `.env` endpoint
overrides by key and selected non-secret values; never dump their full contents.

| Machine / account | Hermes | OMP | pi.dev |
|---|---|---|---|
| Snappy, `/Users/francip` | `.hermes/config.yaml` and `.hermes/profiles/mira/config.yaml` | `.omp/agent/models.yml`, `.omp/agent/config.yml` | `.pi/agent/models.json`, `.pi/agent/settings.json` |
| Smarty, `/home/francip` | `.hermes/config.yaml` exists, but no model/provider sections in the audit; no profile configs found | same OMP paths | same pi paths |
| Scrappy, WSL2 `/home/francip` | `.hermes/config.yaml` exists, but no model/provider sections in the audit; no profile configs found | same OMP paths | same pi paths |
| Moodymoose / Pi 140, `/home/pi` | no local Hermes config/profile provider files found | no local OMP provider/settings files found | no local pi provider/settings files found |

Scrappy is reached with `ssh scrappy` at `192.168.2.5`; the audited environment
is WSL2. No matching native-Windows configs were found under `/mnt/c/Users/*`
in this audit. Check again if a Windows-native harness is installed later.
Moodymoose is `ssh moodymoose`, `pi@192.168.2.140`. Its Hermes Desk client uses
`~/.config/Hermes Desk/settings.json`, with both `realtimeBaseUrl` and
`conferenceBaseUrl` set to `https://api.kortexa.ai`. It has no local model pin
in the audited settings. Update the selected remote profile or API default,
then reconnect the client. Do not create empty local harness installations just
to make a checklist green; record absent installations as not applicable and
apply this guide if they are installed later.

Observed drift at audit time:

| Surface | Actual selection / route |
|---|---|
| Running 27B service | Bonsai on Smarty `2062` before the Shingi borrowing block |
| Hermes primary provider, Snappy default and Mira profiles | `kortexa-primary` still routes to stock Qwen `2053`; auxiliary tasks explicitly pin `qwen-3.8-27b` |
| Hermes main model, both Snappy profiles | cloud `openai-codex/gpt-6-astra` |
| OMP, Snappy and Smarty | direct LAN `kortexa-primary` at `2053`, plus a separate `kortexa-bonsai` at `2062` |
| OMP, Scrappy | `kortexa-ai` provider at `https://api.kortexa.ai/v1`, with several explicit model IDs |
| OMP selected default, all three audited installations | cloud `openai-codex/gpt-5.6-luna`; the tiny role is the separate VL helper |
| pi.dev, all three audited installations | `kortexa-primary` at `2053`; selected default `qwen-3.8-27b` |
| API Realtime, Smarty | allowlist `lfm2.5-8b-a1b,bonsai-2-27b`; default `lfm2.5-8b-a1b` |
| API Realtime config, Snappy | allowlist/default only `lfm2.5-8b-a1b`; API is not installed/running there in this audit |
| Phone / Switchboard, Snappy | explicit `bonsai-2-27b`, voice `mira`, through the Kortexa voice engine |

Thus changing the running service alone does not update all consumers. This
dated table records findings, not desired values to copy over newer settings.

## 3. Prepare the server without selecting it as default

1. Track the rollout in a `models.server` issue and claim the affected GPU,
   service, ports and repository paths. Coordinate any active experiment. Follow
   [the 6000 operating guide](https://github.com/kortexa-ai/legolm/blob/main/SMARTY_6000_GUIDE.md) for authorized
   downtime, a live baseline, deliberate headroom and exact restoration.
2. Review the target manifest, artifact checksum, runtime revision, model alias,
   per-request context, cache type, concurrency, vision and speculative backend.
   For these fast models, run `./scripts/setup-cinference.sh <model-id>` on
   Smarty. Setup downloads/verifies weights; it does not install a service.
3. Synchronize reviewed Git commits to Smarty and run the relevant checks.
   Confirm the target port is free and the manifest's bind address belongs to
   Smarty. Retain the old model and runtime for rollback.
4. Record `ktxsvc list`, process-to-GPU UUIDs and health. Stop only the authorized
   service set needed for the budget, using `ktxsvc`. Do not touch the 4090.
5. For the actual scheduled rollout, use `ktxsvc install models/<new-id>`
   if the unit is absent. **On Smarty, install also enables and starts it.**
   This is a cutover action, not preparation. For an already installed unit,
   `ktxsvc start models/<new-id>` performs a stop/start in the current tool.
   Avoid an unnecessary second start after installation. Verify all three
   installed/enabled/running states. Finalize boot policy after acceptance:
   disable the replaced model and keep only the intended production model
   enabled. `enable`/`disable` change boot policy without starting/stopping;
   a failed rollout must restore both running and enabled state. `ktxsvc`
   has no restart verb; use its documented stop/start commands when needed.
6. Prove health, the exact `/v1/models` alias, streaming, image input, a tool call
   plus tool-result continuation, cancellation, and reasoning on/off. Run a
   small canary before a long-context request. Watch actual free VRAM.

The fast launchers require at least 50 GiB free before loading. Their approximately
40 GiB footprint is for one process, not for the whole card. Eight slots share
512K KV: eight simultaneous 130K contexts do **not** fit. Requests reserve input
plus output and can queue. Do not increase the pool or concurrency without a
new memory budget and measurement. The 6000 remains capped at 450 W.

## 4. Hermes provider, explicit pins, and profiles

On each installed Hermes profile, inspect its effective configuration rather
than only the default profile. Snappy currently runs the default gateway and
`mira` as separate profile services.

Update these fields together where they refer to the local primary:

- `providers.kortexa-primary.api`: the direct base URL ending in `/v1`.
- `providers.kortexa-primary.default_model` and the model-ID key under
  `providers.kortexa-primary.models`; set `context_length` to the per-request
  window, not the shared cache allocation. Preserve `discover_models: false`
  for the explicit pinned list unless discovery is deliberately adopted.
- The model-ID key under `model_overrides.custom:kortexa-primary`, including
  `context_window`, vision, tools and reasoning capabilities.
- Every `auxiliary.<role>.model` that explicitly names the old primary, and any
  explicit `base_url`. In the audited profiles these include `vision`,
  `skills_hub`, `mcp`, `session_search`, and `flush_memories`.
- `model.default`, `model.provider`, `model.base_url`, and `model.api_mode`
  **only if** that profile should use the new model as its main model. Keep
  provider/model/URL/API mode consistent; a custom Chat Completions endpoint
  must not inherit a stale Codex Responses URL.
- `fallback_model`, `fallback_providers`, `smart_model_routing.cheap_model`,
  `moa`, delegation settings, cron jobs and saved session overrides: inspect
  explicit pins and change only those intended to follow this local role.

A provider URL change does not rewrite an auxiliary task's model ID. Conversely,
a model-ID change with the old port still sends requests to the wrong backend.
For a single-model direct endpoint, keep only its served ID in that provider.
Retain the previous model as a separate named provider if it must remain
selectable for rollback.

Preserve independent auxiliary assignments, including the VL helper, small
approval model, cloud compaction model, and the reflex service at `2051`.
Do not rewrite caches, auth files or historical session records by search/replace.
If context metadata appears stale, inspect Hermes's `context_length_cache.yaml`
and provider discovery cache through its normal tooling; explicit current
model metadata and a fresh request are the acceptance evidence.

Reload the CLI selection or start a fresh CLI session. For the affected running
gateways on Snappy, use the profile-aware Hermes manager, not raw launchctl:

```bash
hermes gateway restart
hermes --profile mira gateway restart
hermes gateway status
hermes --profile mira gateway status
```

Run only the commands for profiles actually affected, after draining active
turns/calls. These gateways are installed by Hermes and are not entries in
`ktxsvc list`; do not guess a Kortexa service name for them. Prove the new PID,
profile home, effective provider/model, and a real auxiliary/tool request.
Existing sessions may have an explicit `/model ... --session` selection;
check or reset those individually rather than assuming a gateway restart
changes their saved model.

## 5. OMP on each installed machine

Provider definitions are in `~/.omp/agent/models.yml`. Selected roles are in
`~/.omp/agent/config.yml` under `modelRoles` (`default`, `tiny`, `advisor`, and
any additional roles). These are different responsibilities.

For Snappy/Smarty, update `providers.kortexa-primary.baseUrl` and its model
entry's `id`, `name`, `contextWindow`, `maxTokens`, input modalities, tools,
reasoning and compatibility fields. For Scrappy's public `kortexa-ai` provider,
keep `https://api.kortexa.ai/v1` and its existing authentication; add/update the
model ID in its list. The public gateway selects the downstream port.

Update `modelRoles.<role>` as `provider/model-id` only for roles meant to follow
the switch. Preserve cloud defaults and the small `tiny` role otherwise.
Inspect project configuration, launch flags and session model selections for
overrides. Reload/select the target in the model picker or start a fresh OMP
session, then verify the resolved endpoint and model. `models.db` and its WAL
are runtime state, not the provider configuration to edit manually.

For Cinference, set per-model `compat.supportsStrictMode: false`. Keep
`api: openai-completions` for the first rollout. The installed OMP client
supports that flag. Its Qwen effort handling (`supportsReasoningEffort` and
`qwenTemplateReasoningEffort`) must be checked with actual requests at the
selected reasoning levels. Do not assume hiding thinking in the UI disables it.

## 6. pi.dev on each installed machine

Update the `kortexa-primary` provider in `~/.pi/agent/models.json`: `baseUrl`,
model `id`/name, `contextWindow`, `maxTokens`, `reasoning`, input modalities and
`compat`. Preserve the existing authentication mechanism. The pi client needs
an auth entry or dummy value even for a keyless local server; do not remove it
as part of a model change.

If the startup selection should follow the switch, update
`~/.pi/agent/settings.json` together:

```json
{
  "defaultProvider": "kortexa-primary",
  "defaultModel": "qwen-3.8-27b-fast",
  "defaultThinkingLevel": "medium"
}
```

Merge these fields into the existing file. Preserve unrelated settings. Audit
`.pi/settings.json` in active projects, per-model `modelThinkingLevels`, launch
flags, extensions, and saved session selections. Project settings override
global settings. This installed pi version reloads `models.json` when `/model`
opens; select the new model and use Ctrl+S if saving the startup default.

For Cinference, use `api: openai-completions`,
`compat.supportsStrictMode: false`, `compat.maxTokensField: max_tokens`, and
supported reasoning controls. The installed client supports `thinkingFormat:
qwen-chat-template` to send explicit `enable_thinking` and `preserve_thinking`.
Validate an off/none request as well as a medium-effort request before adopting
that setting. Keep the existing conservative system/developer-role compatibility
unless a tested client payload justifies changing it.

Do not set `contextWindow` to 524288. The API limit is 262144 including output.
The existing pi configuration advertises 65536 output tokens; review that
allowance against the desired long-context budget rather than copying it
blindly. A 32768 maximum is consistent with the registered backend default;
short voice requests should use a smaller explicit output bound.

## 7. API catalog, routing, radio and Realtime

The authoritative implementation is in
[models.ts](https://github.com/kortexa-ai/api.server/blob/main/src/routes/models.ts),
[modelProtocols.ts](https://github.com/kortexa-ai/api.server/blob/main/src/routes/modelProtocols.ts),
[upstreams.ts](https://github.com/kortexa-ai/api.server/blob/main/src/config/upstreams.ts), and
[realtimeProviders.ts](https://github.com/kortexa-ai/api.server/blob/main/src/agent/realtimeProviders.ts).

- Sync `models.server` on Smarty so the API can read the new manifest. Discovery
  uses the control service's running-model inventory and a five-minute cache;
  registration alone does not make an inactive model available in `/v1/models`.
- Verify `KORTEXA_MODEL_HOST=192.168.2.3` and the intended
  `KORTEXA_LLM_HOSTS` / `KORTEXA_UPSTREAM_HOSTS`. The role-host override takes
  precedence over a manifest host. Do not redirect the whole upstream chain
  just to change one model's port.
- Check authenticated `/ops/models` and `/v1/models`, then an actual gateway
  request with the exact new model ID and, when needed,
  `X-Kortexa-Provider: kortexa-ai`. Normal chat requires a model ID; there is no
  single universal API “primary model” setting that replaces every caller pin.
- The gateway rewrites downstream IDs through `resolveHfModel`. Cinference
  expects its exact `--model-id`; its current manifests resolve to that ID.
  Do not add a Hugging Face repository ID as `hf_model` without checking this.
- **Cinference catalog gap before default rollout:** the current API learns
  tool/reasoning capabilities from llama.cpp `/props` and advertises extra
  protocols only for recognized engines. Cinference has no `/props`; current
  discovery will advertise chat/streaming and configured image input, but not
  its full tool/reasoning/Responses/Messages capabilities. Add and test engine
  support in `api.server` before clients depend on capability discovery. Merely
  adding capability fields to these manifests does not fix the current builder.
- Radio has its own `KORTEXA_RADIO_PRIMARY_MODEL` and
  `KORTEXA_RADIO_FALLBACK_MODEL`. They are unset in the audited production env;
  source defaults are stock Qwen and LFM2.5-8B-A1B. Set the primary explicitly
  if radio should follow the switch; preserve the intended fallback. Its retry
  circuit breaker can temporarily keep using a fallback after the target recovers.
- Direct Kortexa Realtime has a separate `KORTEXA_REALTIME_MODELS` allowlist and
  `KORTEXA_REALTIME_DEFAULT_MODEL`. Add the target to the allowlist even if it
  is only explicitly selectable. Change the default only if desired. The
  default must be in the allowlist. Keep the small-model voice default if the
  large-model switch is for coding only.
- Compare Snappy's standby API configuration with Smarty's active configuration
  deliberately. They already differ in the Realtime allowlist. Do not enable
  Snappy's API or overwrite it wholesale just to achieve textual parity.

Inspect the effective env file chain. `loadEnv()` currently loads
`.env.<mode>.local`, `.env.<mode>`, `.env.local`, then `.env`, all with
`override: true`: later general files win and can override process env. Review
only relevant keys; never print whole environment or credential files. See the
[environment audit](https://github.com/kortexa-ai/api.server/blob/main/docs/environment-audit.md).

For API code changes, follow its normal build/test/deploy workflow, sync through
Git, and stop/start the exact managed service with `ktxsvc`. For env-only changes,
restart the affected API service to avoid stale import-time values and clients.
A fresh catalog alone is not a successful inference test. Preserve API credentials,
public DNS, TLS, nginx, ASR and TTS endpoints unless they are independently changing.

## 8. Hermes-LiveKit, desktop clients and phone

`hermes-livekit` is a plugin in a Hermes gateway; it does not have a separate
universal local-model default. Its adapter resolves the active profile's model
configuration. On Snappy, inspect both default and Mira profile homes plus their
LiveKit/Realtime plugin settings. Both audited profiles currently select cloud
Astra, so switching the local provider only affects their explicitly assigned
local auxiliary calls until a profile/session model is deliberately changed.

Check session-only model overrides and reconnect calls after the profile change.
A Realtime protocol label such as `hermes` identifies the gateway and is not the
underlying Qwen model ID to replace. Keep profile identity, voice, room and
transport endpoints unless the rollout explicitly changes them. Verify a fresh
call, non-thinking behavior where intended, one tool round trip, interruption,
and the first audible response. LM TTFT alone excludes ASR/TTS and buffering.
The [Hermes-LiveKit readme](https://github.com/kortexa-ai/hermes-livekit/blob/main/README.md) documents no-microphone,
no-playback latency and tool probes; use bounded probes rather than real phone
calls without separate authorization.

Snappy's phone pipeline has two independent private selections:

- `~/.config/phone/config.toml`
- `~/.local/share/switchboard/phone.personal/config.toml`

Both use `[livekit].voice_engine = "kortexa"` and explicitly set
`realtime_model = "bonsai-2-27b"`, with voice `mira`. Update both model fields
if the phone should follow the new primary. Add that model to the API Realtime
allowlist first. Keep voice and private identity/credentials unchanged.
`phone-kortexa --json doctor` and
`switchboard-kortexa phone.doctor --ns phone.personal --json` are the documented
no-dial checks; acceptance still needs a bounded audio/tool test. The private
operating notes are `~/.local/share/phone-setup/README.md`.

Also inspect saved model selections in Hermes Desk, browser/mobile clients,
scripts, cron jobs and project launchers. Moodymoose's Hermes Desk is currently
a remote API client, so its important checks are the selected remote profile,
new-call behavior and model availability through the public API. Refresh/reselect
cached model lists after rollout. Never migrate old conversation history by
blind string replacement. Snappy's audited Codex/Claude default config files
had no local-primary references; only change such harnesses if an explicit
local provider/endpoint is actually configured later.

## 9. Acceptance and rollback

Use a per-machine completion matrix in the rollout issue. Mark an absent
installation as not applicable, with the audited account/path. Required evidence:

- Only the intended large-model service runs and boot enablement is deliberate;
  target health, exact model alias, GPU UUID, memory and headroom pass.
- Every affected installed Hermes profile, OMP and pi provider resolves the new
  ID at the intended direct or public endpoint. Selected defaults/roles change
  only where requested. An actual fresh request and tool-result continuation
  pass from each machine; stale provider entries are not treated as success.
- Image input and reasoning on/off pass. For Cinference, no client sends
  `strict: true`, forced/required tool selection, constrained JSON output or
  `parallel_tool_calls: false` with tools. Disable unsupported guarantees in
  the caller; do not silently promise schema enforcement the backend lacks.
- One representative cold long-context coding request and a short voice request
  pass with bounded output and realistic client/proxy timeouts. Prefix reuse
  may improve repeats but is not a substitute for a cold-prefill test.
- API catalog/capabilities, public chat, any used Responses/Messages path, radio,
  Realtime allowlist/default, affected Hermes-LiveKit profiles, and both phone
  configs are checked. Verify each actual path in use, not just `/health`.
- All temporarily stopped services are restored exactly as recorded. The 4090
  and unrelated cloud/fallback roles are unchanged. Record commits, private
  backup locations, restart evidence and any intentionally unchanged consumers.

If acceptance fails, stop the new model, restore the previous service and boot
state, restore only changed private settings, revert the focused tracked change
through normal Git delivery if needed, and restart/reselect affected consumers.
Repeat the same health, tool, routing and voice checks on the old model. Retain
both model artifacts until rollout acceptance; deleting the rollback weights
is not part of switching a default.

Model-level evidence: [Cinference benchmark and launcher checks](bench/cinference-6000/RESULTS.md).
This guide's inventory is read-only; it does not claim an end-to-end fast-Qwen
rollout through all harnesses or the public API has already happened.
