# Issue #4: Smarty Qwen coding-agent capacity lease

## Goal

Provide OMP and Prime Agent workers with a non-disruptive lease for the
already-running managed Qwen 3.8 27B endpoint on `smarty`. Preserve production
services, deliberate VRAM and request-slot headroom, other CUDA work, and all
active LegoLM research.

## Work plan

1. Record the live `ktxsvc` roster, protected health endpoints, Qwen listener,
   GPU allocation, free VRAM, and any active LegoLM or unknown CUDA owner.
2. Add an atomic owner-only lease registry with durable owner identity,
   bounded capacity, heartbeat, expiry, crash recovery, and release evidence.
3. Admit only after fail-closed preflight, a small request through the existing
   Qwen endpoint, and a matching postflight. Never manage a service or signal a
   process from the lease tool.
4. Exercise concurrent admission, stale lease, agent crash, LegoLM activity,
   unknown CUDA work, insufficient headroom, slot contention, service health,
   ownership, secret rejection, and Qwen PID-change behavior in tests.
5. Recheck the live baseline and exclusive writer claim, validate locally,
   then Git-sync and run one bounded live acquire/release canary only if the
   measured state remains safe. Do not stop or restart any service.
6. Freeze the focused evidence for a different agent to audit. Keep the issue
   open and do not mark it Done from the builder session.

## Status

- In progress under the explicit non-atomic/manual serialized-writer claim
  recorded on issue #4. The atomic cross-harness provider remains tracked by
  `kortexa-ai/esp32-dash#3`; this work does not claim that dependency is done.
- The approved surfaces are `scripts/qwen_capacity_lease.py`,
  `tests/test_qwen_capacity_lease.py`, this PLAN section, and the matching
  README lease section. No other repository surface is in scope.
- The initial Smarty baseline was read-only: one healthy managed Qwen process
  owned port 2053 with two idle slots; all observed CUDA PIDs belonged to known
  managed services; no LegoLM GPU owner or prior agent lease was observed.
- No production service or LegoLM code, data, job, service, or experiment has
  been stopped, restarted, started, or changed.

---

# OrcaRouter Qwen 3.8 27B uncensored alignment

## Goal

Use the official OrcaRouter collection for llama.cpp Q4_K_M, MLX 4-bit, and
vLLM FP8, and configure the embedded MTP head where the runtime supports it.

## Status

- Configuration complete. Both Qwen 3.8 variants now share the same serving
  knobs. llama.cpp and vLLM use MTP depth 3; both MLX entries use `mlx_vlm`
  with their repository-root 4-bit models and no separate drafter.
- Runtime download and capability validation are complete. The official GGUF
  passed all five text, tool, and vision canaries. A paired 450 W SPEED-Bench
  run against vanilla Qwen 3.8 found equivalent practical throughput: the
  uncensored model was 1.0% slower on the qualitative decode mix, 2.2-3.8%
  slower on the fixed single-request passes, and 4.1% faster in aggregate with
  two clients. A matched MTP depth/on-off sweep remains optional tuning work.
- The authenticated smarty Hugging Face account still needs access approval
  for the gated FP8 repository before vLLM can download it.

---

# Ornith 1.5 onboarding and agentic-model profile alignment

## Goal

Add Ornith 1.5 9B and 35B-A3B on `smarty` llama.cpp and `snappy` MLX, select
the best supported MTP mode by measurement, and align the comparable Qwen 3.x
and Gemma 4 models around one single-request serving policy.

## Work plan

1. Verify native context, official GGUF/MLX artifacts, and available MTP heads.
2. Add managed model entries on unused ports and normalize both comparison
   families to Q4 weights, q8 KV, one slot, and native maximum context.
3. Launch each Ornith model on smarty to populate the normal cache, run
   multimodal/tool canaries, and compare MTP on/off where a real head exists.
4. Validate the MLX repository/config metadata without starting work on busy
   snappy; leave matched MLX runtime tests for a separate window.
5. Record results, restore the original smarty service set, validate, commit,
   push, and sync the relevant hosts.

## Status

- Complete on smarty. Official Ornith GGUF repositories provide `Q4_K_M`, not
  an Unsloth dynamic quant; official 4-bit MLX repositories exist for both.
- The upstream models advertise 262,144-token context. The common profile is
  therefore 262K and one slot across dense and MoE families, except for the
  selected Gemma 12B GGUF, which reports a 131,072-token trained context.
- Ornith 9B uses the protoLabs distilled MTP GGUF at depth 2; it was 55.1%
  faster than the official no-MTP GGUF in the matched smoke. Ornith 35B keeps
  its official GGUF with MTP disabled; its native head was 10.6% slower.
- Qwen 3.6 35B-A3B keeps MTP depth 3 after a fresh 50.3% matched win. Qwen 3.5
  9B and both Gemma llama.cpp targets run without a compatible GGUF drafter.
- All six models completed the standard 450 W smarty suite. The original
  Qwen 3.8, LFM2.5 VL 3B, and Gemma 4 E2B services were restored and verified
  healthy. Snappy was intentionally left untouched while it was busy.

---

# 2026-08-20 llama.cpp benchmark rebaseline

## Goal

Rebaseline Qwen 3.8, the LFM family, and Gemma 4 on `smarty` with auditable
runtime provenance, without unified CUDA memory or overlapping model servers.

## Status

- Completed. Seventeen recorded runs covered Qwen 3.8 27B; all applicable LFM2/LFM2.5
  models from VL 3B through 230M; Gemma 4 E4B, E2B, and 12B; plus preliminary
  task-specific serving canaries for LFM Extract and Embedding.
- Every CUDA run used the 450 W cap with CUDA graphs disabled and unified
  memory absent. CPU runs were explicitly recorded as CUDA-inference false.
- No OOM or GPU fault occurred. The LFM VL 450M tool-call grammar failure and
  LFM 350M CPU long-context collapse are recorded rather than discarded.
- The temporary services were uninstalled. The exact original service set—
  Qwen 3.8 27B, LFM2.5 VL 3B, and Gemma 4 E2B—was restored and health-checked.
- LFM2.5 350M was subsequently changed from CPU-first everywhere to platform
  auto-detection, matching 230M's GPU-on-smarty/CPU-on-Pi behavior. Its 450 W
  CUDA rebaseline completed without errors, and the original services were
  restored again.
- Its follow-up alignment adds LiquidAI's official 8-bit MLX checkpoint on
  snappy and replaces the oversized 128K-per-slot llama allocation with the
  model's native 32K per slot. Embedding 350M remains llama.cpp-backed because
  no official MLX export exists, but its two slots now match the trained
  512-token maximum instead of allocating 1K each.
- The edge-serving follow-up normalizes 230M, 350M, and VL 450M to four native
  32K slots. Their Pi CPU paths use Q4_K_M weights and q4 KV; VL 450M expresses
  the same shape in MLX as a 32K per-sequence KV limit with four sequences.
  Extract 350M now selects Metal-backed llama.cpp on snappy while retaining its
  separate long-context CPU configuration elsewhere.
- Native-context 230M and four-slot VL 450M CUDA refreshes are complete at
  450 W. Both retained their single-stream performance; the original
  oversized/one-slot records remain in the local provenance archive.

---

# Benchmark methodology reset

## Goal

Archive unauditable llama.cpp benchmark records and prepare a reproducible,
endpoint-only benchmark workflow without starting or stopping any model.

## Work plan

1. Archive the pre-reset benchmark log and bespoke direct-launch harnesses.
2. Define the current model matrix, exclusions, test battery, and acceptance
   rules.
3. Add automatic capture of exact model configuration, live arguments and
   safe environment, llama.cpp revision, GPU power cap, CUDA graph state, and
   telemetry.
4. Integrate upstream SPEED-Bench for standard qualitative, fixed-shape, and
   concurrency workloads.
5. Add capability canaries and validate all tools in dry-run mode only.

## Status

- Completed 2026-08-20. The old log and direct-launch harnesses are archived.
  The new endpoint-only workflow captures exact configuration, live runtime
  state, GPU power and telemetry, pins the SPEED-Bench dataset revision, and
  refuses CUDA unified memory. Syntax, dry-run, fixture, and metadata capture
  validation passed without inference or service changes.

---

# Qwen 3.8 27B Xid 8 investigation

## Goal

Preserve the known failing long-context OMP session, disable only llama.cpp
MTP, and replay the same workload to test whether MTP is required to trigger
the RTX PRO 6000 Blackwell Xid 8 lock.

## Work plan

1. Preserve the OMP session, client transport log, checksums, system state,
   failure timeline, and baseline MTP-on performance.
2. Disable only MTP in the Qwen 3.8 27B llama.cpp configuration.
3. Restart the managed service and verify port ownership, health, launch
   arguments, GPU recovery state, and the retained 450 W runtime limit.
4. Resume the preserved OMP session and record stability and MTP-off
   performance before selecting the next controlled variable.
5. Stop the managed service and build a direct `llama-server` replay harness
   from the session boundary immediately before the first crash.
6. Run the fixed replay with MTP depths 1 and 2, then with the pinned DFlash v1
   bootstrap drafter. Capture server output, GPU telemetry, kernel events, and
   OMP output for each isolated run.
7. Restore the default 600 W power limit and run the depth-3 replay again to
   compare throughput, temperature, sustained load, and fault behavior.
8. Preserve the later `magicapp` crash session and replay it through the
   managed service at MTP depths 3, 2, and 1 under the persistent 450 W cap.
9. Disable CUDA graphs at runtime and repeat the managed replay at MTP depths
   1 and 3, keeping the session, power cap, and server configuration fixed.
10. Make CUDA graphs default-off in the shared llama.cpp launcher with an
    explicit per-model opt-in, then remove Qwen's host-local systemd drop-ins.

## Status

- Completed 2026-08-19. The known failing OMP session and matching client log
  are preserved under `investigations/2026-08-19-qwen-3.8-27b-xid8/` with
  matching source and destination SHA-256 checksums.
- MTP was disabled for the isolation run. The managed service restarted on a
  new PID, owned port 2053, passed health and a small decode canary, and
  retained the 450 W runtime power limit. Its launch arguments contained no
  speculative-decoding flags during that test.
- Disabling MTP reduced idle Qwen VRAM from 30,236 MiB to 28,116 MiB. The next
  step was to resume the preserved OMP session and record stability and speed.
- The replay completed 11 inference turns and generated 18,294 tokens while
  context grew from 116,485 to 140,442 tokens. No Xid or CUDA error occurred.
  Long-context decode ran at approximately 41-45 tok/s, about 40-45% slower
  than the comparable MTP-on attempts. The next isolation test is MTP depth 1.
- Completed 2026-08-19: the direct-server harness replays the first failed
  turn through OMP's actual `/retry` operation with GooeyPi's extensions and
  yolo approval mode. It captures raw server output, GPU telemetry, kernel
  events, and metadata without changing `model.json`.
- One full yolo pass each for MTP depths 2 and 3 and the pinned DFlash v1
  bootstrap completed without Xid 8. The depth-3 control also passed, so this
  session is a realistic stress fixture but not a deterministic reproducer.
  DFlash worked at 56.52 tok/s with 33.6% initial acceptance; it was slower
  than native MTP but faster than the earlier MTP-off run.
- A second depth-3 pass at the default 600 W limit completed 13 inference turns
  and generated 17,965 tokens without an Xid. Initial prefill throughput was
  2,286.90 tok/s and decode was 87.65 tok/s, 26.5% and 10.8% above the 450 W
  depth-3 samples. The run reached 89 C and had 325 busy telemetry seconds.
  This result does not support a simple temperature or cumulative-load
  threshold, but it does not exclude retained GPU driver or GSP state.
- Two additional 600 W depth-3 passes ran back-to-back and completed without a
  fault. Across all three 600 W passes, the model completed 65 inference turns,
  generated 75,468 tokens, and had 1,185 sampled busy seconds. The final pass
  reached 157,381 context tokens and 91 C. This strongly reduces the
  probability of context size, temperature, or cumulative load as the primary
  trigger.
- A later `magicapp` session provided a stronger reproducer. At 450 W, fresh
  managed-service runs failed with the same Xid 8 at MTP depth 3 after 26
  completed turns and at depth 2 after 60 completed turns. MTP depth 1
  completed all 82 turns, generated 42,032 tokens, and reached 146,887 context
  tokens without an Xid. The passing run peaked at 86 C, above both failures.
- With `GGML_CUDA_DISABLE_GRAPHS=1`, MTP depth 1 ran for 15 minutes through 47
  completed turns, 33,400 generated tokens, and 150,289 main-session context
  tokens without an Xid. Depth 3 then ran for 15 minutes through 88 completed
  turns, 47,402 generated tokens, and 179,075 main-session context tokens
  without an Xid. Both clients stopped at the configured time limit.
- The shared llama.cpp launcher now disables CUDA graphs by default. Individual
  models can opt in with `llama.cuda_graphs: true`; no model currently does.
- A direct startup probe measured two full 262K slots at 38,223 MiB for Qwen,
  leaving 27,420 MiB free with the normal service stack. Production was set to
  three full slots through llama-specific context and parallel overrides; the
  uncensored variant has matching slot dimensions. The managed three-slot
  service leaves 13,174 MiB free at idle, and a three-request canary completed
  on all slots.
- The final production state is 450 W, three 262K slots, MTP enabled at depth
  3, CUDA graphs disabled, managed service active, port 2053 healthy, and the
  live process environment and arguments verified. The host-local systemd unit
  applies the power cap and remains outside `ktxsvc` discovery.

---

# Qwen 3.8 27B Uncensored roster entry and archive

## Goal

Add the Qwen 3.8 27B Uncensored OrcaRouter GGUF release with its available
`Q4_K_M` quant, 262K context, one request slot, q8 KV cache, and managed service
definitions. Preserve the selected weights and release metadata on `smarty`
without starting the model or changing any running GPU service.

## Work plan

1. Verify the Hub revision, available quants, context, projector, MTP layout,
   and an unused port.
2. Add the GGUF-only model config, service units, model-named launchd delegate,
   and inventory documentation.
3. Download the pinned Q4 model, projector, optional MTP companion, license,
   checksums, manifest, and reproduction notes to `smarty`'s data disk.
4. Validate checksums, config parsing, backend dispatch, plist XML, shell
   syntax, systemd, and roster discovery without loading weights or starting a
   server.
5. Commit and push under Sparta rules, sync the clean `smarty` checkout, and
   verify that its original services and GPU workloads remain unchanged.

## Status

- Completed 2026-08-18. Hub revision
  `58ebd123013160600229eda180b5b17f3fb7af9d` provides a 16.8 GB `Q4_K_M`
  GGUF, matching 931 MB multimodal projector, and optional 3.2 GB MTP-only
  companion. The GGUF-only roster entry uses unused port 2056 and leaves MTP
  disabled until the release's separate or embedded draft layout is validated.
- The 20 GB pinned archive is under
  `/home/francip/data/models/huggingface/chimingw/Qwen3.8-27B-Uncensored-OrcaRouter-GGUF/58ebd123013160600229eda180b5b17f3fb7af9d`
  on `smarty`. All 21 downloaded files match the pinned Hub revision. The
  model, projector, MTP companion, manifest, license, and reproduction files
  also match the publisher's SHA-256 list; its README checksum is stale, but
  the downloaded README matches the pinned Hub blob exactly.
- JSON/config parsing, llama.cpp argument construction through a non-model
  stub, launchd XML and delegation, systemd structure, unique ID/port checks,
  and `ktxsvc` discovery passed without loading weights or starting a server.
  Port 2056 remains closed. Every original service, GPU PID, and VRAM allocation
  on `smarty` is unchanged from the pre-download baseline.

---

# LFM2.5 VL 3B roster entry

## Goal

Add LFM2.5 VL 3B with LiquidAI's official Q8_0 GGUF and MLX 8-bit
checkpoints, its supported 32K multimodal context, one request slot, and
managed service definitions for Linux and macOS.

## Work plan

1. Verify the official repositories, formats, context, runtime support, and an
   unused port.
2. Add the model config, service units, model-named launchd delegate, and
   inventory documentation.
3. Validate config parsing, backend dispatch, plist XML, shell syntax, systemd,
   and roster discovery without downloading weights or starting a server.
4. Commit and push under Sparta rules, then sync and verify the clean `smarty`
   checkout without installing or enabling the service.

## Status

- Completed 2026-08-18. LiquidAI's model card and Hub repositories confirm the
  32,768-token context, official Q8_0 GGUF with matching Q8_0 vision projector,
  official MLX 8-bit checkpoint, native vLLM support, and `lfm2` tool parser.
  The roster now exposes `lfm2.5-vl-3b` on unused port 2055 with one request
  slot, Q8 KV cache, and model-named service entry points.
- JSON/config parsing, llama.cpp, MLX, and vLLM dispatch, MLX model-family
  loading, launchd XML and delegation, systemd template parity, unique ID/port
  checks, and `ktxsvc` discovery passed without downloading a checkpoint or
  starting a model server. The service remains uninstalled and disabled.

---

# llama-server CORS policy

## Goal

Stop llama-server's permissive-CORS warning and prevent unrelated browser
origins from calling model endpoints, without changing LAN/Tailnet firewall
policy or the authenticated `api.server` proxy.

## Work plan

1. Inspect the upstream CORS implementation and current network topology.
2. Apply the upstream `localhost` policy to every shared llama-server launch
   path, including chat and embeddings on GPU and CPU.
3. Verify localhost preflights are allowed, unrelated origins are omitted,
   non-browser requests still work, and the startup warning is absent.
4. Commit and push under Sparta rules, then sync `snappy` without restarting a
   model service.

## Status

- Completed 2026-08-18. llama.cpp CORS accepts browser origins, not client
  CIDRs. Every GPU/Metal llama-server branch now sets its special `localhost`
  policy; CPU branches do the same when the installed binary supports the
  option, preserving compatibility with one older idle Pi build.
- A model-free loopback router test allowed a localhost preflight, omitted the
  CORS header for an unrelated origin, accepted an originless health request,
  and emitted no permissive-CORS warning. All four launch branches passed
  syntax and argument traces. No model service was restarted, and the live
  LFM2.5 1.2B Thinking endpoint remained healthy.
- The same-origin embedded Web UI and backend clients remain available. Actual
  LAN/Tailnet access stays enforced by the existing firewalls, while public
  model access remains behind the authenticated `api.server` proxy.

---

# LFM2.5 1.2B Instruct roster entry

## Goal

Add LFM2.5 1.2B Instruct beside the existing Thinking variant with the official
Q8_0 GGUF and MLX 8-bit checkpoints, its supported 32K context, one request
slot, and managed service definitions for Linux and macOS.

## Work plan

1. Verify the official repositories, formats, context, and an unused port.
2. Add the model config, service units, model-named launchd delegate, and
   inventory documentation.
3. Validate config parsing, backend dispatch, plist XML, shell syntax, systemd,
   and roster discovery without downloading weights or starting a server.
4. Commit and push under Sparta rules, then sync and verify the clean `snappy`
   checkout without installing or enabling the service.

## Status

- Completed 2026-08-18. Official Hub metadata confirms the Instruct, Q8_0 GGUF,
  and MLX 8-bit repositories. The roster now exposes
  `lfm2.5-1.2b-instruct` on unused port 2054 with the documented 32,768-token
  context, one slot, Q8 KV cache, and model-named service entry points.
- JSON/config parsing, llama.cpp dispatch, launchd XML and delegation, systemd,
  unique ID/port checks, and `ktxsvc` discovery passed without downloading a
  checkpoint or starting a model server. The service remains uninstalled and
  disabled on both hosts.

---

# Model-named launchd entry points

## Goal

Make every model LaunchAgent execute a distinct `kortexa-<model-id>.sh`
entry point that delegates to the shared `run.sh`, so macOS Login Items names
the model instead of showing repeated generic scripts.

## Work plan

1. Inventory all repository plists and wrappers plus installed model agents on
   `snappy`.
2. Add the missing named wrappers and update only their plist executable paths.
3. Validate every plist, wrapper, model argument, and executable mode on macOS.
4. Commit and push under Sparta rules, then sync and verify the clean `snappy`
   checkout without changing which model services are installed or enabled.

## Status

- Completed 2026-08-18. The inventory found 25 model plists: 11 already used
  named wrappers, and the remaining 14 now do too. Every plist has a matching
  executable `kortexa-<model-id>.sh` delegate and no plist directly executes
  `run.sh`.
- All 25 plists and wrappers passed XML, label, argument, executable-mode, and
  Bash 3.2 validation on `snappy`. Its clean checkout was fast-forwarded through
  Git; no model LaunchAgent was installed, enabled, loaded, or restarted.

---

# Qwen 3.8 27B SGLang NVFP4 + DSpark comparison

## Goal

Test the current SGLang Qwen 3.8 recipe on `smarty` with the RadixArk NVFP4
checkpoint and DSpark drafter. Compare it with the production llama.cpp
UD-Q4_K_XL + MTP service using matched single-request workloads, then keep the
faster practical backend. Also verify whether the llama.cpp path can set Qwen's
default `reasoning_effort` to `medium` instead of the model-template default of
`xhigh`.

## Live baseline — 2026-08-17 08:25 PDT

- `smarty` is clean on `main` at `90b97ba`; the root filesystem has about
  500 GiB available.
- Qwen 3.8 is installed, enabled, running, and healthy on port 2053 through
  llama.cpp. Its process uses 30,194 MiB.
- Image base, Vision, Gemma 4 E2B, ASR, and TTS are running and healthy.
  ComfyUI is also running and uses 550 MiB, but its usual port 8188 was already
  closed at baseline; leave that process alone.
- The RTX PRO 6000 uses 59,688 MiB and has 37,554 MiB free.
- The user explicitly authorized production downtime under the Smarty guide.
  Snapshot and restore exactly the services stopped for the benchmark.
- `snappy` has three unrelated user edits in LFM launchd plists. Do not stage
  or alter them.

## Work plan

1. Extract the official RTX PRO 6000 recipe and identify the exact current
   SGLang build, checkpoint, memory, attention, parser, and speculative flags.
2. Add the smallest isolated text-SGLang launch/config support and a matched
   benchmark path; download images and checkpoints without allocating GPU.
3. Stop the authorized safe-list services once, starting with Qwen 3.8, until
   the recipe has deliberate VRAM headroom. Start with a one-request canary.
4. Compare SGLang without speculation, with in-checkpoint MTP/EAGLE, and with
   DSpark. Run the official 8192-input/1024-output case and the existing
   prose/code workloads against the best candidate and llama.cpp.
5. Select the production backend and settings based on measured speed,
   correctness, memory, startup behavior, and API/tool compatibility. Set
   llama.cpp's default reasoning effort to `medium` if the installed build and
   Qwen template honor it.
6. Validate scripts/config, stop the benchmark process, restore and health-check
   exactly the original services, then commit and push only the focused work.

## Status

- Completed 2026-08-17. Recipe and live baseline captured. SGLang stable is
  0.5.17, while the day-0 Qwen 3.8 recipe uses the newer dedicated
  `lmsysorg/sglang:qwen38-27b` image. Docker Hub throttling prevented that image
  from completing, so the comparison used the recipe's pinned current-source
  runtime instead of an older release wheel.
- The RTX PRO 6000 recipe uses the RadixArk NVFP4 checkpoint, FlashInfer,
  FP8 KV, 2048-token prefill chunks, FP32 GDN state, and an explicit Mamba
  state/KV ratio. The comparison harness covers no speculation, MTP/EAGLE with
  ReplaySSM, the linked DSpark low-latency recipe, and a lower-memory DSpark
  configuration sized for one production request.
- At baseline, llama.cpp inherited Qwen's template default of `xhigh`. Config
  parsing and the generic launcher now set `medium` model-wide while retaining
  standard per-request overrides.
- The fresh llama.cpp control on build 10470 (`34af94cd9`) averaged 114.64
  wall tok/s for prose and 156.22 for code over three forced 600-token runs.
  A cold 8,193-input / 1,024-output control reached 119.83 wall tok/s and
  170.57 server decode tok/s, with 3,579 prompt tok/s and 84.25% MTP draft
  acceptance.
- Authenticated Hugging Face access is verified as `francip`. Native Xet made
  no progress in bounded probes, so the checkpoints were fetched through
  authenticated resolver URLs with Xet disabled and resumable HTTP ranges.
  Hub verification passed for all 22 NVFP4 files and all six DSpark files.
- The exact recipe image remains a resumable secondary control because Docker
  Hub throttled its large layers. A pinned checkout of current SGLang `main` at
  `af743371c` is ready in an isolated ignored venv with Torch 2.13 CUDA 13,
  FlashInfer 0.6.17, and SGLang Kernel 0.4.6.post1; every recipe flag passed a
  CLI smoke check. Production remained healthy throughout preparation.
- Current-main EAGLE was the winner. A production-sized 45% memory / one-slot
  configuration averaged 127.95 prose and 165.93 code tok/s, an 8.5% mixed
  gain over the fresh llama.cpp control. The recipe's OpenAI 8K-input / 1K-output
  run reached 142.60 tok/s versus llama.cpp's 119.83 wall tok/s. It passed text,
  tool, and image canaries and left 53,494 MiB free while isolated.
- DSpark crashes on current main because it matrix-multiplies against the
  packed NVFP4 language head. The official pending fix at `da1fbe873` launches,
  but managed only 88.08 prose / 173.85 code. Its earlier native-endpoint 8K/1K
  run reached 90.03 tok/s; that case was not repeated through the corrected
  OpenAI benchmark client. The exact Docker image remained 9.73 GB short after
  prolonged Hub throttling; its pull was stopped and the resumable cache
  retained.
- Keep llama.cpp in production. SGLang's modest gain costs about 14 GiB more
  VRAM and roughly five times the warm startup latency. More importantly,
  SGLang's default chat-template kwargs currently overwrite per-request
  reasoning effort. Restored llama.cpp's final probe rendered 14 / 14 / 56
  prompt tokens for omitted / medium / xhigh, so it supplies the requested
  medium default and standard request override without a custom engine patch.
- Each authorized GPU block restored exactly the six recorded services. Their
  health endpoints are HTTP 200, Qwen is again served by llama.cpp on port 2053,
  and no SGLang, benchmark, aria2, or Docker-pull process remains.

---

# Qwen 3.8 27B bring-up

## Goal

Add Qwen 3.8 27B serving support for `smarty` with its native 262,144-token
context, one llama.cpp request slot, the standard `UD-Q4_K_XL` weight quant,
q8 KV cache, and built-in MTP. Add the completed MLX 4-bit checkpoint as the
`snappy` path. Keep every other production service running. Qwen 3.6 27B is
authorized to stop only for the Qwen 3.8 test block.

## Live baseline — 2026-08-14 08:50 PDT

- `smarty` is clean on `main` at `29502e0`; port 2053 is free and the root
  filesystem has 565 GiB available.
- The RTX PRO 6000 uses 63,351 MiB and has 34,536 MiB free.
- Qwen 3.6 27B is installed, enabled, running, and healthy on port 2032. Its
  llama.cpp process uses 30,180 MiB.
- Image base, Vision, Gemma 4 E2B, ASR, TTS, and Qwen3 Embedding 0.6B are also
  healthy. They must remain running.

## Work plan

1. Add the llama.cpp, MLX, and vLLM config, both service units, and inventory
   entry.
2. Validate the config parser and both service-unit formats locally.
3. Commit and push the change, then fast-forward the clean `smarty` checkout.
4. Stop only Qwen 3.6 27B. Sweep Qwen 3.8 llama.cpp MTP depths 0 through 4
   with matched 600-token prose and Python workloads.
5. Repeat each workload three times at the best depth, measure cold and warm
   long-prompt handling, verify text, tool, and image requests, and record GPU
   memory plus draft acceptance.
6. Stop the test server, restore Qwen 3.6, run the same repeated workloads for
   a matched speed comparison, and verify every baseline endpoint.

## Status

- Completed 2026-08-14. The llama.cpp, MLX, and vLLM configurations, service
  units, inventory entry, and reproducible benchmark harness are committed on
  `main`. The authenticated `UD-Q4_K_XL` snapshot is cached on `smarty`; the
  completed MLX 4-bit repository passed a local architecture/config check on
  `snappy`.
- MTP depth 3 was the best mixed setting. It averaged 117.35 decode tok/s on
  prose and 157.37 on code in the final matched run, versus 120.09 and 156.36
  for Qwen 3.6. With MTP disabled, Qwen 3.8 managed only 72.38 tok/s.
- A 36,887-token cold prompt prefetched at 3,023.94 tok/s and decoded at
  112.57 tok/s; the exact repeat cached 36,883 tokens and decoded at 113.28
  tok/s. Text, required tool-call, and 512px image canaries all passed.
- Qwen 3.8 was stopped after the test. Qwen 3.6 is installed, enabled, running,
  and healthy again; port 2053 is closed, every protected endpoint is HTTP
  200, and idle GPU memory returned to the production baseline.

---

# LFM2.5 1.2B Thinking and 2.6B bring-up

## Goal

Add both text models with official Q8_0 GGUF checkpoints on `smarty` CUDA,
official 8-bit MLX checkpoints on `snappy`, maximum supported context, and one
request slot. Verify and benchmark each backend, then leave both services
uninstalled, stopped, and without downloaded model artifacts on either host.

## Work plan

1. Add model configs, service units, inventory docs, and minimal MLX subfolder
   support for the 2.6B checkpoint's multi-precision repository.
2. Validate parsers, launchers, launchd plists, and systemd units.
3. Temporarily install, smoke-test, and benchmark both MLX services on `snappy`.
4. Commit and sync through Git, then repeat on `smarty` with full CUDA offload
   while preserving resident production services and at least 10 GiB VRAM.
5. Record results; uninstall both services, remove only artifacts downloaded by
   this run, and verify ports, processes, services, caches, and resident health.

## Status

- Completed 2026-08-04. Both models returned the correct arithmetic canary on
  MLX and CUDA; llama.cpp reported one 32,768-token slot for 1.2B Thinking and
  one 128,000-token slot for 2.6B.
- Warm 256-token decode measured 147.9 / 68.5 wall tok/s on `snappy` MLX and
  749.6 / 330.8 server tok/s on `smarty` CUDA for 1.2B / 2.6B, respectively.
  Matched 4.1K-token cold and cached-prompt results are in
  `bench/BENCHMARKS.md`.
- The launchd plists use model-named wrapper scripts, so macOS Login Items do
  not show two more generic `run.sh` entries.
- No production service was stopped. Both new services are uninstalled and
  off on both hosts; ports 2026 and 2027 are free, all four downloaded model
  caches were removed, all resident `smarty` endpoints are HTTP 200, and both
  repositories are clean at the same commit.

---

# LFM2.5 VL 450M GPU benchmark on smarty

## Goal

Measure the same text, 512px photo, and 1536px tiled-photo workloads on
`smarty` with full GPU offload, without stopping or disturbing resident
production services.

## Live baseline — 2026-08-04

- RTX PRO 6000: 64,798 MiB used, 32,453 MiB free, 0% utilization, 32 C.
- Running GPU set: image base, ComfyUI, Vision, Qwen 3.6 27B, Gemma 4 E2B,
  ASR, TTS, and Qwen3 Embedding 0.6B.
- Repository is clean on `main` at `830e373`; port 2052 is free and no LFM2.5
  VL benchmark process exists.
- No service will be stopped. Keep at least 10 GiB free and begin with one
  short image request before running the matched benchmark.

## Work plan

1. Verify every resident GPU endpoint and capture the exact GPU baseline.
2. Start one isolated llama.cpp GPU server and run a small canary.
3. Measure matched text, 512px, and 1536px cold/warm requests.
4. Stop the benchmark server, recheck every baseline service, and log results.

## Status

- Completed 2026-08-04 without production downtime. Full GPU offload delivered
  1,310.64 text tok/s, 0.084s / 0.026s cold/warm 512px images, and 0.159s /
  0.062s cold/warm 1536px tiled images.
- Peak sampled use left 30,955 MiB VRAM free. The benchmark server was stopped,
  port 2052 is free, all eight resident endpoints are HTTP 200, and final GPU
  use returned to within 90 MiB of baseline.
- Results are recorded in `bench/BENCHMARKS.md`.

---

# LFM2.5 VL 450M deployment and Pi benchmark

## Goal

Keep the model available on `snappy` as a low-latency camera/photo service,
then measure its CPU-only image performance on one 8 GB Raspberry Pi 5 with
a matched Q8_0 model and projector.

## Work plan

1. Inspect the existing service and both Pis; use the quieter healthy Pi.
2. Install, start, and exercise the managed MLX service on `snappy`.
3. Update llama.cpp on the selected Pi only if needed, copy the non-repository
   Q8_0 artifacts, and benchmark text plus small and tiled image inputs.
4. Record results, stop the temporary Pi server, and verify affected services.

## Status

- Completed 2026-08-04: the managed MLX service is installed, enabled at
  login, running, and photo-tested on `snappy` at port 2052 with a 32K
  effective context.
- `happyhippo` was idle and selected for the Pi test. llama.cpp was updated to
  build 10267, and the verified 459.7 MiB Q8_0 model/projector pair remains on
  the Pi for later use.
- The Pi delivered 26.71 text tok/s, a 4.07s cold / 0.60s warm 512px image,
  and a 60.48s cold / 1.01s warm 1536px tiled image. All image responses were
  correct in the smoke test.
- Results are logged in `bench/BENCHMARKS.md`. The temporary Pi server and test
  images were removed, port 2052 is free there, and the `snappy` service is
  healthy.

---

# Qwen 3.6 27B engine and memory tuning

## Goal

Configure Qwen 3.6 27B for production with a fixed vLLM KV-cache byte
budget, native 262,144-token maximum context, up to four concurrent
requests sharing the cache, native MTP speculative decoding, and deliberate
VRAM headroom alongside the other services on `smarty`. Measure the latest
stable vLLM against the current llama.cpp build before choosing the production
engine and settings.

## Live baseline — 2026-08-01 13:35 PDT

- Host: `smarty`, Linux, RTX PRO 6000 Blackwell, 97,887 MiB VRAM.
- GPU: 36,280 MiB used; 61,607 MiB free.
- Qwen 3.6 27B and Gemma 4 31B are not installed or running.
- Healthy GPU services to preserve: Gemma 4 E2B, Qwen3 Embedding 0.6B,
  ASR, TTS, Vision, and LegoLM.
- Vision uses 14,846 MiB and may be stopped temporarily if the user-authorized
  benchmark cannot retain safe headroom. If stopped, restore it and verify
  `http://127.0.0.1:4001/health` returns HTTP 200.
- Repository: clean `main` at `f6ede81`.
- llama.cpp: build 10200 (`5f55650a7`).
- vLLM before upgrade: `0.19.2rc1.dev131+gac58e2a17.cu130`.
- vLLM after upgrade: stable `0.26.0` with FlashInfer `0.6.14`.

## Work plan

1. Inspect current launch/config support and select a reproducible latest
   stable vLLM build.
2. Add minimal configuration plumbing for exact KV bytes and native Qwen MTP.
3. Cache required model files without allocating GPU memory.
4. Measure current llama.cpp with matched prose/code workloads, MTP depths,
   GPU memory, and acceptance rate.
5. Measure latest vLLM with the same workloads, first without and then with
   native MTP; determine the smallest exact KV budget that admits a 262,144
   token request while leaving comfortable production headroom.
6. Select the production engine/configuration, document results, validate
   parsers/launch commands, and restore every temporarily stopped service.

## Safety bounds

- Keep at least 10 GiB VRAM free during canaries; prefer at least 15 GiB in
  the final production state.
- Stop only `vision.server` if the initial canary cannot preserve the bound.
- Run one model/engine candidate at a time and begin with a short batch-1
  request.
- Use exact `kv_cache_bytes`; do not rely on GPU-utilization percentages.
- Do not leave a benchmark server or model process running after measurement.

## Status

- Baseline captured.
- Exact-KV and native-MTP launcher plumbing implemented.
- Dummy-weight allocator canary confirmed that `9,804,185,600` bytes
  (9.130859 GiB) produces exactly 262,144 aggregate KV tokens and 1.00x
  full-context concurrency with MTP depth 2. `max_num_seqs=4` shares this pool
  dynamically. The pool is 176 allocator pages: 164 full-attention pages for
  262,144 tokens plus 12 small recurrent-state reservations (three hybrid
  groups times four schedulable sequences). Four-way scheduling therefore
  costs only 478.125 MiB more than a one-sequence pool, not four full contexts.
- The 60% GPU-utilization value is only vLLM's startup admission guard; vLLM
  explicitly confirms that the exact KV byte value overrides it.
- Both model artifacts are cached locally. The final vLLM configuration uses
  MTP depth 4 and a 183-page, `10,194,124,800`-byte (9.494019 GiB) FP8 KV
  pool. vLLM reports exactly 262,144 aggregate tokens and 1.00x full-context
  concurrency. Four requests share the pool dynamically.
- Current llama.cpp build 10200 is fastest at MTP depth 3: 115.19 tok/s prose
  and 156.26 tok/s code (three 600-token runs). Its measured footprint is
  about 28,409 MiB above the resident-service baseline.
- vLLM 0.26.0 with automatic FlashInfer attention is fastest at MTP depth 4:
  85.91 tok/s prose and 129.48 tok/s code (three 600-token runs), plus 310.44
  aggregate tok/s for four simultaneous 600-token requests. It uses about
  42,411 MiB above baseline and leaves 18,537 MiB free with every resident
  service running.
- Forcing Triton attention was rejected: single requests worked, but the
  four-request test caused an illegal CUDA memory access. GPU state recovered
  when the isolated benchmark process exited and every resident health check
  remained HTTP 200.
- Production remains on llama.cpp for the common single-request workload;
  the exact-cache vLLM configuration is available through `--engine vllm` for
  bursty concurrent workloads.
- Completed 2026-08-01: configuration and launcher syntax pass, the vLLM
  environment has no dependency conflicts, no benchmark process remains, GPU
  use returned to the 36,302 MiB resident baseline, and all six preserved GPU
  services are running with HTTP 200 health checks.
