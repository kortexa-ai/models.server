# Qwen 3.8 27B Xid 8 investigation

## Purpose

Preserve the known failing OMP session and the observed server state before
changing one variable: disable llama.cpp MTP speculative decoding. The copied
session can be resumed to replay the approximately 116K-token request that
repeatedly locked the GPU.

## Preserved artifacts

- `omp-session.jsonl` is the complete OMP session copied from `snappy`:
  `~/.omp/agent/sessions/-src/2026-08-18T17-49-54-690Z_01a015fe-7c82-7000-8b72-e721a5d7b2a4.jsonl`
- `omp-client.log` is the matching OMP transport log copied from `snappy`:
  `~/.omp/logs/omp.2026-08-18.92852.log`
- SHA-256 of `omp-session.jsonl`:
  `f024ad940f4ea81769243dd1167f45ee9498559ea599c99d47ec5e3df6c7cadf`
- SHA-256 of `omp-client.log`:
  `dcdd9c1658657cce3879dc789c6f0b35f639dfbdc98e85c41f4cf127bbf17536`

The source and copied checksums matched on 2026-08-19 before the server
configuration changed.

## Pre-change system state

- Host: `smarty`
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB
- GPU UUID: `GPU-a71210ca-e14a-755a-88bb-77f53a2102f6`
- Driver: NVIDIA open kernel module 595.84
- Kernel: Ubuntu `7.0.0-29-generic`
- Runtime power limit: 450 W; default: 600 W
- llama.cpp binary: build 10480, commit `01818e495`
- llama.cpp source checkout at capture: `6d05498314db1b57f81c271080018aa2d0b89be9`
- Model: `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL`
- Context: 262,144
- Parallel slots: 1
- KV cache: q8_0
- Flash attention: enabled
- Context shift: disabled
- MTP: enabled, maximum draft depth 3
- Reasoning effort: medium

The service was the only owner of TCP port 2053. OMP had stopped its active
turn before capture. The model endpoint was healthy and the GPU was idle.

## Failure signature

Every failure had the same kernel and CUDA signature:

```text
NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked! Notify Timeout Seconds: 7
NVRM: Xid (PCI:0000:01:00): 8, name=llama-server, channel 0x0000000d
CUDA error: the launch timed out and was terminated
ggml_backend_cuda_synchronize -> cudaStreamSynchronize
```

No kernel OOM or ECC error was present. The service aborted, systemd waited 15
seconds, and then the GPU and endpoint recovered without a host reboot.

## Event timeline

| Local time (PDT) | Approximate prompt state | Last generation sample | Notes |
|---|---:|---:|---|
| 2026-08-18 23:02:12 | approximately 65K tokens | 3,363 tokens, 101.36 tok/s | Initial failure at the default 600 W limit |
| 2026-08-18 23:47:08 | approximately 116K tokens | 4,818 tokens, 75.97 tok/s | 450 W limit; 3-second rate 60.98 tok/s |
| 2026-08-18 23:48:49 | 116,456-token replay | 917 tokens, 75.58 tok/s | Fresh server, full prompt prefill, then decode failure |
| 2026-08-18 23:50:38 | 116,456-token replay | 1,384 tokens, 76.13 tok/s | Fresh server, full prompt prefill, then decode failure |

The monitored temperature did not exceed 78 C. GPU utilization was about
94-100%, the power limit remained 450 W, and the fan was about 47%. The 450 W
limit reduced noise but did not prevent the lock.

## OMP session shape

- 8 user messages
- 37 assistant messages
- 36 tool calls and 36 tool results
- 115,185 characters of tool-result text
- 160,402 characters of assistant reasoning
- 44,488 characters of tool arguments
- 8,183 characters of assistant-facing text
- 1,061 characters of user text
- Largest tool result: 51,203 characters; it completed successfully
- Last tool result before the repeated failure: 1,199 characters
- OMP context before the failing turn: 115,973 cached tokens
- OMP auto-compaction threshold: 222,823 tokens

The content was ordinary text and source code. It contained no image or binary
blocks. The failing turn was preparing a self-contained interactive HTML graph.
It failed during reasoning, before it issued the planned write tool call.

OMP recorded socket-close errors at 23:02:12, 23:47:08, 23:48:50, and
23:50:38. The intermediate attempts at 23:47:25 and 23:49:07 received
`503 Loading model` while systemd was restarting llama-server.

## Current interpretation

The evidence does not support context exhaustion, concurrent GPU inference,
thermal protection, a 600 W transient, or one malformed tool result:

- The longest prompt was less than half the configured context window.
- llama.cpp had one slot. Multiple tool calls happened while inference was not
  running, and the separate OMP title request was serialized.
- The same prompt failed at different decode positions after clean server
  restarts.
- Three failures occurred with a 450 W limit and a maximum observed
  temperature of 78 C.
- Two 450 W failures occurred only 85 and 93 seconds after fresh llama-server
  process starts. Each process did one full prompt prefill and then failed
  after 917 or 1,384 decoded tokens.

The leading trigger envelope is long-context Qwen 3.8 decode with MTP,
CUDA-graph reuse, and llama.cpp prompt-cache/checkpoint state on GB202. The
driver/GSP stops processing work and reports Xid 8; llama.cpp detects the
failure when it synchronizes the CUDA stream.

## Controlled test sequence

1. Disable only MTP and replay this OMP session.
2. If the failure repeats, keep MTP disabled and test
   `GGML_CUDA_DISABLE_GRAPHS=1`.
3. If the failure repeats, also test `--cache-ram 0 --ctx-checkpoints 0`.
4. Separately test earlier OMP compaction, for example at 64K-96K tokens.

For each replay, record prompt tokens, prefill speed, generated tokens, decode
speed, temperature, power, and any Xid. Do not combine parameter changes in one
run.

## MTP-off test state

At 2026-08-19 00:08 PDT, `qwen-3.8-27b/model.json` changed only
`llama.mtp` from `true` to `false`. The managed service was stopped and started
with `ktxsvc`.

- Old PID: 603211
- New PID: 609163
- Port 2053 owner: new llama-server PID 609163
- Health endpoint: healthy
- Launch arguments: no `--spec-type` or `--spec-draft-n-max`
- Runtime power limit: still 450 W
- Idle VRAM with MTP: 30,236 MiB
- Idle VRAM without MTP: 28,116 MiB
- Measured MTP overhead: 2,120 MiB
- Tiny MTP-off canary: 17 prompt tokens and 16 generated reasoning tokens at
  69.94 tok/s; no CUDA or kernel error

The MTP context shares the target model weights. In this llama.cpp build it
inherits the full context length and one sequence, but its memory filter keeps
full-length F16 draft KV only for the embedded NextN/MTP layer(s). The main
context keeps the trunk attention KV and recurrent state. Drafted tokens are a
small temporary frontier; the larger overhead is the separate MTP context,
NextN compute state, graph allocations, and its F16 draft KV.

The preserved artifacts passed high-confidence checks for common OpenAI,
GitHub, AWS, private-key, bearer-token, and JWT secret formats before commit.
`gitleaks` was not installed.

## MTP-off replay result

The original OMP session resumed at 2026-08-19 00:12 PDT and finished its
task without a service or GPU failure.

- Initial prompt: 116,485 tokens
- Initial prefill: 59.79 seconds, 1,948.33 tok/s
- First generation: 6,299 tokens, 44.91 tok/s
- Second long generation: 5,746 tokens, 43.11 tok/s
- Completed inference turns: 11
- Total generated tokens: 18,294
- Final recorded context: 140,442 tokens
- Maximum observed temperature: 78 C
- Power limit: 450 W
- Kernel Xid events: none
- CUDA errors: none
- Final endpoint state: healthy and idle

The MTP-on full-prompt replays took approximately 66.6-66.8 seconds to prefill
116,456 tokens. MTP-off prefill was approximately 10% faster because the MTP
context did not also process the prompt. MTP-off long-context decode was about
41-45 tok/s, compared with approximately 75-76 tok/s for the directly
comparable MTP-on attempts. The cost of this successful run was therefore
approximately 40-45% lower decode throughput.

This result shows that the 116K prompt alone is not sufficient to trigger the
failure. It strongly implicates the MTP execution path, but disabling MTP also
removed its 2,120 MiB context allocation. It does not yet distinguish an MTP
state-management defect from a depth-dependent draft/verify defect.

The next clean isolation test is MTP with `mtp_n_max` set to 1. That retains
essentially the same full-length MTP context and draft KV allocation while
reducing each speculative frontier to one token and each target verification
batch to at most two positions. Interpretation:

- Depth 1 fails: suspect the common MTP context, KV/checkpoint, recurrent-state,
  or rollback path.
- Depth 1 succeeds and depth 3 fails: suspect multi-token drafting, the larger
  target verification graph, or rollback after partial multi-token acceptance.
- To test capacity separately, keep depth 3 and reduce only the MTP draft KV
  footprint, for example with q8 draft K/V or a smaller controlled context.

Static VRAM exhaustion is unlikely: the MTP-on service had approximately 35 GB
of free VRAM, and the failures produced no CUDA allocation, OOM, or ECC error.
A memory-state correctness bug remains possible.

## Resume reference

The original OMP RPC process used this session path on `snappy`:

```text
/Users/francip/.omp/agent/sessions/-src/2026-08-18T17-49-54-690Z_01a015fe-7c82-7000-8b72-e721a5d7b2a4.jsonl
```

Use the preserved `omp-session.jsonl` to restore the known state if the live
OMP session changes.

## Direct replay harness

The managed service stays stopped during experiments. `repro.sh` launches
`llama-server` directly on port 2053, derives a fixture ending at the first
failed assistant record, and uses OMP's `/retry` operation. The replay uses
the original `high` thinking level, all three GooeyPi extensions, and `yolo`
approval mode so tool results immediately feed the next inference turn.

```bash
./investigations/2026-08-19-qwen-3.8-27b-xid8/repro.sh mtp1
./investigations/2026-08-19-qwen-3.8-27b-xid8/repro.sh mtp2
./investigations/2026-08-19-qwen-3.8-27b-xid8/repro.sh mtp3
./investigations/2026-08-19-qwen-3.8-27b-xid8/repro.sh dflash
```

The harness refuses to run if the managed service is active, port 2053 is in
use, or the existing GPU limit does not match `EXPECTED_POWER_LIMIT_W` (450 by
default). For a deliberate 600 W comparison, use
`EXPECTED_POWER_LIMIT_W=600 ./repro.sh mtp3`. Each run gets raw server output,
one-second GPU telemetry, kernel logs, metadata, and a small OMP control log
under `runs/`. It stops its direct server on exit and does not restart the
managed service.

## Speculative decoding comparison

The corrected yolo replay did not reproduce Xid 8 in one pass for any tested
configuration. This fixture is a realistic stress workload, not a
deterministic crash trigger. A passing run cannot establish that a mode is
safe; repeated trials are required.

| Mode | Initial prompt | Initial output | Initial decode | Initial acceptance | Final context | Result |
|---|---:|---:|---:|---:|---:|---|
| MTP depth 3 | 112,739 | 3,688 | 79.08 tok/s | 63.7% | 140,534 | completed, no Xid |
| MTP depth 2 | 112,739 | 5,183 | 75.49 tok/s | 74.8% | 136,539 | completed, no Xid |
| DFlash v1 bootstrap | 112,739 | 4,555 | 56.52 tok/s | 33.6% | 140,999 | completed, no Xid |

The generation paths differ because sampling is not deterministic. Do not
compare total turn counts or final context as if they were matched benchmark
outputs. The initial request shape was matched, and the full agent workflow
was allowed to finish.

- MTP depth 3: 21 inference turns, 20,463 generated tokens, maximum 78 C.
- MTP depth 2: 15 inference turns, 20,945 generated tokens, maximum 78 C.
- DFlash v1: 12 inference turns, 24,541 generated tokens, maximum 80 C.

One-second telemetry observed 95-100% utilization and brief sampled power
readings above the configured 450 W limit in all modes. No run logged a kernel
Xid, CUDA error, ECC error, or thermal shutdown.

## 600 W MTP depth-3 replay

A depth-3 replay at the GPU's default 600 W limit completed without an Xid or
CUDA error. OMP produced a different valid generation path, so this is a
stress comparison and not a deterministic benchmark.

| Limit | Initial prompt | Initial prefill | Initial output | Initial decode | Initial acceptance | Result |
|---:|---:|---:|---:|---:|---:|---|
| 450 W | 112,739 | 1,808.48 tok/s | 3,688 | 79.08 tok/s | 63.7% | completed, no Xid |
| 600 W | 108,265 | 2,286.90 tok/s | 3,010 | 87.65 tok/s | 64.1% | completed, no Xid |

The 600 W run had 26.5% higher initial prefill throughput and 10.8% higher
initial decode throughput. Later cached turns reached 92.35 tok/s. Do not use
the different total run times as a speed comparison because the generated
agent paths and tool calls differed.

- Completed inference turns: 13
- Total generated tokens: 17,965
- Final recorded context: 139,929 tokens
- Maximum observed temperature: 89 C
- Maximum observed fan speed: 53%
- Maximum sampled power: 603.66 W
- Busy telemetry samples: 325 of 351 seconds
- Samples at 95-100% utilization: 162 seconds; longest consecutive sequence:
  14 seconds
- Kernel Xid, CUDA, ECC, or thermal events: none

This successful pass had more sustained sampled load than the 450 W depth-3
replay: 87.2% average utilization instead of 55.2%, with fewer and shorter tool
gaps. It also ran hotter. This result does not support a simple cumulative busy
time or temperature threshold. A new llama-server process does not power-cycle
the GPU or fully reset all driver and GSP state, so a retained marginal device
state is still possible. The stronger current hypothesis is a specific
MTP/CUDA graph or state sequence on GB202.

During the run, `clocks_throttle_reasons.active` reported `0x4`. This is the
software power-cap reason, not a thermal slowdown reason. Driver 595.84 also
reported impossible temperature limit fields through `nvidia-smi -q`, including
a 6 C target limit and negative shutdown and slowdown limits. Treat those limit
fields as bad driver telemetry; the sampled GPU temperature and kernel event
log remained coherent.

## Repeated 600 W heat-soak result

Two more depth-3 passes ran back-to-back at 600 W. The second pass started one
second after the first pass ended, with a 77 C starting temperature. Both
completed without an Xid, CUDA error, or thermal event.

| Run | Turns | Generated tokens | Final context | Busy samples | Maximum temperature | Maximum fan |
|---|---:|---:|---:|---:|---:|---:|
| Initial 600 W pass | 13 | 17,965 | 139,929 | 325 s | 89 C | 53% |
| Heat-soak pass 2 | 24 | 26,710 | 146,422 | 421 s | 90 C | 61% |
| Heat-soak pass 3 | 28 | 30,793 | 157,381 | 439 s | 91 C | 74% |

Across the three 600 W passes, MTP depth 3 completed 65 inference turns and
generated 75,468 tokens during 1,185 sampled busy seconds. The final pass ran
hotter and longer than any failed attempt. At 90 C, the driver reported no
hardware or software thermal slowdown.

The successful logs contain `non-consecutive token position` warnings during
prompt-cache restoration. The final pass logged 30 of these warnings. They are
evidence of cache-state transitions, but they are not sufficient to cause the
lock.

This heat-soak test strongly reduces the probability of context length,
temperature, or cumulative high-utilization time as the primary trigger. A
rare driver fault remains possible. The best current target is a specific
MTP, CUDA graph, prompt-cache, or checkpoint state transition that this
stochastic replay did not select.

After the heat-soak test, the runtime power limit was restored to 450 W and the
managed service was temporarily returned to MTP depth 3. A later controlled
replay superseded that production setting; see below.

MTP depth 1 was tested before the approval-mode mismatch was found. It
completed two long turns totaling 10,982 output tokens at 67.32 tok/s, but it
stopped at a write approval rather than completing the yolo workflow. Treat
that result as supporting data, not as a matched run.

The DFlash candidate is pinned at Hub revision
`3c89ca499fa04f89a0b4b5ca9b5867953261db39`; its Q8_0 GGUF SHA-256 is
`5a5b354855ffa4cc2aed92297fe3fcc696039b27b533168a4ae458305c2b1b84`.
It is a community bootstrap/transplant, not DFlash2 and not an official
Qwen3.8-trained drafter. Its observed 33.6% first-turn acceptance closely
matches its publisher's approximately 34% overall claim. It used about
1.5 GiB more GPU memory than MTP in this shared-workload snapshot.

## Magicapp reproducible session

A later OMP session from `~/src/magicapp` reproduced the original fault at
16:50:18 PDT with MTP depth 3 and the 450 W limit. It is preserved under
`magicapp-20260819T165018/` with its matching client log, checksums, replayed
sessions, service logs, kernel events, and one-second GPU telemetry.

The exact preserved input was replayed through OMP's `/retry` operation using
the managed service, yolo approval mode, `medium` thinking, one llama.cpp slot,
and a fresh server process for each depth.

| MTP depth | First decode | Completed turns | Generated tokens | Last context | Peak temp | Result |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 80.78 tok/s | 26 | 16,367 | 115,787 | 80 C | Xid 8 |
| 2 | 82.44 tok/s | 60 | 28,205 | 127,726 | 84 C | Xid 8 |
| 1 | 71.52 tok/s | 82 | 42,032 | 146,887 | 86 C | completed |

Depth 2 changed the stochastic trajectory and ran longer, but did not solve
the fault. Depth 1 was 11.5% slower than depth 3 on the matched first turn and
completed the full workflow. Because the passing run was longer and hotter
than either failure, the result further weakens context capacity, temperature,
and cumulative load as primary explanations. It implicates a depth-dependent
multi-token draft/verify, rollback, CUDA-graph, or cache-state transition,
although one successful depth-1 run cannot establish long-term safety.

After the depth-only tests, the managed service was temporarily left at MTP
depth 1 and the persistent 450 W limit before the CUDA graph isolation below.

## CUDA graph isolation

The managed service was then given the runtime environment variable
`GGML_CUDA_DISABLE_GRAPHS=1`; no llama.cpp rebuild was needed. The exact
Magicapp fixture was replayed at MTP depths 1 and 3 with the same 450 W cap.

| MTP depth | Duration | Completed turns | Generated tokens | Last main context | Peak temp | Result |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 15 min | 47 | 33,400 | 150,289 | 86 C | client timeout, no Xid |
| 3 | 15 min | 88 | 47,402 | 179,075 | 87 C | client timeout, no Xid |

Both clients reached their configured time limit while the server remained
healthy. The graph-disabled depth-3 run lasted more than three times as long
as the graph-enabled depth-3 failure and exceeded its last context by 63,288
tokens. It also ran beyond the depth-2 failure by 51,349 tokens. Initial
depth-3 decode was 84.18 tok/s without CUDA graphs versus 80.78 tok/s in the
graph-enabled failure, so this workload showed no material speed penalty.

This is the strongest controlled evidence in the investigation. It points to
CUDA graph capture or replay interacting with multi-token MTP, cache
restoration, or long-context graph-shape transitions on GB202. The result is
not proof of permanent stability because the two graph-disabled workflows
timed out rather than completing and generation paths are stochastic.

The shared llama.cpp launcher now disables CUDA graphs by default, with an
explicit per-model `llama.cuda_graphs: true` opt-in. The final managed state is
MTP depth 3, CUDA graphs disabled through that launcher default, and the
persistent 450 W limit active. The endpoint is healthy.
