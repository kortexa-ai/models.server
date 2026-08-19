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

## Resume reference

The original OMP RPC process used this session path on `snappy`:

```text
/Users/francip/.omp/agent/sessions/-src/2026-08-18T17-49-54-690Z_01a015fe-7c82-7000-8b72-e721a5d7b2a4.jsonl
```

Use the preserved `omp-session.jsonl` to restore the known state if the live
OMP session changes.
