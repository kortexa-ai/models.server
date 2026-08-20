# Magicapp managed-service replay

This directory preserves the OMP session that triggered a Qwen 3.8 27B
`llama-server` Xid 8 on `smarty` at 2026-08-19 16:50:18 PDT. The client was
the only active OMP session under `~/src/magicapp` on `snappy`.

## Preserved input

- Source session:
  `/Users/francip/.omp/agent/sessions/-src-magicapp/2026-08-19T23-45-56-166Z_01a01c6a-cbc6-7000-b4d7-c18ed2ccd9c3.jsonl`
- Source client log: `/Users/francip/.omp/logs/omp.2026-08-19.55015.log`
- `omp-session.jsonl` SHA-256:
  `2570600738ad744b4adca283a9dc1d272ac665bd7db21bd526ed9068d0f3b3e3`
- `omp-client.log` SHA-256:
  `94ad88c6ebd18b8c4841ddf795bff6364a3217a9691f50b0516977e8ec993e1f`

The source and copied checksums matched before the replays. The original
failure logged the same RC-watchdog, Xid 8, CUDA launch-timeout, and
`cudaStreamSynchronize` signature as the earlier investigation. It occurred
at a 450 W limit with MTP depth 3. The request began at approximately 82,233
context tokens and failed after about 500 generated tokens, well below the
262,144-token model context.

## Replay method and result

Each run copied the preserved session to a disposable path on `snappy`, then
resumed it through OMP with `medium` thinking, yolo approval mode, and
`/retry`. The managed Qwen service was freshly restarted at the selected MTP
depth. All runs used one llama.cpp slot and the persistent 450 W limit.

| MTP depth | First output | First decode | Acceptance | Completed turns | Generated | Last context | Peak temp | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 2,840 | 80.78 tok/s | 53.3% | 26 | 16,367 | 115,787 | 80 C | Xid 8 |
| 2 | 2,126 | 82.44 tok/s | 69.0% | 60 | 28,205 | 127,726 | 84 C | Xid 8 |
| 1 | 2,921 | 71.52 tok/s | 77.4% | 82 | 42,032 | 146,887 | 86 C | completed |

The generated paths are stochastic, so turn counts and failure positions are
not matched benchmarks. The matched first turn shows depth 1 was 11.5% slower
than depth 3 and 13.2% slower than depth 2. In exchange, it completed the full
known-bad workflow and exceeded both higher-depth failure contexts.

The depth-3 and depth-2 metadata record OMP status 130 and `client-error`
because the operator interrupted OMP after the kernel Xid was confirmed; the
GPU failure preceded that interruption. Their `kernel.log` and `service.log`
files preserve the fault. The depth-1 run exited normally with status 0.

## Interpretation

This fixture reproduces the fault with speculative depths 2 and 3 after fresh
service starts, but not with depth 1 in one complete run. That argues against
static KV capacity, total context length, temperature, cumulative service
uptime, and depth-3-only behavior. It instead points toward a depth-dependent
multi-token draft/verify, rollback, CUDA-graph, or cache-state transition on
GB202. A single passing depth-1 run is encouraging, not proof that it can never
fail.

After these depth-only tests, the managed service was temporarily left at MTP
depth 1 and 450 W before the CUDA graph isolation below.

## CUDA graphs disabled

The same fixture was replayed again after adding
`GGML_CUDA_DISABLE_GRAPHS=1` to the managed service environment. This is a
runtime switch; llama.cpp was not rebuilt. Both replays used the persistent
450 W limit and stopped at the configured 15-minute client limit without an
Xid.

| MTP depth | First prompt | First output | First decode | Turns | Generated | Last main context | Peak temp | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 79,254 | 1,813 | 74.06 tok/s | 47 | 33,400 | 150,289 | 86 C | timeout, no Xid |
| 3 | 83,728 | 2,361 | 84.18 tok/s | 88 | 47,402 | 179,075 | 87 C | timeout, no Xid |

The graph-disabled depth-3 run lasted more than three times as long as its
graph-enabled failure, completed 62 more inference turns, and reached 63,288
tokens beyond that run's last completed context. It also exceeded the
graph-enabled depth-2 failure by 51,349 context tokens.

No performance penalty was visible in the matched initial requests. The
graph-disabled depth-3 first turn ran at 84.18 tok/s versus 80.78 tok/s with
graphs enabled, and depth 1 ran at 74.06 versus 71.52 tok/s. The stochastic
outputs and differing prompt lengths make those stress comparisons rather
than precise benchmarks, but they rule out a large expected slowdown in this
workload.

This A/B result strongly implicates CUDA graph capture or replay interacting
with multi-token MTP, cache restoration, or changing long-context graph shapes
on GB202. It does not prove the workaround is permanently safe: neither
graph-disabled branch completed before the client time limit, and rare driver
faults remain possible.

The managed service now remains at MTP depth 3 with CUDA graphs disabled and
the 450 W cap active.
