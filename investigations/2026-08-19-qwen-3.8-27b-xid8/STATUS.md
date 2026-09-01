# Xid 8 on smarty RTX PRO 6000 — consolidated status

Living document. Last consolidation: 2026-08-31 ~23:05 PDT (Fable).
Detail files: `README.md` (llama.cpp arc, Aug 18-19, controlled MTP/graph
isolation), `2026-08-31-legolm-9b-pytorch-xid8.md` (PyTorch arc + triage +
driver-610 acceptance), `magicapp-20260819T165018/`, `runs/`.

## TL;DR

A stochastic GPU lock (`NVRM: Xid 8`, RC-watchdog "GPU probably locked",
`cudaErrorLaunchTimeout`) fires on the GB202 RTX PRO 6000 under sustained
heavy compute. It spans two driver branches (595.84, 610.43.02-open), two
CUDA stacks (llama.cpp, eager-mode PyTorch), and model scales 4B-27B. The
GPU recovers in ~15 s without reboot; no artifact corruption has ever been
observed. Memory hardware is clean. Disabling CUDA graphs stabilized
production llama-server; PyTorch training (which never used graphs) still
locks. Working hypothesis: a GSP-RM channel-scheduling defect (or GB202
erratum surfacing through it) under sustained high-density kernel
submission. NVIDIA escalation is the actionable next step.

## Occurrence ledger (all known events)

| # | When (PDT) | Driver | Process | Workload | Channel | Outcome |
|--:|---|---|---|---|---|---|
| 1 | 2026-08-18 23:02:12 | 595.84 | llama-server | Qwen3.8-27B MTP-3 decode, ~65K ctx, 600 W | 0xd | service abort, auto-recover |
| 2 | 2026-08-18 23:47:08 | 595.84 | llama-server | same session, ~116K ctx, 450 W | 0xd | same |
| 3 | 2026-08-18 23:48:49 | 595.84 | llama-server | 116K replay, fresh server | 0xd | same |
| 4 | 2026-08-18 23:50:38 | 595.84 | llama-server | 116K replay, fresh server | 0xd | same |
| 5 | 2026-08-19 16:50:18 | 595.84 | llama-server | magicapp session, MTP-3, 450 W | — | preserved as fixture |
| 6 | 2026-08-19 eve | 595.84 | llama-server | magicapp replay, MTP depth 3 | — | Xid 8 |
| 7 | 2026-08-19 eve | 595.84 | llama-server | magicapp replay, MTP depth 2 | — | Xid 8 |
| 8 | 2026-08-29 16:06:26 | 595.84 | python3 | PA-SCALE-4B FP32 training (eager) | 0x2d | absorbed unnoticed by campaign |
| 9 | 2026-08-31 20:32:14 | 595.84 | python3 | 9B FP32 calibration, PA cell, validation | 0x5 | cell FAIL, retry passed |
| 10 | 2026-08-31 20:44:41 | 595.84 | python3 | 9B FP32 calibration, LoRA cell, step 256 | 0x5 | campaign abort (budget spent) |
| 11 | 2026-08-31 21:43:44 | 610.43.02 | python3 | 9B calibration, prefix cell, step 384+ | 0x2 | retry passed |
| 12 | 2026-08-31 22:27:07 | 610.43.02 | python3 | 9B canonical, s42 LoRA train | 0x2 | restart passed |
| 13 | 2026-08-31 22:47:34 | 610.43.02 | python3 | 9B canonical, s1042 PA train | 0x2 | restart passed |

Notable non-events: MTP-off 18K-token replay (Aug 19), two 15-min
graphs-off replays at depth 1 and 3 (Aug 19), MTP depth-1 replay
completing 42K tokens, the entire PW0-S/PB-S0 0.8B program, and most 4B
cells — passing runs prove nothing (stochastic), but the rate clearly
scales with sustained GPU saturation.

## Environment

GB202 RTX PRO 6000 Blackwell Workstation 96 GB, board serial
1794225038906; Raptor Lake host, kernel 7.0.0-30-generic; Secure Boot on,
Canonical-signed open kernel modules (GSP mandatory on this
card/driver combo — GSP-off is not testable). Driver 595.84 through
2026-08-31 21:00, then 610.43.02-open. Persistent 450 W limit (default
600 W). ECC volatile+aggregate zero, row remapper zero (checked
2026-08-31). PCIe: GPU x16 (2.5 GT/s idle downspeed is normal ASPM); the
RxErr corrections on port 00:06.0 belong to a Samsung NVMe, not the GPU.

## Ruled out

Thermal (locks at ≤ 87 C, heat-soak passed), OOM/context capacity (failed
at half the window; passed far beyond failing contexts), power transient
as sole cause (fails at both 600 W and 450 W), ECC/VRAM degradation
(counters clean), llama.cpp-specific code (eager PyTorch locks too),
CUDA graphs as the mechanism (PyTorch never used them), driver 595.84
specifically (610 locks too), our workload's correctness (two independent
stacks, all controls clean, artifacts always intact after recovery).

## Working hypothesis (ranked)

1. **GSP-RM channel-scheduling defect on GB202 under sustained
   high-density kernel submission** — present in both driver branches'
   firmware lineage. Fits: workload-agnostic beyond "keeps the GPU
   saturated", graphs act as an amplifier (denser submission) rather than
   the cause, clean 15 s recovery via RC, no memory-error correlates,
   channel id varies (just allocation order).
2. GB202 silicon/board erratum surfacing through the same path (a
   marginal unit that stalls without ECC-visible corruption). Not
   excludable with host-side tools; distinguishable only by NVIDIA or by
   a second identical card behaving differently.
3. Host-platform interaction (Raptor Lake PCIe/power) — weak; no AER on
   the GPU port, no host-side instability.

## Active mitigations and rules

- Production llama-server: CUDA graphs disabled by launcher default
  (stable since 2026-08-19), MTP depth 3 retained, 450 W cap.
- LegoLM campaigns: registered Xid-8 retry-once-per-cell (calibration) /
  restart-once-per-run (canonical; second hit invalidates that seed pair)
  — five events absorbed with zero lost cells as of consolidation.
- Untested lever for training-side rate reduction (registered, unused):
  `CUDA_DEVICE_MAX_CONNECTIONS=1` / periodic `torch.cuda.synchronize()`.

## Next steps

1. **NVIDIA escalation** with this dossier; capture `nvidia-bug-report.sh`
   immediately after the next event to attach fresh GSP logs.
2. Keep the per-event ledger current (campaign runners log
   `XID8 event=...` lines automatically; consolidate here periodically).
3. If a run of quiet weeks follows a future driver release, re-run the
   hottest reproducer (9B FP32 training cells) as acceptance before
   trusting it.

## AD102 control burn (registered 2026-08-31 23:35 PDT, before data)

Franci granted both smarty GPUs for the night. Control experiment: loop
the 0.8B FP32 PA pilot training cell on the RTX 4090 (AD102, same host,
same driver 610.43.02, same PyTorch stack) until ~05:00 PDT with the
4090 trio (asr, tts, lfm2.5-8b-a1b) stopped and process-owned restore.
The 0.8B profile is submission-density-heavy (small kernels at high
rate) — arguably a hotter trigger profile than 9B per the leading
hypothesis. Interpretation registered in advance: GB202 locking while
AD102 stays clean over ~5 h of saturation isolates the fault to
GB202 silicon/GSP; an AD102 lock would instead implicate the shared
driver/stack/host and materially change the NVIDIA report.

Ledger event 14: 2026-08-31 23:27:50 PDT — 9B canonical, s2042 prefix
train attempt 1 (pid 116380, channel 0x2, driver 610); restart engaged.
Post-event nvidia-bug-report captured 23:28:10 (20 s after the lock):
captures/nvidia-bug-report-postevent-20260831T232810.gz.
