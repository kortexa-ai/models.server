# NVIDIA submission draft — Xid 8 GPU lock on RTX PRO 6000 (GB202) under sustained compute

Ready to paste into a developer-forum post or support ticket. Attach:
the freshest `nvidia-bug-report.log.gz` from `captures/` (auto-collected
immediately after an event) and optionally STATUS.md.

---

**Title:** Recurring Xid 8 (RC watchdog "GPU probably locked") on RTX PRO
6000 Blackwell Workstation under sustained compute — both 595.84 and
610.43.02-open, two independent CUDA stacks

**System:** RTX PRO 6000 Blackwell Workstation Edition 96 GB (GB202),
Ubuntu 24.04, kernel 7.0.0-30-generic (HWE), Secure Boot on,
Canonical-signed open kernel modules (GSP), currently driver
610.43.02-open (previously 595.84 — occurs on both). Persistent power
limit 450 W (default 600 W — occurs at both). Host: Raptor Lake desktop
platform.

**Signature (identical in every event):**
```
NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked!  Notify Timeout Seconds: 7
NVRM: Xid (PCI:0000:01:00): 8, pid=<compute pid>, name=<llama-server|python3>, channel <varies>
```
Application receives `CUDA error: the launch timed out and was
terminated` (cudaErrorLaunchTimeout). The GPU recovers in ~15 s without
reboot; no persistent damage or data corruption observed.

**Occurrence pattern:** 13 events since 2026-08-18 across two unrelated
workloads: (a) llama.cpp long-context decode with multi-token prediction
and CUDA graphs; (b) pure eager-mode PyTorch FP32 training of 4B/9B-class
transformers (no CUDA graphs, no torch.compile). Under sustained
saturation the rate reaches roughly one event per 20-45 minutes. The most
reliable reproducer we have is (b): dense FP32 training keeping the GPU
at ~100% for tens of minutes.

**Isolation already done:**
- Disabling CUDA graphs (GGML_CUDA_DISABLE_GRAPHS=1) stabilized workload
  (a) with no recurrence since 2026-08-19 — but (b) never used CUDA
  graphs and still locks, so graphs look like an amplifier (submission
  density), not the cause.
- Driver 595.84 → 610.43.02-open: no change in signature or rate.
- ECC volatile+aggregate zero; row remapper zero, none pending.
- Not thermal (≤ 87 C observed at failure; also fails at 51 C ambient
  card temp minutes after idle), not context/VRAM capacity (fails with
  > 55 GB free), not power-limit-specific (600 W and 450 W both fail).
- Same-prompt replays fail at different decode positions after clean
  restarts → stochastic, not input-deterministic.

- Control experiment: 26,912 steps (5.35 h) of the identical FP32 training
  stack on the RTX 4090 (AD102) in the SAME host under the SAME driver
  (610.43.02) produced zero events, while the GB202 locked five times in
  the same evening under comparable sustained load. The fault isolates to
  the GB202 card.

**Ask:** Is this a known GSP-RM / channel-scheduling issue on GB202
under sustained high-density kernel submission? Fresh
`nvidia-bug-report.log.gz` captured within a minute of an event is
attached; we can reproduce within an hour on request, run instrumented
drivers/firmware, or collect additional traces.
