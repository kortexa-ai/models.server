# Xid 8 during pure PyTorch training — no llama.cpp resident (2026-08-31)

New occurrence, materially different workload state from the original report.

## Kernel log

```
[Mon Aug 31 20:29:49 2026] pcieport 0000:00:06.0:    [ 0] RxErr                  (First)
[Mon Aug 31 20:32:14 2026] NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked!  Notify Timeout Seconds: 7
[Mon Aug 31 20:32:14 2026] NVRM: Xid (PCI:0000:01:00): 8, pid=908552, name=python3, channel 0x00000005
```

Driver 595.84 (unchanged). A corrected PCIe RxErr on 0000:00:06.0 landed
~2.5 min before the lock — first time a PCIe event has been observed near an
Xid 8 in this investigation.

## Workload state at the lock

- LegoLM PA-SCALE-9B part-1 calibration (issue kortexa-ai/legolm#81), cell
  `pa cap100-lr1p5e-4`: Qwen3.5-9B FP32, PyTorch, pa_s0.py validation_nll
  after training step 512 (pa_s0.py:743, torch.AcceleratorError:
  cudaErrorLaunchTimeout).
- **All six safe-list services were STOPPED** under smarty_gpu_block.sh
  (--until-free 85): no llama-server / MTP / CUDA-graph / prompt-cache code
  resident on the GPU. GPU 0 held only the training process.
- Several prior cells of the same shape (same model, same script, PA
  cap050 and LoRA/prefix cells) completed PASS in the same block before
  this one.

## Recovery

GPU recovered without reboot; block restore PASS, all endpoints 200
(comfyui, alt-image-gen, qwen-3.8-27b, lfm2.5-vl-3b, hy-mt2-7b, vision).
Retry of the failed cell per the campaign retry-once rule was in progress
at the time of writing.

## Read

This weakens the "llama.cpp MTP / CUDA-graph / prompt-cache state
transition" leading hypothesis as the sole trigger: the signature fires
under a pure PyTorch FP32 training load with llama.cpp absent. Either the
trigger is lower (GB202 + driver 595.84 under sustained heavy compute), or
there are two paths to the same lock. The preceding PCIe RxErr is worth
correlating against past occurrences (check dmesg history for RxErr near
earlier Xid 8 events).

## Second occurrence, same session (added ~20:50 PDT)

```
[Mon Aug 31 20:44:41 2026] NVRM: Xid (PCI:0000:01:00): 8, pid=925689, name=python3, channel 0x00000005
```

Killed the generated-LoRA LR 5e-5 calibration cell after step 256 — a
different cell, different process, same signature. **Two Xid 8 locks in
12.5 minutes under pure PyTorch FP32 9B training**, llama.cpp absent both
times. Between them, the retried PA cap100 cell and two other PA cells ran
to PASS, so the trigger remains stochastic, but the rate under this
workload is far above anything previously observed (prior occurrences were
days apart under llama-server).

## PCIe RxErr correction

The RxErr cluster (20:27:54-20:29:49, four events) is on root port
0000:00:06.0, whose LnkCap is 16GT/s x4 - not the GPU's port (the 6000 is
x16 at 01:00.0; its LnkSta reads 2.5GT/s at idle, which is normal ASPM
downspeed, LnkCap 32GT/s x16). The RxErr breadcrumb is therefore probably
an unrelated x4 device (NVMe?) and should NOT be read as GPU-link evidence
without mapping the child device of 00:06.0 first.

## Campaign impact

PA-SCALE-9B part-1 calibration: all four PA cells PASS, LoRA sweep aborted
on its first cell, prefix sweep not reached. The campaign's single
authorized block reopen was already spent; partial artifacts committed by
the Sol. Rerun of the remaining ~5 cells needs either a quiet-GPU window
with acceptance of the Xid 8 risk, or resolution here first.

## Triage session (2026-08-31 ~21:00 PDT, Fable)

Franci directed GPU triage before further 9B GPU time. Findings, all
non-disruptive (production serving throughout):

1. **A third PyTorch occurrence predates today.** Aug 29 16:06:26, Xid 8,
   pid 2481386 `python3`, channel 0x2d — during the PA-SCALE-4B campaign.
   Xid 8 has now fired under: llama.cpp MTP long decode (Aug 18-19),
   PyTorch 4B training (Aug 29), PyTorch 9B training (Aug 31, twice).
   Two independent CUDA stacks, three model scales. The MTP-off replay
   success stands, but MTP is best re-read as a high-rate trigger
   workload, not the root cause. The trigger envelope is now: GB202 +
   driver/GSP 595.84 under sustained high-utilization compute, stochastic,
   recovers ~15 s, no reboot needed.
2. **Memory hardware is clean.** ECC volatile+aggregate all zero (SRAM and
   DRAM), row remapper zero on every counter, none pending, no remap
   failure. Not an RMA signature on this evidence.
3. **PCIe RxErr is the NVMe, not the GPU.** Root port 0000:00:06.0 (Raptor
   Lake PCIe 4.0 port, 16GT/s x4) fronts bus 02 = Samsung S4LV008 NVMe.
   The corrected errors cluster near training because checkpoint/weight IO
   coincides with GPU load. Unrelated to Xid 8; worth its own low-priority
   eye (corrected physical-layer errors on a disk link).
4. **Driver lever exists.** Installed: 595.84 (open kernel module, GSP
   595.84). apt offers the nvidia-driver-610 branch (also 590/595 metas).
   Power limit still 450 W runtime / 600 W default.

### Proposed next step (registered, awaiting Franci)

Upgrade to the 610 branch in a maintenance window, then run the five held
PA-SCALE-9B calibration cells as the acceptance test — they are the
hottest known reproducer (~3 events across ~2 evenings of training).
Constraint: the driver touches BOTH GPUs and wants a reboot; the window
must wait until the 4090 (hermes-livekit, in active use) is free. If Xid 8
recurs on 610, the driver is exonerated and the next suspects are VBIOS /
platform BIOS / GSP interaction — escalate to NVIDIA with this file.

## Mitigation state correction + eager-mode finding (2026-08-31, from Franci)

Franci reports the production qwen-3.8-27b mitigation that actually shipped
after this investigation was **disabling CUDA graphs** (README step 2), and
prod has been stable since — MTP itself was retained. Meanwhile the LegoLM
training harness (`tracks/packet-adapters/pa_s0.py`) that produced the
Aug 29 and Aug 31 Xid 8 events is **pure eager-mode PyTorch: no
torch.compile, no CUDA graph capture, no cudnn autotune** (verified by
source grep).

Read together: Xid 8 fires with CUDA graphs entirely absent from the
process, so graph replay is a **trigger amplifier, not the mechanism**.
The graphs-off llama.cpp fix and the eager-mode training crashes are
consistent with a GB202 GSP/driver 595.84 defect in channel scheduling
under sustained high-rate kernel submission — graphs raise submission
density for inference; FP32 9B training is dense by nature.

Candidate submission-throttling knobs for a training-side mitigation
experiment (untested, listed for the record): `CUDA_DEVICE_MAX_CONNECTIONS=1`,
periodic `torch.cuda.synchronize()` every N steps, `CUDA_LAUNCH_BLOCKING=1`
(costly). The primary lever remains the 610 driver branch; this eager-mode
evidence belongs in any NVIDIA escalation.

## Driver upgrade executed (2026-08-31 21:03 PDT, Franci-authorized)

595.84 → **610.43.02-open** (nvidia-driver-610-open + Canonical-signed
linux-modules-nvidia-610-open for kernel 7.0.0-30-generic; Secure Boot
intact, no DKMS). 54 pending general updates applied in the same window.
Clean reboot: both GPUs up on 610.43.02, all production endpoints 200
within a minute, persistent 450 W limit survived, zero NVRM events in the
fresh kernel log. GSP firmware now from the 610 branch.

Acceptance test: the five remaining PA-SCALE-9B calibration cells (two
generated-LoRA, three prefix) rerun under the new driver — the hottest
known reproducer. Zero Xid across them supports the driver hypothesis;
a recurrence exonerates 595.84 specifically and escalates to NVIDIA with
this file (GB202, both driver branches, two CUDA stacks, eager mode).

## Xid 8 RECURS under driver 610.43.02 (2026-08-31 21:43:44 PDT)

```
[Mon Aug 31 21:43:44 2026] NVRM: Xid (PCI:0000:01:00): 8, pid=25595, name=python3, channel 0x00000002
```

During the acceptance run itself: the prefix LR 5e-4 calibration cell,
~40 min into the block, after all three generated-LoRA cells and the
prefix 5e-5 cell completed clean. The registered retry-once rule engaged;
outcome of the retry recorded below by the campaign.

**Driver 595.84 is exonerated as the specific cause.** The signature now
spans: two driver branches (595.84, 610.43.02), two CUDA stacks
(llama.cpp, eager PyTorch), four workloads (27B GGUF decode, 4B / 9B FP32
training), ECC and row remapper clean throughout, recovery ~15 s every
time. Remaining suspect set: GB202 driver-family defect present in both
branches (GSP channel scheduling under sustained high-density submission),
VBIOS, board-level electrical (PCIe signal integrity or power transients
at the 450 W cap — note the cap predates all PyTorch occurrences; a 600 W
default-limit test run is an untested variable for the training workload),
or genuine silicon. This is now NVIDIA-escalation material; this file plus
the README constitute the dossier. Incidence anecdote: 2 locks in ~12.5
min under 595.84 that evening vs 1 lock in ~40 min of comparable load
under 610 — too few events to claim a rate change.

## Occurrence ledger under 610.43.02 (running)

1. 2026-08-31 21:43:44 PDT — part-1 acceptance, prefix 5e-4 cell, step 384+; retry passed.
2. 2026-08-31 22:27:07 PDT — part-2 canonical, seed 42 LoRA train attempt 1 (pid 52604, channel 0x2); registered restart engaged, outcome in campaign integrity log.
3. 2026-08-31 22:47:34 PDT — part-2 canonical, seed 1042 PA train attempt 1 (pid 66483, channel 0x2); restart engaged. Spacing under 610: 44 min, then 20 min — no clear rate improvement over 595.84; channel 0x2 in all three.
