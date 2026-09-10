# Smarty GPU recovery, 10 September 2026

Both GPUs recovered after Franci powered the machine down and started it again.
All nine checked service endpoints returned HTTP 200. TTS reported ready. A real
request to the 27B model returned the correct arithmetic answer. A separate CUDA
check, pinned by UUID to the RTX PRO 6000, completed and used less than 9 KiB of
tensor memory. The final service and process snapshots are in this directory.

The GPU-only attempt did not reach the reset command. The first managed stop,
for ComfyUI, timed out. The restoration attempt also timed out. Franci then
authorized a direct kill of the exact ComfyUI process. Its remaining kernel-bound
thread did not exit after SIGKILL. No other managed service or greeter was stopped
by the procedure. Neither GPU received a reset command.

Franci rebooted the host. The 6000 telemetry returned, but the 4090 reported an
unknown error and NVIDIA UVM reported global fatal error 0x60. CUDA still failed
when pinned to the 6000. A later full shutdown and startup recovered both cards.
The intermediate warm-boot observations were terminal diagnostics; they were
not captured as a separate raw file. Do not treat this report as a full kernel
incident dump or proof of the initial fault's cause.

No experimental workload ran on either GPU before recovery. The 4090 was not
used for an experiment. Its existing services resumed through normal boot.
The cold-boot snapshot contains the same six healthy 6000 services as the original
baseline, and the GDM greeter is active. No service configuration changed.

Evidence: `baseline.jsonl`, `reset.jsonl`, `reset.stderr`,
`direct-kill.json`, `after-cold-boot.jsonl`, `cuda-after-cold-boot.json`, and
`restored-services.json`. The baseline and reset files are unchanged raw output.

The scoped recovery script remains an operator-authorized attempt, not a fix for
this driver fault. NVIDIA documents that GPU reset requires clients to release
the device and that failed recovery can require a power cycle:
[NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
