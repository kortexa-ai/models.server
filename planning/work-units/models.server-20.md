# Isolated RTX PRO 6000 recovery

Issue: https://github.com/kortexa-ai/models.server/issues/20

Franci requested a GPU-only recovery on Smarty during hamster-orchestra
Experiment 105. The target is UUID `GPU-a71210ca-e14a-755a-88bb-77f53a2102f6`.
The 4090, its inference services, and host uptime must be preserved.

Before any mutation, the 6000 returned ERR/N/A telemetry, CUDA initialization
failed, and the kernel reported a locked GPU. The PCI function advertises FLR.
The known 6000 users are six managed services and the GDM login greeter.
No graphical user session was active. The existing service HTTP health replies
did not establish GPU health.

`scripts/recover-smarty-6000.py` captures the exact live set, stops it through
`ktxsvc`, temporarily stops the unused greeter, attempts a UUID-specific reset,
and restores the captured set in `finally`. It never escalates to a bus reset,
shared driver unload, 4090 reset, or host reboot. CUDA arithmetic and real model
inference are separate checks from process and HTTP health.

The attempted managed ComfyUI stop and restoration timed out before any reset.
An explicitly authorized direct SIGKILL did not release its remaining thread.
Franci performed host recovery. Both GPUs, the CUDA arithmetic check, all nine
service endpoints, and actual 27B inference passed after the full shutdown and
startup. See `results/smarty-6000-recovery-20260910/REPORT.md` for the outcome and
the preserved evidence. No configuration change was needed.
