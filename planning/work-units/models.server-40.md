# models.server #40 — Single-Spark TensorFold Qwen3.8 Flash Next experiment

Issue: <https://github.com/kortexa-ai/models.server/issues/40>

## Scope and decisions

- Add `qwen-3.8-flash-next-fast` as a separate text-only model ID on static port 2067. Keep the two-Spark NVFP4 model at port 2066 for rollback and side-by-side selection.
- Test MiaAI Lab's TensorFold recipe at its pinned repository commit, TensorFold v0.3.6.2, `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`, five streams, 262,144 context, int8 KV, PLE on SSD, six MTP drafts at 0.60 confidence, and recipe-default thinking.
- Assess the current NVIDIA NVFP4 checkpoint against TensorFold's supported formats. Do not convert or overwrite model files.
- TensorFold v0.3.6.2 rejects image, audio, and video input; mark this candidate text-only.

## Planning and validation

- Factory surface preflight returned `alternate-plan`. The claim covers this note, the fast model config, runner, systemd unit, README, and model-config test.
- The existing #39 claim was active under this same Agent Bus root and includes the dual-Spark runtime. Its fence was not available, so no forced expiry or scope mutation was attempted. This experiment has a separate issue and fresh claim.
- Validation, machine deployment, speed measurements, and runtime decision will be added after the trial.
