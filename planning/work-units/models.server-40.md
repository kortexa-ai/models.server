# models.server #40 — Single-Spark TensorFold Qwen3.8 Flash Next experiment

Issue: <https://github.com/kortexa-ai/models.server/issues/40>

## Scope and decisions

- Add `qwen-3.8-flash-next-fast` as a separate text-only model ID on static port 2067. Keep the two-Spark NVFP4 model at port 2066 for rollback and side-by-side selection.
- Test MiaAI Lab's TensorFold recipe at its pinned repository commit, TensorFold v0.3.6.2, `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`, five streams, 262,144 context, int8 KV, PLE on SSD, six MTP drafts at 0.60 confidence, and recipe-default thinking.
- Assess the current NVIDIA NVFP4 checkpoint against TensorFold's supported formats. Do not convert or overwrite model files.
- The initial TensorFold v0.3.6.2 pin was text-only. The later v0.3.6.3 follow-up adds the MiaAI vision patch and image/video input; audio remains unsupported.

## Planning and validation

- Factory surface preflight returned `alternate-plan`. The claim covers this note, the fast model config, runner, systemd unit, README, and model-config test.
- The existing #39 claim was active under this same Agent Bus root and includes the dual-Spark runtime. Its fence was not available, so no forced expiry or scope mutation was attempted. This experiment has a separate issue and fresh claim.
- The fast recipe is deployed and measured on static. The candidate is available on port 2067; the original dual-Spark service remains a rollback option and is currently stopped/uninstalled.

## Trial results — 2026-09-29

- Pinned MiaAI recipe: `4cd99569f00f84b397cee77900bec9cdf43287bd`; TensorFold v0.3.6.2 (`71377a5373ed7b394f1b480ba2a6a3986b03af1c`); downloaded checkpoint `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` (MLX 4-bit, group 32). Startup completed in 225.9 seconds. TensorFold reported a 102.60 GiB startup estimate within 105.14 GiB, five 262,144-token streams and int8 KV. Service is installed and active on static at port 2067; `/v1/models` returned `qwen-3.8-flash-next-fast`.
- All requests were sequential and uncached (`cached: 0`). Prefill rates below use server-reported prefill time and actual prompt token count. Decode rates use generated tokens divided by server-reported decode time.

| Prompt tokens | Completion | Prefill | Prefill tok/s | Decode | Decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 7,922 | 16 | 3.53 s | 2,245 | 0.16 s | 98 |
| 31,540 | 16 | 12.35 s | 2,554 | 0.17 s | 95 |
| 125,914 | 16 | 56.55 s | 2,226 | 0.29 s | 54 |
| 60 | 64 | 0.38 s | — | 1.17 s | 55 |
| 67 | 2,048 | 0.19 s | — | 41.37 s | 49.5 |
| 240,117 | 2,048 | 118.75 s | 2,022 | 42.01 s | 48.7 |

The largest request used 240,117 prompt tokens plus 2,048 generated tokens (242,165 total, within the 262,144 context ceiling), taking 161.2 seconds end to end. A first attempt used a one-sentence summary instruction and ended at 177 tokens; the recorded largest result above is the corrected run that generated the full 2,048-token cap.

- Static control service reports `ai.kortexa.llm-qwen-3.8-flash-next-fast` running and healthy. Smarty API config includes `http://192.168.2.101:4100`, and a direct request from smarty to that control endpoint returned the fast service as running. The fast model config is therefore eligible for the API's control-backed catalog discovery.
- Direct TensorFold NVFP4 serving is unsupported in v0.3.6.2: its `tensorfold info` classified the existing checkpoint as `modelopt` and found no compatible Qwen3.8 engine. Its Qwen3.8 engine accepts MLX affine 4-bit group-32 or EXL3.
- The old two-Spark model config and weights remain intact for rollback. Its managed services are uninstalled on static and shock; reinstall worker on shock before head on static. Both were verified stopped before starting this trial.
- Focused config tests (9/9), JSON validation, runner `bash -n`, parser, and `git diff --check` passed.


## Vision-enabled follow-up — 2026-09-29

- Updated the fast model to MiaAI recipe commit `856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e` and TensorFold v0.3.6.3. `scripts/prepare.sh` pulled the published image tag `v0.3.6.3-5f313914582d` (Docker digest `sha256:05620ace4085baa58e1e4049cd6e9512c5c54959d32744019766672a5d2c824a`). The image is labeled with MiaAI patch hash `5f313914582d`; the existing checkpoint was reused. TensorFold startup reported vision enabled with a 0.84 GiB tower and startup estimate 102.50 GiB within 105.34 GiB. It loaded and served in 141.1 seconds.
- The model config now explicitly enables `multimodal` and vision, while keeping public URL fetching disabled. The recipe accepts image/video data URLs; audio is unsupported. A small PNG smoke request succeeded and returned “The dominant color in the image is red.”
- Re-ran single, sequential text requests with a 1,024-token completion cap. Both reached the full cap and were uncached:

| Prompt tokens | Completion | Prefill | Prefill tok/s | Decode | Decode tok/s | Total |
|---:|---:|---:|---:|---:|---:|---:|
| 7,951 | 1,024 | 3.20 s | 2,486 | 20.22 s | 50.7 | 23.44 s |
| 31,569 | 1,024 | 12.69 s | 2,488 | 20.18 s | 50.7 | 32.93 s |

- Deployment commit: `d57211b`; tests, JSON validation, parser, runner syntax, and `git diff --check` passed.
