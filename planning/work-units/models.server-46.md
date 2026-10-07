# GLM-5.3 Flash EXL3 on Shock

Issue: https://github.com/kortexa-ai/models.server/issues/46

Reproduce the sxuff single-GB10 recipe with zoo and API ID
`glm-5.3-flash-exl3`. Qwen TensorFold runs independently on Static.

The recipe is pinned to `c25551cd1492a93f2e564ce1b706d821a959b91e`, TensorFold
0.6.5 to `609ca419abecebdc5a059498a613680bd3aa847f`, and the intact
`turboderp/GLM-5.3-Flash-exl3` checkpoint to
`51058cd551c7e570d87bd32a4adee720edce2349`. Both recipe patches are applied.
All 24 checkpoint files match their Hub hashes. A public pinned ARM64 NVIDIA
PyTorch 25.11 image replaces the recipe's unpublished custom container and
host imports; the patched standard CLI provides text-only serving.

The selected profile is a 262144-token window, BF16 latent KV, MTP-1, one
active request and a 3-GiB budget for saved prompts. A 259985-token cold
prompt plus 256 generated tokens completes at 28.19 tok/s, with 803.78-second
TTFT; short-to-64K decode measures about 31–33 tok/s. A 512K allocation and
short canaries also pass, but only about 0.36 GiB remains for saved prompts.
Native 1M is refused by the 100-GiB CUDA admission cap.

Validation includes 26 loader/transport checks, 31 selected packed-arithmetic
and row-invariance checks on GB10, strict arithmetic and generated-code
canaries, MTP/serial token equality, A/B/A prompt reuse, serialized concurrent
requests, tool calls/results, thinking budgets, OpenAI Responses, Anthropic,
and image/context refusals. Twenty focused repository tests pass.

The experiment uses `/home/francip/tmp/glm53-gb10-20261007` before promotion
to ignored `.engines/tensorfold-exl3/glm-5.3-flash-exl3`. Runtime delivery uses
Git and `ktxsvc` on Shock; the launcher never installs or builds at startup
and stops only its own labeled container. Deployment readbacks belong in the
issue. Evidence is retained under
`~/Desktop/ai reports/glm53-shock-20261007/` on Snappy.
