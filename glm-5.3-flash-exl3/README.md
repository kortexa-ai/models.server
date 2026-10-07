# GLM-5.3 Flash EXL3 on GB10

Source recipe: [sxuff/glm53-flash-single-gb10](https://github.com/sxuff/glm53-flash-single-gb10),
pinned to [`c25551c`](https://github.com/sxuff/glm53-flash-single-gb10/tree/c25551cd1492a93f2e564ce1b706d821a959b91e).
This entry targets **Shock**, while Qwen TensorFold runs on Static.

The recipe's two patches add single-GPU loading for the original packed
[turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3)
checkpoint: 2.05 average bits per weight, about 85.2 GB of download files.
Weights are pinned to `51058cd551c7e570d87bd32a4adee720edce2349` and TensorFold
0.6.5 to `609ca419abecebdc5a059498a613680bd3aa847f`.

The upstream custom server imports unpublished host modules. Our reproduction
uses its patched standard `tensorfold serve` command in the public ARM64
NVIDIA PyTorch 25.11 container (CUDA 13), pinned by image digest. This serving
path is text only; the upstream recipe's custom vision server is a separate
path. No checkpoint conversion is needed.

The profile serves `glm-5.3-flash-exl3` at `http://192.168.2.102:2069/v1`:
262144 tokens including output, a fixed BF16 latent live cache, MTP-1 and one
active decode lane. Queued requests can reuse up to eight saved prompts within
a 3-GiB budget. The 1M window advertised by the checkpoint does not fit the 100-GiB
CUDA admission budget; 512K leaves only about 0.36 GiB for saved prompts.

Measured on Shock: about 31–33 tok/s for short-to-64K prompts and 28.2 tok/s
at 259985 prompt tokens. That cold long prompt took 804 seconds to its first
token. A 512K allocation and short canaries also passed; full prefill was
tested through 259985 tokens, with 256 generated tokens.

On Shock, with Git, Docker/NVIDIA Container Toolkit, `uv` and Python 3.12:

```bash
cd ~/src/models.server
./scripts/setup-tensorfold-exl3.sh glm-5.3-flash-exl3
ktxsvc install glm-5.3-flash-exl3  # enables and starts the service
ktxsvc status glm-5.3-flash-exl3
ktxsvc stop glm-5.3-flash-exl3
ktxsvc start glm-5.3-flash-exl3
```

Setup verifies all checkpoint hashes and records an immutable runtime image
ID. Startup checks the receipt and file sizes/timestamps, requires 110 GiB
available, and refuses another GPU workload or listener. Files stay under
ignored `.engines/tensorfold-exl3/glm-5.3-flash-exl3`; Docker layers stay in
Docker's image store. To retire it, stop and uninstall this exact `ktxsvc`
service before archiving or removing its runtime directory.

Thinking defaults on with a native 3000-token budget. Use
`"chat_template_kwargs": {"enable_thinking": false}` to disable it per request.
The upstream custom server's soft-landing/repetition guard is not used here.
For a fresh serial reference, send `"draft": false`. The standard GLM CLI
does not forward `"prefix_cache": false` to its engine.
