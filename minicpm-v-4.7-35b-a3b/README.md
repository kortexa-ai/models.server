# MiniCPM-V 4.7 35B-A3B

OpenBMB's vision-language MoE on port **2072**, served through llama.cpp on
Metal or CUDA. The 262,144-token unified KV pool is shared by eight slots;
concurrent requests must fit collectively. The advertised per-request
context is also 262,144 tokens.

The [official GGUF repository](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf)
does not publish `UD-Q4_K_XL`, so this entry uses `Q4_K_M` (21.17 GB),
q8_0 KV, and the default F16 vision projector (1.12 GB, 16x downsampling).
The projector URL is explicit because the repository also contains a
different 4x projector. Memory use includes KV and runtime overhead in
addition to these weights.

MiniCPM-V 4.7 requires canvas M-RoPE support from
[llama.cpp PR #29416](https://github.com/ggml-org/llama.cpp/pull/29416).
The manifest pins that merged upstream revision in an isolated runtime;
the launcher refuses to fall back to a shared binary if it is missing.
See the publisher's [deployment guide](https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp/minicpm-v4_7_llamacpp.md).

From the repository root, prepare the runtime explicitly before launch:

```bash
./scripts/setup-llama-runtime.sh minicpm-v-4.7-35b-a3b
./run.sh minicpm-v-4.7-35b-a3b
```

First launch downloads the model and projector. Sampling defaults follow
the publisher's GGUF examples: temperature 0.7, top-p 0.8, top-k 100, and
repeat penalty 1.05. The chat template enables thinking by default; append
`--reasoning off` to the launch command for plain replies. Requests use
`/v1/chat/completions` with model `minicpm-v-4.7-35b-a3b`; image input uses
OpenAI-style `image_url` content parts.

The service definitions are registration only. Check service, GPU and port
ownership before starting it. Adding this entry does not install a service,
download the weights, or select a new default model. Runtime inference has
not been validated by the roster addition.
