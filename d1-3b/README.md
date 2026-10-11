# Liquid d1 3B

[LiquidAI/d1-3B](https://huggingface.co/LiquidAI/d1-3B) is a 3.12B-parameter
decision model for text, JSON and images. It returns typed probabilities
through `POST /v1/systemone`, with zero generated tokens. It supports `noul`
(yes/no), `choice` (named options) and `score` (ordered levels) questions.

The roster ID is `d1-3b`, on port **2071**. The configuration uses the
[official Q8_0 GGUF](https://huggingface.co/LiquidAI/d1-3B-GGUF), its Q8_0
vision projector, q8_0 KV, a 32,768-token context and one request slot.
This is the open 3B checkpoint; Liquid's hosted `d1` API is a separate offering.

## Setup and launch

Run from the repository root:

```bash
./scripts/setup-llama-runtime.sh d1-3b
./run.sh d1-3b
```

Setup builds the upstream llama.cpp revision pinned in `model.json` into
the ignored `.engines/llama-d1` directory, using Metal on macOS or CUDA on
supported Linux hosts. The pin includes native LFM2 d1 decision support.
The shared `llama-server` and Shingi's Prism runtime are separate.
Startup downloads the weights to the Hugging Face cache and the pinned
projector to the llama.cpp cache; it does not build or install a runtime.
There is no MLX or vLLM serving recipe.

Before launch, inspect `ktxsvc` and port ownership, and check available
memory. The Linux service definition selects Smarty's RTX PRO 6000 UUID;
follow the GPU borrowing procedure before using that shared GPU.
The launchd and systemd definitions are available to `ktxsvc` for later
installation. Registration does not install a service or change any default.

## Decision request

```bash
curl --fail-with-body http://127.0.0.1:2071/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "My package arrived with a broken screen. Please replace it.",
    "questions": {
      "damaged": {
        "type": "noul",
        "instructions": "Does the customer report a damaged item?"
      },
      "route": {
        "type": "choice",
        "instructions": "Which team should handle the request?",
        "criteria": {"returns": "Damaged items and replacements", "billing": "Payments"}
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is the request?",
        "criteria": ["Routine", "Time sensitive", "Emergency"]
      }
    }
  }'
```

Responses contain `answers` keyed by question ID and `usage.output_tokens: 0`.
For images, add an `images` array of base64 data URLs
(`data:image/png;base64,...`); `state` may be `null` for an image-only request.
Use the decision endpoint rather than chat completions. `GET /health` and
`GET /v1/models` provide readiness and model discovery.

Protocol reference: [llama.cpp System One API](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/tools/server/README.md#post-v1systemone-typesafe-compatible-system-one-api).
