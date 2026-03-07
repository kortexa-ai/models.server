# Model Serving Infrastructure

This directory contains model configurations and launch scripts for locally-hosted LLMs. Each model gets its own subdirectory with:

- `model.json` — model metadata (name, id, port, HuggingFace model ID)
- `run.sh` — starts the inference server
- `setup.sh` — prepares whatever dependencies that model needs

## Shared Setup

Run `./setup.sh` from `models.server/` for the common environment:

- macOS: creates/updates a shared `.venv` with `mlx-vlm` and `mlx-lm`
- Linux: checks that `llama-server` is available for the llama-backed models

The Qwen 3.5 family now delegates to this shared setup. `nanbeige-4.1-3b` and
`lfm-2.5-1.2b-thinking` also reuse the shared macOS environment, but keep their
own local `vLLM` env on Linux because that part is still model-specific.

## Serving Backends

We use three inference backends depending on the model architecture and hardware:

### llama-server (llama.cpp)

GGUF-quantized models served via [llama.cpp](https://github.com/ggerganov/llama.cpp). Works on both macOS (Metal) and Linux (CUDA). Serves an OpenAI-compatible API at `/v1/chat/completions`.

**Used by:** GLM 4.7 Flash, GPT OSS 20B/120B, LFM-2 24B A2B

**Pros:** Universal — runs on any hardware, wide model support, mature.
**Cons:** Slower generation on Apple Silicon vs native MLX (~22 tok/s vs ~50 tok/s on M4 Pro for Qwen 3.5 35B).

### mlx-vlm

Vision Language Models served via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). macOS only (Apple Silicon / MLX). Uses `mlx-community/` quantized models from HuggingFace. Serves at `/chat/completions` (no `/v1` prefix — see [API routing](#api-routing) below).

**Used by:** Qwen 3.5 family (0.8B, 2B, 4B, 9B, 27B, 27B Opus, 35B-A3B)

**Pros:** ~2x faster generation on Apple Silicon vs llama-server GGUF. Native MLX acceleration.
**Cons:** macOS only. Requires `torch` + `torchvision` as dependencies (for the video processor). Currently needs two sed patches in setup.sh (see [Patches](#mlx-vlm-patches)).

### mlx-lm

Text-only LLMs served via [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm). macOS only. Similar to mlx-vlm but for non-vision models. Serves at `/v1/chat/completions`.

**Used by:** Nanbeige 4.1 3B

### vLLM

GPU-accelerated serving via [vLLM](https://github.com/vllm-project/vllm). Linux only (CUDA). Used as the Linux counterpart for models that use mlx-lm/mlx-vlm on macOS.

**Used by:** Nanbeige 4.1 3B (Linux), Qwen 3.5 family (Linux)

Spark-specific Docker experiments and benchmark artifacts live in
`vllm-spark/`.

## Choosing a Backend

```
                    macOS (Apple Silicon)         Linux (NVIDIA GPU)
                    ─────────────────────         ──────────────────
Vision models       mlx-vlm                       vLLM
Text-only models    mlx-lm                        vLLM
GGUF models         llama-server (Metal)           llama-server (CUDA)
```

**Rule of thumb:** If an `mlx-community/` quantization exists, prefer mlx-vlm/mlx-lm on macOS for best performance. Fall back to llama-server with GGUF if no MLX quant is available.

## Benchmark: mlx-vlm vs llama-server

Tested on Mac Mini M4 Pro with Qwen 3.5 35B-A3B, 256 output tokens:

| Backend | Quant | Generation tok/s | Time |
|---------|-------|----------------:|-----:|
| mlx-vlm | 4-bit (MLX) | **~50** | 5.1s |
| llama-server | UD-Q4_K_XL (GGUF) | ~22 | 11.3s |

Prompt processing is faster on llama-server (~113 tok/s vs ~40 tok/s), but generation speed dominates user-perceived latency.

## Model Inventory

| Model | ID | Port | macOS Backend | HF Model |
|-------|-----|------|---------------|----------|
| Qwen 3.5 0.8B | `qwen-3.5-0.8b` | 2031 | mlx-vlm | `mlx-community/Qwen3.5-0.8B-MLX-4bit` |
| Qwen 3.5 2B | `qwen-3.5-2b` | 2030 | mlx-vlm | `mlx-community/Qwen3.5-2B-MLX-4bit` |
| Qwen 3.5 4B | `qwen-3.5-4b` | 2029 | mlx-vlm | `mlx-community/Qwen3.5-4B-MLX-4bit` |
| Qwen 3.5 9B | `qwen-3.5-9b` | 2024 | mlx-vlm | `mlx-community/Qwen3.5-9B-MLX-4bit` |
| Qwen 3.5 27B | `qwen-3.5-27b` | 2026 | mlx-vlm | `mlx-community/Qwen3.5-27B-4bit` |
| Qwen 3.5 27B Opus | `qwen-3.5-27b-opus` | 2032 | mlx-vlm | `mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit` |
| Qwen 3.5 35B A3B | `qwen-3.5-35b-a3b` | 2027 | mlx-vlm | `mlx-community/Qwen3.5-35B-A3B-4bit` |
| Nanbeige 4.1 3B | `nanbeige4.1-3b` | 2025 | mlx-lm | `Nanbeige/Nanbeige4.1-3B` |
| GLM 4.7 Flash | `glm-4.7-flash` | 2021 | llama-server | `unsloth/GLM-4.7-Flash-GGUF` |
| GPT OSS 20B | `gpt-oss-20b` | 2020 | llama-server | `unsloth/gpt-oss-20b-GGUF` |
| GPT OSS 120B | `gpt-oss-120b` | 2023 | llama-server | `unsloth/gpt-oss-120b-GGUF` |
| LFM 2.5 1.2B | `lfm-2.5-1.2b-thinking` | 2022 | mlx-lm | `LiquidAI/LFM2.5-1.2B-Thinking` |
| LFM-2 24B A2B | `lfm-2-24b-a2b` | 2028 | llama-server | `LiquidAI/LFM2-24B-A2B-GGUF` |
| Penumbra | `penumbra` | 4007 | custom | `karpathy/nanochat_d32` |

## API Routing

The api.server (`src/routes/ai.ts`) routes requests with `X-Kortexa-Provider: kortexa` to local model servers:

1. Looks up the model by ID in `model.json` files
2. Swaps `req.body.model` to the `hf_model` value (so mlx-vlm gets the HF repo ID it expects)
3. Constructs the base URL:
   - **mlx-vlm models** (`hf_model` starts with `mlx-community/`): `http://{host}:{port}` (no `/v1`)
   - **All other models**: `http://{host}:{port}/v1`
4. Forwards extra body params (e.g. `enable_thinking`) directly in the request body for kortexa provider

## mlx-vlm Patches

The `setup.sh` for mlx-vlm models applies two sed patches after installation. These are needed because mlx-vlm doesn't forward extra request body parameters (like `enable_thinking`) to the Jinja chat template.

**Upstream PR:** https://github.com/Blaizzy/mlx-vlm/pull/784

The patches:
1. **server.py** — Forward `request.__pydantic_extra__` (extra body params) to `apply_chat_template()`
2. **prompt_utils.py** — Forward `**kwargs` from `apply_chat_template()` to `get_chat_template()`

Once the PR is merged, these patches can be removed and we can switch from `git+https://github.com/Blaizzy/mlx-vlm.git` to a pinned release version.

## Future Work

- **mlx-vlm `/v1` prefix:** Submit a PR to add a `--prefix` or `--root-path` CLI argument to mlx-vlm so it can serve at `/v1/chat/completions` natively, eliminating the need for special-cased URL routing in the api.server.
- **Shared venvs:** The three Qwen 3.5 models each create their own `.venv` with identical dependencies. Could share a single venv to save disk space and setup time.
- **mlx-lm for text-only Qwen:** If Qwen 3.5 models are used without vision features, mlx-lm could be a lighter alternative to mlx-vlm (no torch/torchvision dependency). However, Qwen 3.5 is architecturally a VLM, so mlx-vlm is the correct backend.

## Service Management (launchd / systemd)

Each model can include service unit files for automatic management:

```
<model-id>/
├── launchd/
│   ├── ai.kortexa.<model-id>.plist       # macOS launchd job
│   └── kortexa-<model-id>.sh             # shell wrapper (just execs ../run.sh)
└── systemd/
    └── kortexa-ai-llm-<model-id>.service # Linux systemd unit
```

### macOS (launchd)

Install and manage:

```bash
# Install (symlink into LaunchAgents)
ln -s ~/src/models.server/<model-id>/launchd/ai.kortexa.<model-id>.plist ~/Library/LaunchAgents/

# Load / start
launchctl load ~/Library/LaunchAgents/ai.kortexa.<model-id>.plist
launchctl start ai.kortexa.<model-id>

# Stop / unload
launchctl stop ai.kortexa.<model-id>
launchctl unload ~/Library/LaunchAgents/ai.kortexa.<model-id>.plist

# View logs
tail -f ~/Library/Logs/<model-id>.log
```

Configuration notes:
- `RunAtLoad` is `false` — models are started on demand, not at login
- `KeepAlive` is `false` — if the process exits, it stays down
- `ThrottleInterval` is 15s — prevents rapid restart loops
- Logs go to `~/Library/Logs/<model-id>.log`

### Linux (systemd)

Install and manage:

```bash
# Install (symlink into systemd user directory, or copy to /etc/systemd/system/)
sudo ln -s ~/src/models.server/<model-id>/systemd/kortexa-ai-llm-<model-id>.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start / stop
sudo systemctl start kortexa-ai-llm-<model-id>
sudo systemctl stop kortexa-ai-llm-<model-id>

# Enable at boot
sudo systemctl enable kortexa-ai-llm-<model-id>

# View logs
journalctl -u kortexa-ai-llm-<model-id> -f
```

Configuration notes:
- `Restart=always` with `RestartSec=15` — auto-restarts on crash
- `StartLimitBurst=3` within 120s — gives up after 3 rapid failures
- Security hardening: `ProtectSystem=full`, `NoNewPrivileges=true`, `PrivateTmp=true`
- `ReadWritePaths` grants access to the project directory and HuggingFace cache

## Adding a New Model

1. Create a directory under `models.server/` with the model's slug
2. Add `model.json` with name, id, port (pick an unused one), host, and hf_model
3. Add `run.sh` — use an existing model as a template (e.g. `qwen-3.5-9b` for mlx-vlm, `nanbeige-4.1-3b` for mlx-lm, `glm-4.7-flash` for llama-server)
4. Add `setup.sh` if the model needs a Python venv
5. Add `launchd/` and `systemd/` directories with service units (copy from an existing model and update the model ID, description, and paths)
6. Run `./setup.sh && ./run.sh` to test
7. The api.server will auto-discover the model from `model.json`
