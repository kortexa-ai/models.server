# Model Serving Infrastructure

Local model serving across multiple machines. Each model gets its own directory with configuration; shared engine scripts handle the actual launching.

## Quick Start

```bash
# Setup (once per machine)
./setup.sh

# Run a model
./run.sh qwen-3.5-4b                    # from root
cd qwen-3.5-4b && ../run.sh             # from model dir
./run.sh gemma-4-26b-a4b --engine vllm  # override engine
./run.sh qwen3-tts-0.6b-customvoice     # MLX-Audio on macOS, vLLM-Omni on CUDA
```

## Machines

| Host/IP | Hardware | Memory | OS | Primary Backend |
|---------|----------|------------|------|-----------------|
| **smarty** | RTX PRO 6000 Blackwell | 96 GB VRAM | Ubuntu Linux | `llama-server`, vLLM, vLLM-Omni, SGLang-Omni |
| **snappy** | Mac Mini M4 Pro | 64 GB unified | macOS | `mlx-vlm`, `mlx-lm`, `mlx-audio` |
| **scrappy** | RTX 3070 Laptop | 8 GB VRAM | Windows 11 | — |
| **sparky** | DGX Spark GB10 | 128 GB unified | Ubuntu Linux | offline |
| **192.168.2.144** | Raspberry Pi 5 | 8 GB RAM | ARM Linux | `llama-server` CPU |
| **192.168.2.145** | Raspberry Pi 5 | 8 GB RAM | ARM Linux | `llama-server` CPU |

### Persistent GPU Power Limit

`smarty` applies a 450 W limit to GPU 0 at boot with the host-local
`/etc/systemd/system/nvidia-power-limit.service`. The oneshot unit runs
`nvidia-smi --id=0 --power-limit=450` and is enabled for `multi-user.target`.
It deliberately lives outside this repository so `ktxsvc` does not discover it
as a managed application service.

The power-limit unit's relevant settings are:

```ini
[Unit]
Wants=nvidia-persistenced.service
After=systemd-modules-load.service nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi --id=0 --power-limit=450
RemainAfterExit=yes
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Recreate this independent unit and run the following after rebuilding the
host. Model units intentionally have no dependency or drop-in for the cap, so
they remain identical to their repository definitions.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-power-limit.service
systemctl status nvidia-power-limit.service
nvidia-smi --query-gpu=power.limit --format=csv,noheader
```

The expected production reading is always `450.00 W`. The card's 600 W
factory limit is only for an explicitly authorized temporary experiment, and
the experiment must restore 450 W when it exits. Do not restore a previously
observed 600 W value: treat it as configuration drift.

## Model Inventory

| Port | Model | Type | Quant | KV Cache | Context | Parallel |
|------|-------|------|-------|----------|---------|----------|
| 2025 | Qwen 3.5 9B | agentic dense | UD-Q4_K_XL / MLX 4-bit | q8_0 | 262K | 1 |
| 2026 | LFM2.5 1.2B Thinking | reasoning dense | Q8_0 / MLX 8-bit | q8_0 | 32K | 1 |
| 2027 | LFM2.5 2.6B | agentic dense | Q8_0 / MLX 8-bit | q8_0 | 128K | 1 |
| 2028 | Qwen 3.6 35B A3B | agentic MoE | UD-Q4_K_XL / MLX 4-bit | q8_0 | 262K | 1 |
| 2029 | Qwen 3.5 4B | small dense | UD-Q4_K_XL | q8_0 | 64K | 2 |
| 2030 | Qwen 3.5 2B | small dense | Q8_0 | q8_0 | 32K | 2 |
| 2031 | Qwen 3.5 0.8B | small dense | Q8_0 | q8_0 | 32K | 2 |
| 2032 | Qwen 3.6 27B | big dense | UD-Q4_K_XL / FP8 | q8_0 / fp8 | 262K | 1 / 4 |
| 2033 | Qwen3 TTS 0.6B CustomVoice | streaming TTS | BF16 / MLX BF16 | — | 4K | 1 |
| 2034 | Audio8 TTS Preview 0.6B | cloning TTS | BF16 / MLX BF16 | — | 2K | 1 |
| 2035 | Qwen3 TTS 1.7B CustomVoice | streaming controlled TTS | BF16 / MLX BF16 | — | 4K | 1 |
| 2036 | Gemma 4 26B-A4B | agentic MoE | UD-Q4_K_XL / MLX 4-bit | q8_0 | 262K | 1 |
| 2037 | Gemma 4 31B | big dense | UD-Q4_K_XL | q8_0 | 64K | 2 |
| 2038 | Gemma 4 E4B | small dense | UD-Q4_K_XL | q8_0 | 64K | 2 |
| 2039 | Gemma 4 E2B | small dense | Q8_0 | q8_0 | 32K | 2 |
| 2040 | Qwen3 Embedding 0.6B | embedding | Q8_0 | q8_0 | 32K | 4 |
| 2041 | EmbeddingGemma 300M | embedding | Q4_0 | q8_0 | 2K | 1 |
| 2043 | Gemma 4 12B | agentic dense | UD-Q4_K_XL / MLX 4-bit | q8_0 | 131K | 1 |
| 2045 | LFM2.5 230M | tiny dense / edge | Q8_0 (CPU Q4_K_M) | q8_0 (CPU q4_0) | 32K/slot | 4 |
| 2046 | LFM2.5 350M | tiny dense / edge | Q8_0 / MLX 8-bit / CPU Q4_K_M | q8_0 (CPU q4_0) | 32K/slot | 4 |
| 2047 | LFM2 350M Extract | structured extraction | Q8_0 | q8_0 | 128K/slot | 4 |
| 2048 | LFM2.5 Encoder 350M | masked-LM encoder | FP32 (MPS / CPU) | — | 8K | 1 |
| 2042 | LFM2.5 Embedding 350M | embedding | Q8_0 (Metal / CPU) | q8_0 | 512/slot | 2 |
| 2052 | LFM2.5 VL 450M | tiny VLM / edge | Q8_0 / MLX 8-bit / CPU Q4_K_M | q8_0 (CPU q4_0) | 32K/slot | 4 |
| 2053 | Qwen 3.8 27B | big dense | UD-Q4_K_XL / MLX 4-bit / FP8 | q8_0 / fp8 | 131K/slot (llama); 262K/seq (MLX/vLLM) | 2 / 1 / 4 |
| 2054 | LFM2.5 1.2B Instruct | instruction dense | Q8_0 / MLX 8-bit | q8_0 | 32K | 1 |
| 2055 | LFM2.5 VL 3B | VLM / edge | Q8_0 / MLX 8-bit / FP8 | q8_0 / fp8 | 32K | 1 |
| 2056 | Qwen 3.8 27B Uncensored | uncensored big dense | Q4_K_M / MLX 4-bit / FP8 | q8_0 / fp8 | 131K/slot (llama); 262K/seq (MLX/vLLM) | 2 / 1 / 4 |
| 2057 | Ornith 1.5 9B | agentic dense | Q4_K_M / MLX 4-bit | q8_0 | 262K | 1 |
| 2058 | Ornith 1.5 35B-A3B | agentic MoE | Q4_K_M / MLX 4-bit | q8_0 | 262K | 1 |
| 2059 | LFM2.5 8B-A1B | reasoning MoE | Q8_0 / MLX 8-bit / CPU Q8_0 | q8_0 | 128K | 1 |
| 2060 | Hy-MT2 7B | translation dense | Q4_K_M | q8_0 | 8K | 1 |

Qwen 3.8 27B Uncensored uses the source repository's recommended `Q4_K_M`
GGUF because it does not publish the standard `UD-Q4_K_XL` quant. Its matching
projector and optional separate MTP companion are preserved with the model
weights on `smarty`, but MTP is not enabled until that release's runtime path
is validated. The publisher describes the model as refusal-removed; keep it
behind application-level access controls and safeguards.

### Available and Reserved Ports

| Port | Status | Notes |
|------|--------|-------|
| 2049 | Blocked | NFS port; Fetch implementations such as Node reject it as an unsafe port |
| 2050 | Reserved | Default `hermes-router` sidecar port; do not assign to a model |
| 2051 | Reserved | Default port for the `hermes-auxiliary-brain` managed llama.cpp server; do not assign to a model |

## Directory Structure

```
models.server/
├── run.sh                  # Single entry point — detects platform, dispatches
├── setup.sh                # Platform setup, including TTS serving engines
├── scripts/
│   ├── run-llama.sh        # Generic llama.cpp launcher
│   ├── run-mlx.sh          # Generic MLX launcher
│   ├── run-mlx-audio.sh    # Generic MLX-Audio TTS launcher
│   ├── run-vllm.sh         # Generic vLLM launcher
│   ├── run-vllm-omni.sh    # Generic vLLM-Omni TTS launcher
│   ├── run-sglang-omni.sh  # Generic SGLang-Omni launcher
│   ├── run-cpu.sh          # Generic CPU-only launcher (Pi)
│   ├── run-transformers.sh # Generic Transformers launcher
│   ├── mlx-audio-server.py # Pins one MLX TTS model behind its roster alias
│   ├── transformers-server.py # Server for non-generative tasks
│   ├── parse-config.py      # Reads model.json → shell variables
│   ├── setup-common.sh      # Shared helpers (CUDA env, venv paths)
│   ├── setup-vllm.sh        # Creates/updates .venv-vllm
│   ├── setup-vllm-omni.sh   # Creates/updates .venv-vllm-omni
│   ├── setup-sglang-omni.sh # Installs Audio8's pinned SGLang adapter
│   ├── setup-mlx.sh         # Creates/updates .venv-mlx
│   └── setup-transformers.sh # Creates/updates the Transformers .venv
├── <model-id>/
│   ├── model.json          # All config: ports, quants, engine settings
│   ├── launchd/            # macOS service unit
│   └── systemd/            # Linux service unit
├── .venv-mlx/              # Shared MLX venv (macOS)
├── .venv-vllm/             # Shared vLLM venv (Linux)
├── .venv-vllm-omni/        # Isolated vLLM-Omni venv (Linux)
├── .venv-sglang-omni/      # Isolated Audio8 SGLang-Omni venv (Linux)
├── .engines/               # Gitignored pinned engine/adapter checkouts
├── llama.cpp/              # llama.cpp build scripts
├── whisper.cpp/            # whisper.cpp build scripts
└── bench/                  # Benchmark results
```

## Engine Auto-Detection

`run.sh` picks the engine automatically:

- A model's `default_engine` wins when configured. It may be a single engine or a per-platform map keyed by `uname`. The TTS models select `mlx-audio` on Darwin and their CUDA serving engine on Linux.
- **macOS** → `mlx` (mlx-vlm or mlx-lm)
- **ARM Linux without CUDA** → `cpu` (Raspberry Pi)
- **Linux with CUDA** → `llama` (llama.cpp), or `vllm` if model has no GGUF (NVFP4)

Override with `--engine`: `./run.sh qwen-3.5-4b --engine vllm`

## Serving Backends

### llama-server (llama.cpp)
GGUF-quantized models via [llama.cpp](https://github.com/ggerganov/llama.cpp). OpenAI-compatible APIs at `/v1/chat/completions`, or `/v1/embeddings` for embedding models. CUDA + flash attention on smarty, Metal on snappy.

All llama-server launch paths set `--cors-origins localhost` when the installed
binary supports it. This prevents unrelated browser origins from calling a
model server while preserving its same-origin Web UI and non-browser clients
such as `api.server`. The CPU launcher omits the option for older Pi binaries
that predate llama.cpp's CORS controls. CORS is not a network ACL; LAN and
Tailnet access remains the responsibility of the host and network firewalls,
while public model access goes through the authenticated `api.server` proxy.

`model.context` is the default total llama.cpp KV allocation. Optional
`llama.context` and `llama.parallel` values override the shared defaults for
that backend. The launcher enables llama.cpp's unified KV cache, so parallel
slots draw dynamically from one shared allocation instead of receiving rigid
`context / parallel` partitions. A request can grow to the model's training
context while capacity remains in the shared pool; simultaneous requests must
still fit collectively. Continuous batching remains enabled through
llama-server's default.

For example, Qwen 3.8 uses `llama.context=524288` and `llama.parallel=8`.
Any slot can grow to the model's 262K training context while capacity remains,
and short requests let the eight slots share the 524K allocation dynamically.

#### Prime Agent long-context cache misses

**Symptom:** a warm Prime Agent session can suddenly prefill its full context
again even though `llama-server` is still running. **Cause:** llama.cpp reuses
only a matching prompt prefix, while Prime Agent auto-refinement can update its
continual harness and rebuild the system prompt at the front of the request.
The request can therefore miss even when it returns to the same live KV slot.

**Workaround:** disable automatic refinement in
`~/.prime/agent/settings.json`; manual `/refine` remains available:

```json
"autoRefine": {
  "enabled": false
}
```

This can recur after manual refinement, after auto-refinement is re-enabled,
or whenever a client changes the system prompt or tool definitions near the
front of a long conversation. Confirm it in the service log: a miss selects a
slot by LRU and evaluates the full prompt, while a warm hit selects by LCP
similarity and evaluates only the new suffix.

llama.cpp [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) adds MTP (Multi-Token Prediction) speculative decoding using draft heads baked into the main GGUF (no separate drafter file). Set `llama.mtp=true` in `model.json` to pass `--spec-type draft-mtp`; optional `llama.mtp_n_max` overrides `--spec-draft-n-max`. Requires a llama.cpp build from after PR #22673 and a GGUF repo that ships MTP heads (e.g. unsloth's `*-MTP-GGUF` variants). Standard Qwen 3.8 27B and both Qwen 3.6 models use measured depth 3. The official OrcaRouter Qwen 3.8 27B Uncensored GGUF also embeds its MTP head and provisionally uses depth 3 pending a matched sweep. Ornith 1.5 9B uses the protoLabs distilled head at measured depth 2; Ornith 1.5 35B-A3B explicitly disables its native head because it reduced matched throughput. See `bench/BENCHMARKS.md`.

The shared llama.cpp launcher disables CUDA graphs by default at runtime with
`GGML_CUDA_DISABLE_GRAPHS=1`; no llama.cpp rebuild is required. A model can
explicitly enable them in `model.json`:

```json
"llama": {
  "cuda_graphs": true
}
```

The default-off policy works around reproducible GB202 Xid 8 faults during
long-context MTP depth-2 and depth-3 inference; the same fixture ran for 15
minutes at depth 3 without a fault when graphs were disabled. See
`investigations/2026-08-19-qwen-3.8-27b-xid8/` for the captured evidence.

Set `llama.reasoning_effort` to pass a model-wide default through
`--reasoning-effort`. Requests can still override it. Qwen 3.8 uses `medium`;
without this setting its chat template defaults to `xhigh`.

### mlx-vlm / mlx-lm
Vision Language Models via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), and text-only MLX models via `mlx-lm` when `mlx.backend` is `mlx_lm`. macOS only (Apple Silicon / MLX). VLMs serve at `/chat/completions` (no `/v1` prefix); `mlx-lm` serves OpenAI-compatible `/v1` routes.

`mlx-lm` does not take a llama-style context flag. Use `mlx.prompt_concurrency` and `mlx.decode_concurrency` for request batching, plus optional prompt-cache fields. For `mlx-vlm`, `mlx.max_kv_size` is the per-sequence KV limit and `mlx.max_num_seqs` is the concurrent-sequence limit; optional vision-cache and prefill fields are also passed when set. Unlike llama.cpp, MLX does not express this as one total context divided among slots.

For `mlx_lm`, optional `mlx.chat_template_args` is serialized as JSON and
passed through `--chat-template-args`. When the field is absent, the launcher
does not pass the option. It is never passed to `mlx_vlm`, whose CLI does not
support it.

When an official MLX repository stores precision variants in subdirectories,
set `mlx.subdir`; the launcher downloads only that subtree and gives its local
path to `mlx-lm`. LFM2.5 2.6B uses this for the official `8bit/` checkpoint.

`mlx-vlm>=0.6.0` supports speculative decoding on the server. Add optional `mlx.draft_model`, `mlx.draft_kind`, and `mlx.draft_block_size` fields in `model.json` to pass `--draft-model`, `--draft-kind`, and `--draft-block-size`; set `MLX_DISABLE_DRAFT=1` when launching to run without the configured drafter.

LFM2.5 VL 450M uses LiquidAI's official 8-bit MLX checkpoint on snappy. It has a 32K multimodal context per sequence and up to four concurrent sequences; `mlx-vlm>=0.6.14` is required for that concurrency flag plus the LFM2-VL loader and tokenizer fixes.

**Gemma 4 MTP drafters** work but only help large/slow targets. E2B/E4B run with `mlx.draft_enabled=false` (MTP measured *slower* than no-drafter on E4B — 66.8 vs 70.6 tok/s). Gemma 12B keeps MTP enabled after a matched 28.9 versus 21.1 tok/s win. The 26B-A4B/31B entries keep `draft_enabled=true` pending matched MLX tests. The Gemma 4 MTP rollback crash ([mlx-vlm#1260](https://github.com/Blaizzy/mlx-vlm/issues/1260), `AttributeError: 'list' object has no attribute 'max'`) is fixed upstream in `mlx-vlm 0.6.1` (our PR [#1261](https://github.com/Blaizzy/mlx-vlm/pull/1261)). The old local patch has been removed; the current setup floor also includes the later LFM2-VL fixes.

The Ornith MLX entries use the official 4-bit repositories without a separate
drafter. No matched Ornith or Gemma 26B MLX result was run during the smarty
rebaseline because snappy was busy.

### MLX-Audio

[mlx-audio](https://github.com/Blaizzy/mlx-audio) serves TTS on Apple Silicon
through the OpenAI-compatible `POST /v1/audio/speech` endpoint. The local
launcher preloads the model configured by `mlx_audio.model` and registers it
under the roster's stable `id`, so clients use the same short model name on
both platforms.

Qwen3 TTS uses the 0.6B and 1.7B CustomVoice checkpoints from
`mlx-community`; Audio8 uses
`mlx-community/Audio8-TTS-Preview-0.6b-bf16`. Audio8 requires
`mlx-audio>=0.4.7` for the `arktts` loader. Qwen supports incremental audio
chunks by setting `"stream": true`; the current Audio8 MLX implementation
returns the completed clip, while its CUDA SGLang-Omni path streams chunks.

MLX-Audio 0.5.0 currently requires `setuptools<81`, so its shared environment
cannot install the Setuptools 83 fix for `PYSEC-2026-3447`. Keep the latest
compatible Setuptools release until MLX-Audio removes that upper bound; the
setup script separately enforces fixed Aiohttp and Datasets releases.

### vLLM
GPU-accelerated serving via [vLLM](https://github.com/vllm-project/vllm). Linux only (CUDA). Supports online FP8 quantization, Marlin NVFP4, and continuous batching for high-throughput concurrent serving.

vLLM treats context as per-sequence length. Use `vllm.max_model_len` for `--max-model-len` and `vllm.max_num_seqs` for request concurrency. If `vllm.max_model_len` is absent, the launcher falls back to `model.context`.

Set `vllm.kv_cache_bytes` to pass an exact `--kv-cache-memory-bytes` pool instead of sizing KV from a percentage of VRAM. The pool is shared dynamically by up to `vllm.max_num_seqs` requests: one request may consume the full pool, while concurrent requests divide it according to their live token counts. `gpu_memory_utilization` remains a startup admission guard when exact bytes are configured; it does not resize the pool. Native MTP is configured through `vllm.speculative_config`.

Qwen 3.6 27B uses a 10,194,124,800-byte FP8 KV pool, which vLLM 0.26.0 reports as exactly 262,144 aggregate tokens with MTP depth 4 and up to four scheduled requests. Four simultaneous 262K requests do not fit; four equally long requests can use about 65K tokens each. The default Linux engine remains llama.cpp because it is faster and smaller for the usual single request. Use `./run.sh qwen-3.6-27b --engine vllm` when continuous batching is more valuable; the measured four-request aggregate was 310 tok/s. Keep the vLLM attention backend on `auto`: forcing Triton caused an illegal-memory-access crash under four-way load.

The vLLM environment enforces the fixed Setuptools 83 floor. Its current
`diskcache` 5.6.3 dependency still reports `PYSEC-2026-2447`; no fixed
DiskCache release is available yet.

### vLLM-Omni

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) provides the CUDA TTS
pipeline and OpenAI-compatible speech API for Qwen3 TTS. It runs in the
isolated `.venv-vllm-omni` environment because each vLLM-Omni release requires
the matching vLLM release. The setup currently installs vLLM and vLLM-Omni
0.26.0.

Both Qwen3 TTS sizes use the shared `scripts/configs/qwen3-tts.yaml`, copied
from vLLM-Omni 0.26.0's two-stage talker/code2wav deployment with codec chunk
streaming. Keeping the config in this repository avoids relying on non-Python
YAML files being included in the PyPI wheel. The 0.6B and 1.7B servers expose
`qwen3-tts-0.6b-customvoice` on port 2033 and
`qwen3-tts-1.7b-customvoice` on port 2035.

### SGLang-Omni

Audio8's CUDA path uses its upstream
[SGLang-Omni adapter](https://github.com/Audio8-AI/Audio8_TTS/tree/master/sglang_omni).
The setup script checks out the exact SGLang-Omni and Audio8 commits validated
by the model authors, installs them in `.venv-sglang-omni`, and keeps those
generated source trees under `.engines/`.

The adapter provides dynamic batching, reference-audio voice cloning, and
streaming output through `/v1/audio/speech`. It automatically selects
FlashInfer on consumer Blackwell GPUs that do not have an FA3 kernel image.
Audio8 is exposed as `audio8-tts-0.6b` on port 2034.

### Text-to-Speech API

All TTS ports use the OpenAI speech route and stable roster aliases:

```bash
curl http://localhost:2033/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts-0.6b-customvoice",
    "input": "Hello from Qwen text to speech.",
    "voice": "ryan",
    "response_format": "wav"
  }' \
  --output qwen3-tts.wav

curl http://localhost:2034/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "audio8-tts-0.6b",
    "input": "Hello from Audio8 text to speech.",
    "response_format": "wav"
  }' \
  --output audio8-tts.wav
```

Use port 2035 and model `qwen3-tts-1.7b-customvoice` for the larger Qwen
checkpoint with natural-language instruction control.

### CPU llama-server
ARM Linux without CUDA auto-selects the `cpu` engine. This is mainly for the Raspberry Pi 5 nodes (`192.168.2.144` and `192.168.2.145`). LFM2.5 230M, 350M, and VL 450M use their `cpu` configs with GGUF `Q4_K_M`, q4 KV cache, flash attention, and `checkpoint_min_step=0` for effective warm prompt reuse. `Q4_K_M` matches Liquid's general recommended GGUF balance; flash attention is their Pi-specific note.

LFM2.5 350M follows platform auto-detection: CUDA-backed llama.cpp on Linux
GPU hosts such as smarty and scrappy, the official 8-bit MLX checkpoint on
snappy, and the CPU engine on ARM Linux without CUDA, including both Raspberry
Pis. It uses its native 32K context per request: 128K total across four llama
slots. LFM2 350M Extract keeps its separate 512K total configuration and CPU
default on Linux, but uses Metal-backed llama.cpp on snappy. Its CPU path uses
q8 KV cache, flash attention, and warm prompt reuse.

LFM2.5 VL 450M and 3B use LiquidAI's official `Q8_0` GGUFs and matching
vision projectors with llama.cpp. Both keep a 32K multimodal context per slot;
450M uses four slots while 3B uses one. Both default to the official 8-bit MLX
checkpoint on macOS. The 450M also supports the Q4_K_M/q4 CPU backend; the 3B
offers optional FP8 vLLM serving on CUDA.

LFM2.5 1.2B Thinking, LFM2.5 1.2B Instruct, and LFM2.5 2.6B use the official
`Q8_0` GGUFs with full CUDA offload on `smarty`, and official 8-bit MLX
checkpoints on `snappy`. They run one request at a time with their supported
32K, 32K, and 128K contexts, respectively. None has a CPU backend configured.

LFM2.5 Embedding 350M uses the official `Q8_0` GGUF on both platforms. Liquid
does not publish an MLX export for this bidirectional embedding model, and its
own Mac benchmark uses llama.cpp. The per-platform default therefore selects
Metal-backed llama.cpp on snappy and CPU-only llama.cpp on Linux, including
smarty. It exposes `/v1/embeddings` on port 2042 with two 512-token slots,
matching its trained maximum sequence length. Its output vector has 1,024
dimensions; that dimension count is not its token context. Port 2049 is
deliberately skipped because Fetch implementations block the historical NFS
port.

#### LFM default platform matrix

This is the engine selected by `run.sh` without `--engine`. `llama GPU` means
full CUDA offload on NVIDIA hosts and Metal offload on snappy.

| Model | smarty | scrappy (WSL) | sparky | snappy | Raspberry Pis |
|---|---|---|---|---|---|
| LFM2.5 230M | llama GPU | llama GPU | llama GPU | MLX GPU | llama CPU |
| LFM2.5 350M | llama GPU | llama GPU | llama GPU | MLX GPU | llama CPU |
| LFM2 350M Extract | llama CPU | llama CPU | llama CPU | llama GPU | llama CPU |
| LFM2.5 Embedding 350M | llama CPU | llama CPU | llama CPU | llama GPU | llama CPU |
| LFM2.5 Encoder 350M | Transformers CPU | Transformers CPU | Transformers CPU | Transformers MPS | Transformers CPU |
| LFM2.5 VL 450M | llama GPU | llama GPU | llama GPU | MLX GPU | llama CPU |
| LFM2.5 1.2B Thinking | llama GPU | llama GPU | llama GPU | MLX GPU | unsupported |
| LFM2.5 1.2B Instruct | llama GPU | llama GPU | llama GPU | MLX GPU | unsupported |
| LFM2.5 2.6B | llama GPU | llama GPU | llama GPU | MLX GPU | unsupported |
| LFM2.5 VL 3B | llama GPU | llama GPU | llama GPU | MLX GPU | unsupported |
| LFM2.5 8B-A1B | llama GPU | llama GPU | llama GPU | MLX GPU | llama CPU* |

`*` The 8B-A1B CPU path is available for sufficiently large CPU hosts. Its
Q8 weights and full 128K q8 KV allocation do not fit an 8 GB Raspberry Pi.

| Model | llama weight / KV | Pi CPU weight / KV | MLX weight / KV | Configured context / slots |
|---|---|---|---|---:|
| LFM2.5 230M | Q8_0 / q8_0 | Q4_K_M / q4_0 | 8-bit / default unquantized | 128K / 4 (32K each) |
| LFM2.5 350M | Q8_0 / q8_0 | Q4_K_M / q4_0 | 8-bit / default unquantized | 128K / 4 (32K each) |
| LFM2 350M Extract | Q8_0 / q8_0 | Q8_0 / q8_0 | —; snappy uses llama | 512K / 4 (128K each) |
| LFM2.5 Embedding 350M | Q8_0 / q8_0 | Q8_0 / q8_0 | —; snappy uses llama | 1K / 2 (512 each) |
| LFM2.5 Encoder 350M | FP32 / — | FP32 / — | FP32 MPS / — | 8K / 1 |
| LFM2.5 VL 450M | Q8_0 / q8_0 | Q4_K_M / q4_0 | 8-bit / default unquantized | 128K / 4 (32K each) |
| LFM2.5 1.2B Thinking | Q8_0 / q8_0 | unsupported | 8-bit / default unquantized | 32K / 1 |
| LFM2.5 1.2B Instruct | Q8_0 / q8_0 | unsupported | 8-bit / default unquantized | 32K / 1 |
| LFM2.5 2.6B | Q8_0 / q8_0 | unsupported | 8-bit / default unquantized | 128K / 1 |
| LFM2.5 VL 3B | Q8_0 / q8_0 | unsupported | 8-bit / default unquantized | 32K / 1 |
| LFM2.5 8B-A1B | Q8_0 / q8_0 | Q8_0 / q8_0 | 8-bit / default unquantized | 128K / 1 |

The MLX launchers do not currently configure KV quantization. For llama.cpp,
the common `context` is one shared KV allocation across all configured slots;
each request remains bounded by the model's training context. MLX-lm uses its
own prompt/decode concurrency controls;
MLX-VLM 450M explicitly sets 32K per sequence and four concurrent sequences.
Optional vLLM configs are not platform defaults: VL 450M and VL 3B specify FP8
weights and FP8 KV at 32K per sequence, while 230M specifies automatic
weight/KV dtype at 32K per sequence.

### Transformers (MPS / CPU)

LFM2.5 Encoder 350M is a bidirectional masked-language model, not a causal LLM. LiquidAI does not publish a GGUF for this exact checkpoint, and llama.cpp cannot serve its masked-LM API. It therefore defaults to the small Transformers backend while the two generative 350M models stay on llama.cpp. The backend uses FP32 on every platform, selecting MPS on Apple Silicon and CPU elsewhere. CUDA is deliberately disabled so running it on smarty does not consume VRAM.

Install its runtime with `scripts/setup-transformers.sh`, then launch it with `./run.sh lfm2.5-encoder-350m`. It exposes `GET /health`, `GET /v1/models`, and `POST /v1/fill-mask`:

```bash
curl http://localhost:2048/v1/fill-mask \
  -H 'Content-Type: application/json' \
  -d '{"input":"The capital of France is <|mask|>.","top_k":5}'
```

## Quantization Standards

The agentic workstation profile is shared by Qwen 3.5 9B, Qwen 3.6 35B-A3B,
Gemma 4 12B, Gemma 4 26B-A4B, and both Ornith 1.5 models: one request slot,
the chosen GGUF artifact's native maximum context, Q4-class weights, q8 KV on
llama.cpp, and 4-bit MLX weights on snappy. Dense versus MoE does not change
that base serving shape. Speculative decoding is enabled only where a matched
on/off test shows a win. Gemma 4 12B is the sole context exception: its current
Unsloth GGUF reports a 131,072-token trained context even though the upstream
Transformers configuration advertises 262,144.

| Model size | Weight quant | KV cache | Context | Parallel slots |
|------------|-------------|----------|---------|----------------|
| >= 4B | UD-Q4_K_XL | q8_0 / fp8 | 64K | MoE: 8, big dense: 2, small: 2 |
| < 4B | Q8_0 | q8_0 / fp8 | 32K | 2 |

LFM2.5 230M, 350M, and VL 450M are small-edge exceptions: CUDA uses Q8_0,
while Pi CPU uses Q4_K_M with q4 KV. Each is configured for four 32K slots on
llama.cpp-style backends; the text models use four-way prompt/decode
concurrency in `mlx-lm`, and VL 450M allows four 32K sequences in `mlx-vlm`.
LFM2.5 1.2B Thinking, 1.2B Instruct, 2.6B, VL 3B, and 8B-A1B are single-slot
Q8 exceptions; they use MLX on macOS and CUDA-backed llama.cpp on Linux. The
8B-A1B also exposes the same Q8/q8 profile through the optional CPU engine.
The smaller LFM generative, vision-language, and embedding models use `Q8_0`;
230M and 350M now auto-select GPU serving where available and their explicit
CPU configs on the Pis. VLMs default to MLX on Darwin, while the embedder
selects Metal on Darwin and CPU on Linux. LFM2.5 Encoder 350M stays FP32
because its bidirectional masked-LM checkpoint has no GGUF.

The TTS entries use their publishers' BF16/MLX checkpoints and codec-aware
serving engines; the GGUF quantization table does not apply to them.

Hy-MT2 7B is a translation-specific exception to the general >=4B profile.
It uses Tencent's official Q4_K_M GGUF and the publisher's recommended 8K
operating context instead of the 262K architectural limit in the checkpoint
metadata. Its server defaults are temperature 0.7, top-p 0.6, top-k 20, and
repeat penalty 1.05, with at most 4,096 generated tokens. The model has no
default system prompt; callers should send the translation instruction and
source text as the user message.

## Adding a New Model

1. Create `<model-id>/` directory
2. Add `model.json` with all engine config (see any existing model for the schema)
3. Add `launchd/` and `systemd/` service units. The launchd plist must execute
   a model-named `launchd/kortexa-<model-id>.sh` wrapper that delegates to the
   repository's `run.sh`; this keeps macOS Login Items unambiguous.
4. Follow the quantization standards above
5. Test: `./run.sh <model-id>`

## Smarty Qwen coding-agent capacity lease

Coding agents must acquire a lease before they use the already-running managed
Qwen 3.8 27B endpoint on `smarty`. The lease is intentionally conservative: it
allows one agent request at a time, limits context to 65,536 tokens and output
to 4,096 tokens, and reserves the second Qwen slot plus at least 16 GiB of free
VRAM for production headroom.

Run the tool on `smarty` from this repository. Supply the same durable owner
fields to `acquire`, every `heartbeat`, and `release`:

```bash
python3 scripts/qwen_capacity_lease.py probe
python3 scripts/qwen_capacity_lease.py acquire \
  --actor-id builder-sol-a --harness Codex --owner-model gpt-5.6-sol \
  --root-session-id codex:0198ff96-2e31-7c30-9eca-4d4f22265e90 \
  --delegated-worker-id /root/builder --session-ref /root/builder \
  --ttl-seconds 900
```

Owner values use a positive ASCII grammar, not a secret denylist. `harness` is
exactly `Codex`, `OMP`, or `Prime Agent`. Actor and model identifiers accept
letters, digits, and the established `. _ : / @ + -` separators; delegated
worker and session references also accept a leading `/` for Codex task paths.
The root session is either a lowercase UUIDv7, as emitted by OMP and Prime, or
an Agent Deck public ID with the registered `codex:`, `claude:`, `hermes:`, or
`prime:` prefix. Controls, whitespace, comments, assignments, serialized
objects or arrays, Unicode lookalikes, unknown harnesses, and oversized values
are rejected before the state file is opened.

An admitted result includes the lease ID, endpoint, fixed budget, heartbeat,
and expiry. Heartbeat immediately before every inference request. Do not send
the request unless that heartbeat returns `"decision": "admit"`; queue or stop
on every `queue`, `blocked`, expired, unreadable, or unknown result. Keep only
one request in flight, keep the complete request context within the recorded
context budget, and set the output-token limit no higher than the recorded
output budget. Direct use without a current same-owner lease is outside this
contract.

```bash
python3 scripts/qwen_capacity_lease.py heartbeat \
  --actor-id builder-sol-a --harness Codex --owner-model gpt-5.6-sol \
  --root-session-id codex:0198ff96-2e31-7c30-9eca-4d4f22265e90 \
  --delegated-worker-id /root/builder --session-ref /root/builder \
  --lease-id qwen-example-lease-id --ttl-seconds 900

python3 scripts/qwen_capacity_lease.py release \
  --actor-id builder-sol-a --harness Codex --owner-model gpt-5.6-sol \
  --root-session-id codex:0198ff96-2e31-7c30-9eca-4d4f22265e90 \
  --delegated-worker-id /root/builder --session-ref /root/builder \
  --lease-id qwen-example-lease-id --outcome completed
```

Admission fails closed when Qwen or a protected production service is
unhealthy, a managed GPU process disagrees with the live `ktxsvc` roster, an
unknown CUDA process or LegoLM GPU owner exists, GPU load or free VRAM is
unsafe, the reserved Qwen slot is busy, or another agent lease is active. A
small canary runs only after those checks and is followed by the same checks
again before the lease is written.

The tool cannot start, stop, or restart services and cannot signal processes.
It observes only `ktxsvc list`, two exact read-only `nvidia-smi` queries, the
Qwen listener, and local health endpoints. It stores owner, capacity, and
lifecycle metadata in an owner-only local state file; it never stores prompts,
credentials, or model output. State version 2 closes the root, active/history,
owner, budget, admission, canary, heartbeat, expiry, and release records: an
unknown member, wrong type, unregistered outcome/reason, or inconsistent
lifecycle fails closed after load and before write. An exact version-1 state is
migrated under the existing lock; an incompatible version-1 or unknown-version
file is rejected without rewrite. Expired crash leases are recorded and
cleared under the same lock before a new admission. Any workflow that needs
downtime must first drain every Qwen worker, stop before mutation, and obtain
Franci's explicit permission under the Smarty GPU guide. LegoLM work is never
mutated by this lease workflow.

## Service Management

### macOS (launchd)
```bash
ln -s ~/src/models.server/<model-id>/launchd/ai.kortexa.<model-id>.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/ai.kortexa.<model-id>.plist
launchctl start ai.kortexa.<model-id>
```

### Linux (systemd)
```bash
sudo ln -s ~/src/models.server/<model-id>/systemd/kortexa-ai-llm-<model-id>.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start kortexa-ai-llm-<model-id>
```
