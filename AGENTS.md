# Models Server

Run `hostname` to check which machine you're on before doing anything.

## Machines

| Hostname | Hardware | Memory | OS | Status |
|----------|----------|------------|------|--------|
| **smarty** | RTX PRO 6000 Blackwell | 96 GB VRAM | Ubuntu Linux | active |
| **snappy** | Mac Mini M4 Pro | 64 GB unified | macOS | active |
| **scrappy** | RTX 3070 Laptop | 8 GB VRAM | Windows 11 | active |
| **sparky** | DGX Spark GB10 | 128 GB unified | Ubuntu Linux | offline |
| **static** | DGX Spark GB10 | 128 GB unified | Ubuntu Linux | active |
| **shock** | DGX Spark GB10 | 128 GB unified | Ubuntu Linux | active |
| **192.168.2.144** | Raspberry Pi 5 | 8 GB RAM | ARM Linux | active |
| **192.168.2.145** | Raspberry Pi 5 | 8 GB RAM | ARM Linux | active |

## How It Works

- `run.sh` is the single entry point — auto-detects platform and dispatches to the right engine
- All model config lives in `<model-id>/model.json` — engines, quants, ports, KV budgets
- Generic engine scripts live in `scripts/` — they read model.json, not hardcoded values
- Usage: `./run.sh qwen-3.5-4b` or `cd qwen-3.5-4b && ../run.sh`
- Override engine: `./run.sh qwen-3.5-4b --engine vllm`
- `vllm-spark-tp2` is a model-specific distributed engine for the `static` + `shock` pair.

## Rules

- Inspect service and port ownership before changing a running model. Use
  `ktxsvc` for managed services and leave unrelated processes alone.
- Prefer checking existing endpoints. Start or stop a model only when the task
  requires it, and never create a duplicate server.
- Quantization: >= 4B → `UD-Q4_K_XL`, < 4B → `Q8_0`
- Exception: LFM2.5 230M uses `Q4_K_M` for Pi CPU serving
- Exception: Bonsai 2 27B uses native ternary `PQ2_0` with an isolated Prism
  llama.cpp fork on Linux and macOS (Metal). Its MLX 2-bit source is registered
  but serving is deferred; never route it through a stock MLX server.
- KV cache: `q8_0` (llama.cpp) / `fp8` (vLLM) everywhere
- Context: max supported by the model
- Parallel: MoE → 8, dense → 1

- Exception: `qwen-3.8-27b-fast` and `qwen-3.8-27b-fast-abliterated` use
  Cinference NVFP4 weights, K8V4 KV, DFlash2-7, a 524288-token shared pool
  and eight slots. They are pinned to the RTX PRO 6000 UUID, with a 262144-token
  per-request ceiling. Registration does not authorize selecting a new default.
- `"type": "systemone"` marks a decision model (currently `shingi-27b`), not a
  chat LLM. It has no `llama` block; its `shingi` block pins the package,
  weights, Prism runtime and memory floors. Run `scripts/setup-shingi.sh` once
  before starting it; `run-shingi.sh` never builds or installs. Its weights
  may declare a vision `projector`; with it Shingi uses about 9.5 GiB on the 4090.
  Shingi is an exception to the dense-model parallel default: its decision worker
  uses four sequences with shared weights and a shared 16K-token context pool.
- Exception: `qwen-3.8-flash-next` uses the pinned `vllm-spark-tp2` engine on
  `static` + `shock`, with NVIDIA NVFP4 weights, YaRN extension to 1M context,
  FP8 KV, MTP-3, and six sequences.
- Exception: `glm-5.3-flash-exl3` uses the pinned patched TensorFold EXL3
  engine on Shock's GB10: original 2.05-bpw packed weights, BF16 latent KV,
  a 262144-token served window, one decode lane, MTP-1 and a 3-GiB prompt
  snapshot budget. Its checkpoint advertises 1M context; that window does not
  fit the host memory budget. Run `scripts/setup-tensorfold-exl3.sh` explicitly
  before launch. The standard CLI path is text only.

- Exception: `glm-5.3-flash-2x-dgx` uses the pinned `vllm-glm53-tp2` engine
  across Static (API/rank 0) and Shock (rank 1), on model port 2070. NVIDIA
  NVFP4 experts, mixed FP8 dense weights and DFlash2 use 6 GiB FP8 KV/rank,
  a 524288-token served window, eight shared slots and a 6919-token batch budget.
  Run explicit setup on both hosts before installing the same service with
  `ktxsvc`; coordinate both ranks and never run another model on either GPU.
