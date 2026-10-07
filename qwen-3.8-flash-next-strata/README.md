# Qwen3.8-Flash-Next with Strata on GB10

Manual reproduction of the temporary Shock benchmark from 2026-10-06/07.
TensorFold on Static remains the selected recipe because it supports images;
see [qwen-3.8-flash-next-fast](../qwen-3.8-flash-next-fast/README.md).
This entry is documentation only, with no `model.json`, launcher or service.

Strata ran the full Unsloth **UD-Q4_K_XL** checkpoint on one DGX Spark GB10
(ARM64, 128 GB unified memory). Upstream's x86 build needed the
[community ARM port, PR #1077](https://github.com/Niko1221/Strata/pull/1077).
No source patches were required. This quant's Strata image path is not wired
up yet, so the tested configuration is **text only**.

## Pins and prerequisites

| Component | Tested version / revision |
| --- | --- |
| Strata ARM port | `johnlockejrr/Strata`, `4240007cb72c9266cc091f872217ae68026f0a06` (engine 0.1.40) |
| llama.cpp / ggml | `3cf03257f219afbe7334045ff7c6a06ac68c627d` |
| Main checkpoint | `unsloth/Qwen3.8-Flash-Next-GGUF`, `38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-Q4_K_XL` |
| MTP source | `Qwen/Qwen3.8-Flash-Next`, `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Host tools | CUDA 13.0.88, driver 580.178.04, GCC 13.3, CMake 3.28.3, Python 3.12.3, `uv`, Git, Make |

Use an idle GB10 and inspect managed services and port ownership before
starting. The four checkpoint shards total 111,334,654,784 bytes; allow about
125 GB of disk space for weights, draft tensors, packs, build and isolated
caches. The original experiment retained about 112 GiB. All commands below
run on the Spark, in the same shell. They use existing host tools.

## Build the ARM engine

Use a fresh temporary directory. Keep downloads, caches and the virtual
environment inside it so the experiment can be removed as one directory.

```sh
set -euo pipefail
export STRATA_TEST_ROOT="$HOME/tmp/strata-gb10-repro"
export HF_HOME="$STRATA_TEST_ROOT/hf-home"
export UV_CACHE_DIR="$STRATA_TEST_ROOT/uv-cache"
mkdir -p "$STRATA_TEST_ROOT"

git clone --no-checkout https://github.com/johnlockejrr/Strata.git "$STRATA_TEST_ROOT/Strata"
cd "$STRATA_TEST_ROOT/Strata"
git checkout --detach 4240007cb72c9266cc091f872217ae68026f0a06
git clone --no-checkout https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp checkout --detach 3cf03257f219afbe7334045ff7c6a06ac68c627d

# /usr/bin/python3 was Python 3.12.3 on Shock; its default PATH Python was 3.14.
uv venv --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python \
  huggingface-hub==2.1.1 numpy==2.5.3 jinja2==3.1.6 regex==2026.9.29 \
  pyyaml==6.0.3 tqdm==4.70.1 requests==2.34.2 pillow==12.3.0 psutil==7.2.2

cmake -G 'Unix Makefiles' -S . -B build \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121 -DSTRATA_GGML_DIR=third_party/llama.cpp \
  -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16 \
  -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build -j 8 --target strata strata-device native_expert_parity prefill_mmq_kquant_test parallel8
build/strata-device
ctest --test-dir build --output-on-failure \
  -R '^(native_expert_parity_(q4_K_q5_1|q4_K_q8_0|q5_K_q8_0)|prefill_mmq_kquant_test)$'
```

`121` selects GB10's SM 12.1 kernels. The explicit ARM ISA enables dot-product,
i8mm and FP16 support; the fork avoids x86 AVX paths. MMQ K-quant kernels serve
this checkpoint's experts during prefill. The Q6 flag is needed by the focused
prefill test's coverage; the served checkpoint is still UD-Q4_K_XL. All four
checks passed in the original test.

## Download, verify and pack

```sh
.venv/bin/hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include 'UD-Q4_K_XL/*.gguf' --local-dir "$STRATA_TEST_ROOT/checkpoint"

(cd "$STRATA_TEST_ROOT/checkpoint/UD-Q4_K_XL" && sha256sum -c - <<'SHA256'
4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082  Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9  Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3  Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a  Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
SHA256
)

.venv/bin/python tools/iq_pack.py \
  --gguf "$STRATA_TEST_ROOT/checkpoint/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" \
  --out "$STRATA_TEST_ROOT/packs/ud-q4_k_xl" --compat-bf16

export STRATA_MTP_REVISION=de4b8e4d43b917e7706784d8bb445c9af86a3540
.venv/bin/python tools/mtp_fetch.py fetch --out "$STRATA_TEST_ROOT/mtp"
.venv/bin/python tools/mtp_fetch.py verify --out "$STRATA_TEST_ROOT/mtp"
.venv/bin/python tools/mtp_pack.py --src "$STRATA_TEST_ROOT/mtp" \
  --experts q2_0 --out "$STRATA_TEST_ROOT/mtp/mtp-q2_0.gguf"
.venv/bin/python tools/mtp_rt.py --gguf "$STRATA_TEST_ROOT/mtp/mtp-q2_0.gguf" \
  --out "$STRATA_TEST_ROOT/mtp/rt"
cp data/draft_vocab.bin "$STRATA_TEST_ROOT/mtp/rt/draft_vocab.bin"
```

`--compat-bf16` converts 195 unsupported small projections to BF16, about
1.20 GiB. Native expert quantization and PLE bytes stay unchanged. Do not add
`--experts-bin`: that duplicates about 77 GB of main-model experts. MTP fetch
reads only the 31 draft tensors via HTTP ranges, rather than downloading the
full BF16 checkpoint. The test used a parallel range-fetch wrapper to reduce
download time; the pinned stock command above produces the same tensors.

## Serve four text streams

Write a config with absolute paths, then start a foreground server on the
temporary loopback port. This uses the tested four-slot, 262K-per-slot settings.

```sh
.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
r = Path(os.environ["STRATA_TEST_ROOT"]).resolve()
config = {
    "exe": str(r / "Strata/build/strata"),
    "args": [
        "--pack", str(r / "packs/ud-q4_k_xl"),
        "--native", str(r / "checkpoint/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"),
        "--expert-profile", str(r / "Strata/data/expert-profile.bin"),
        "--expert-cache", "auto", "--mmap-experts", "--prefill", "auto",
        "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(r / "mtp/rt"),
        "--max-context", "262144", "--kv", "int8", "--batch-mtp"
    ],
    "cwd": str(r / "Strata"), "tokenizer": str(r / "packs/ud-q4_k_xl/tokenizer"),
    "model_name": "qwen3.8-flash-next-ud-q4_k_xl", "parallel": 4,
    "host": "127.0.0.1", "port": 18080, "open_browser": False,
    "log": str(r / "engine.log")
}
(r / "strata.json").write_text(json.dumps(config, indent=2) + "\n")
PY
STRATA_MTP_FULL_HEAD=1 .venv/bin/python -m serve.server --engine strata \
  --config "$STRATA_TEST_ROOT/strata.json" --host 127.0.0.1 --port 18080
```

The pinned grouped-MTP path fails with `mtp: unsupported native MMVQ GGML type`:
binding a shared subset draft head leaves its quantization type at `-1`.
`STRATA_MTP_FULL_HEAD=1` bypasses it with the full native head. MTP stays enabled,
with one proposal per slot per window. See the pinned
[binding code](https://github.com/johnlockejrr/Strata/blob/4240007cb72c9266cc091f872217ae68026f0a06/src/core/mtp.cpp#L601).
Recheck this workaround after changing the source revision.

The single-stream benchmark used `parallel: 1`, omitted `--batch-mtp` and
`STRATA_MTP_FULL_HEAD`, and kept `--spec 4`. On unified memory, do not add a
resident RAM budget or KV streaming to this recipe. All 24,576 experts fit
in the 71.73 GiB GPU cache; the PLE table remained file-backed.

From another shell, a minimal smoke request is:

```sh
curl -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-ud-q4_k_xl","messages":[{"role":"user","content":"What is 17 * 23?"}],"temperature":0,"reasoning_effort":"none","max_tokens":32}'
```

## Measured results and limits

Warm, greedy requests used `reasoning_effort: none` and 256 output tokens.
Single-stream figures are medians of three fresh-prompt runs; the 260K row is
one capacity check. Four-stream figures include prompt admission and all
response time, with about 5,600 tokens per prompt and a shared cached prefix.

| Workload | Prefill tokens/s | Output tokens/s | First token |
| --- | ---: | ---: | ---: |
| Single, 511-token prompt | 410 | 43.1 | 1.29 s |
| Single, 32,768-token prompt | 1,553 | 44.2 | 21.17 s |
| Single, 65,536-token prompt | 1,546 | 45.5 | 42.52 s |
| Single, 260,000-token prompt | 1,406 | 41.6 | 185.24 s |
| Four prefix-cached streams, 262K reserved each | — | 52.2 aggregate | 1.69 s median |

Four 262K reservations fit, with at least 13.94 GiB available RAM. Four
simultaneous 260K prompts were not tested. Each slot allocates its full context;
prefix and conversation caching save prefill work but do not shrink that
reservation. Prompt admissions are serial. Swap occupancy peaked at 0.89 GiB.
Arithmetic, generated-code and conversation smoke checks passed; these do not
establish broad model-quality parity.

Stop the foreground server with Ctrl-C and verify its listener and owned
processes are gone before cleanup. The original experiment used
`/home/francip/tmp/strata-gb10-20261006` on Shock. Detailed evidence is in
`~/Desktop/ai reports/strata-shock-20261006/` on Snappy and
[models.server #44](https://github.com/kortexa-ai/models.server/issues/44).
