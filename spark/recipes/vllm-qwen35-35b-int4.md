# vLLM Qwen 3.5-35B-A3B int4 Recipe

**Target:** DGX Spark (GB10 Blackwell, SM121, 128GB unified memory)
**Result:** warm steady-state 100.1 tok/s aggregate with 4 concurrent 1024-token requests (~25.0 tok/s each)

Primary control knob on Spark: use explicit `--kv-cache-memory-bytes`. But vLLM 0.17.2rc1 still applies the startup free-memory guard using `--gpu-memory-utilization`, so set a conservative cap there too.

## Quick Start

```bash
docker run --rm --name vllm-qwen35-35b \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e VLLM_BASE_DIR=/root/.cache/huggingface \
  vllm-node:latest \
  vllm serve Intel/Qwen3.5-35B-A3B-int4-AutoRound \
  --port 2242 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.4 \
  --kv-cache-memory-bytes 12884901888 \
  --load-format fastsafetensors \
  --kv-cache-dtype fp8 \
  --enable-force-include-usage
```

## Requirements

- **Docker image:** `vllm-node:latest` (vLLM 0.17.2rc1.dev7)
- **Model:** `Intel/Qwen3.5-35B-A3B-int4-AutoRound` (~21GB)
- **Memory:** works alongside existing resident services with ~75 GiB unified memory available
- **Observed process allocation:** ~34 GiB for the 35B vLLM engine (`check_mem`)
- **KV cache budget:** 12 GiB explicit (`--kv-cache-memory-bytes 12884901888`)
- **Startup guardrail:** `--gpu-memory-utilization 0.4` to satisfy vLLM's free-memory check even when KV bytes are explicit

## Benchmark Results

| Test | Shape | Tokens | Duration | TPS |
|------|-------|--------|----------|-----|
| smoke | 1 request | 3 | 48.53s | 0.06 |
| batch-1 (cold-ish after startup) | 4 concurrent | 4096 total | 76.74s | 53.37 aggregate |
| batch-1 per request | 4 concurrent | 1024 each | 76.73s | 13.34 each |
| batch-2 (warm service) | 4 concurrent | 4096 total | 40.90s | 100.14 aggregate |
| batch-2 per request | 4 concurrent | 1024 each | 40.90s | 25.04 each |

The smoke request above was first-token dominated; the real number that matters here is the sustained 4-way decode throughput.

**Warm sustained result: 100.1 tok/s aggregate with 4 concurrent requests (~25.0 tok/s each)**

## Why This Works

1. **vLLM 0.17.2rc1** has proper SM121 (Blackwell) support
2. **FlashInfer attention backend** works correctly on DGX Spark
3. **Intel AutoRound int4** quantization reduces memory bandwidth 4x
4. **MoE architecture** only activates 3B params per token
5. **FP8 KV cache** halves KV memory pressure
6. **The real fix was switching to explicit `--kv-cache-memory-bytes` budgeting** so the cache size is deterministic on a shared Spark
7. **vLLM still checks `gpu_memory_utilization` at startup even with explicit KV bytes**, so the service also needs a conservative cap like `0.4`
8. **CUDA graphs** successfully captured (PIECEWISE=4, FULL=3)

## Key Logs to Verify

```
Using FLASHINFER attention backend
Model loading took 19.3 GiB memory
Available KV cache memory: ~12 GiB
GPU KV cache size: enough for the 4×64k target with headroom
Maximum concurrency for 65,536 tokens per request: comfortably above the 4-request target
Graph capturing finished in 3 secs
Application startup complete
```

## No-Thinking Benchmark Request

Use `chat_template_kwargs.enable_thinking=false` in the request body:

```json
{
  "model": "Intel/Qwen3.5-35B-A3B-int4-AutoRound",
  "messages": [{"role": "user", "content": "Output the word alpha followed by a space over and over until you hit the token limit. No preamble, no explanation."}],
  "max_tokens": 1024,
  "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

## Optimization Opportunities

To push toward 70+ tok/s:

1. **Apply MXFP4 patches** from `namake-taro/vllm-custom`
   - Could add 20-30% performance
   - Requires patching vLLM source

2. **Keep KV budgeting explicit**
   - `--kv-cache-memory-bytes 12884901888` is the coexistence preset for the 4×64k service profile
   - Pair it with `--gpu-memory-utilization 0.4` because current vLLM still uses that flag in the startup admission check

3. **Use TP=2** (if you have 2 GPUs)
   - Would nearly double throughput

## Comparison: BF16 vs int4

| Model | BF16 TPS | int4 TPS | Speedup |
|-------|----------|----------|---------|
| Qwen3.5-4B | 20.6 | - | - |
| Qwen3.5-9B | 12.3 | - | - |
| Qwen3.5-27B | 2.5 | - | - |
| Qwen3.5-35B-A3B | 17.4 | **100.1 aggregate / 25.0 each @ 4-way batch** | **5.8x aggregate** |

## Troubleshooting

### "CUDA out of memory"
- Reduce `--kv-cache-memory-bytes` below `12884901888` for the 4×64k service profile
- Kill other GPU processes

### Service fails immediately at startup with free-memory complaint
- This usually means `--gpu-memory-utilization` was left at the default `0.9`
- Error looked like: `Free memory ... is less than desired GPU memory utilization`
- Fix: keep explicit `--kv-cache-memory-bytes` and also set a conservative `--gpu-memory-utilization` such as `0.4`

### "Model not found"
- Pre-download: `huggingface-cli download Intel/Qwen3.5-35B-A3B-int4-AutoRound`

### Slow startup (>5 min)
- First run compiles CUDA graphs (~3 min)
- Subsequent runs use cached graphs

### "CUTLASS TMA errors"
- This image has issues with Nemotron models
- For Nemotron, use different image or llama.cpp

## Related Files

- `docs/INVESTIGATION_SPARK_PERF.md` - Root cause analysis
- `docs/BENCHMARKS.md` - Full benchmark log
- `patches/vllm_all.patch` - MXFP4 patches for 70+ tok/s
