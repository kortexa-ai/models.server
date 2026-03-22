# vLLM Qwen 3.5-35B-A3B int4 Recipe

**Target:** DGX Spark (GB10 Blackwell, SM121, 128GB unified memory)
**Result:** 50 tok/s (target: 50-79 tok/s)

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
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.7 \
  --load-format fastsafetensors \
  --kv-cache-dtype fp8
```

## Requirements

- **Docker image:** `vllm-node:latest` (vLLM 0.17.2rc1.dev7)
- **Model:** `Intel/Qwen3.5-35B-A3B-int4-AutoRound` (~21GB)
- **Memory:** ~80GB available (model: 19.3 GiB, KV cache: 60+ GiB)

## Benchmark Results

| Test | Tokens | Duration | TPS |
|------|--------|----------|-----|
| 1 | 512 | 9.92s | 51.60 |
| 2 | 512 | 11.07s | 46.23 |
| 3 | 512 | 9.92s | 51.62 |
| 4 | 1024 | 19.94s | 51.36 |
| 5 | 100 | 1.99s | 50.35 |

**Average: 50.2 tok/s**

## Why This Works

1. **vLLM 0.17.2rc1** has proper SM121 (Blackwell) support
2. **FlashInfer attention backend** works correctly on DGX Spark
3. **Intel AutoRound int4** quantization reduces memory bandwidth 4x
4. **MoE architecture** only activates 3B params per token
5. **FP8 KV cache** halves KV memory pressure
6. **CUDA graphs** successfully captured (51 PIECEWISE + 35 FULL)

## Key Logs to Verify

```
Using FLASHINFER attention backend
Model loading took 19.3 GiB memory
Available KV cache memory: 60.32 GiB
Graph capturing finished in 35 secs
Application startup complete
```

## Optimization Opportunities

To push toward 70+ tok/s:

1. **Apply MXFP4 patches** from `namake-taro/vllm-custom`
   - Could add 20-30% performance
   - Requires patching vLLM source

2. **Kill competing processes**
   - Stop llama-server, music-gen, etc.
   - Allows `--gpu-memory-utilization 0.85`

3. **Use TP=2** (if you have 2 GPUs)
   - Would nearly double throughput

## Comparison: BF16 vs int4

| Model | BF16 TPS | int4 TPS | Speedup |
|-------|----------|----------|---------|
| Qwen3.5-4B | 20.6 | - | - |
| Qwen3.5-9B | 12.3 | - | - |
| Qwen3.5-27B | 2.5 | - | - |
| Qwen3.5-35B-A3B | 17.4 | **50.2** | **2.9x** |

## Troubleshooting

### "CUDA out of memory"
- Reduce `--gpu-memory-utilization` to 0.5-0.6
- Kill other GPU processes

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
