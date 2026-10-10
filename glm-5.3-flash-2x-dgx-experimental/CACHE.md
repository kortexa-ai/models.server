# Cache controls in the pinned source

This recipe pins vLLM `276fbcff2717bd934cfa37c8a2e4c391f3e7237b`.
The controls below describe that source, not a measured capacity promise.
The [experiment plan](PLAN.md) defines the runtime comparisons.

GB10 uses `FLASHINFER_MLA_SPARSE_SM120`, not the SM100 backend named
`FLASHINFER_MLA_SPARSE`. Upstream canonicalizes its requested `fp8_e4m3`
MLA cache to the packed `fp8_ds_mla` layout (656 bytes of state content per
token before page padding). The DFlash attention cache remains explicitly
`fp8_e4m3`. Record these resolved formats when comparing older layouts.
At this pin, `FlashInferMLASparseSM120Impl` inherits `supports_dcp=False`
and returns no attention log-sum-exp values. The SM100 implementation's
`supports_dcp=True` does not apply to GB10. Enabling the flag alone cannot
make the SM120 kernel perform distributed attention. The backend validator
rejects that combination; other cache controls can still be tested.

The image also contains an optional adapter, enabled only with
`GLM_SM120_DCP=1`. It connects upstream's DCP index filter and merge to
FlashInfer's SM120 LSE output, uses base-2 normalization and supplies neutral
values for empty local shards. It preserves the original DCP=1 path and
excludes PCP and HiSparse. This is a local experimental patch, not upstream
support. Run `bench/glm53-tp2/sm120_dcp_probe.py` inside the image before any
distributed test; a passing arithmetic probe alone does not establish model
correctness or DFlash compatibility.

| Control | What it changes | How to interpret it |
|---|---|---|
| `--kv-cache-memory-bytes` | Explicit cache budget **per GPU/rank** | At 6 GiB on each Spark, the pair reserves about 12 GiB in total. This overrides the utilization-based cache calculation. |
| `--max-model-len` | Maximum request context and some temporary workspace dimensions | A 512K request ceiling is not a promise of eight independent 512K caches. |
| `--max-num-seqs` | Scheduler concurrency ceiling | Actual active concurrency still depends on blocks required by the requests. |
| `--decode-context-parallel-size 2` | Shards eligible attention history across the existing TP2 ranks when the backend implements it | The pinned SM120 sparse backend does not implement DCP. Even supported backends do not make all recurrent, tail and speculative state shared. |
| `--cp-kv-cache-interleave-size 4` | Token distribution across DCP ranks | GLM's four-token kpool requires compatible interleave. Backend and draft compatibility must pass live tests. |
| Speculative `num_speculative_tokens` | Maximum lookahead, draft work and state reservations | Lowering the global maximum can reduce reservations. A smaller adaptive choice for one decode step does not necessarily shrink the allocated state. |
| Speculative `kv_cache_dtype` | Draft attention-cache precision | Set explicitly so a target FP8 cache is not compared with a BF16 draft-cache variant by accident. |
| `--mamba-cache-dtype` | KDA convolution-state dtype in this GLM implementation | It is separate from the much larger recurrent state. |
| `--mamba-ssm-cache-dtype` | General vLLM recurrent-state control | At this pin GLM's caller does not pass this field into the KDA dtype calculator; the recurrent state remains FP32. Changing the flag alone is not a BF16 experiment. |
| `VLLM_KV_CACHE_LAYOUT` | Selects a physical layout supported by the backend | Upstream resolves the layout before profiling. Packed grouping and padding affect bytes per block; a block count from the older recipe is not directly comparable. |
| `--num-gpu-blocks-override` | Overrides the calculated number of blocks | This is not free capacity. Extra blocks still allocate memory and can exceed the intended pool budget. |

KDA's recurrent shape divides its head count by TP size already. DCP does not
divide that state a second time. The kpool tail is a circular buffer with its
own per-request allocation. DFlash has additional cache groups. Inspect the
whole allocation, not only the main MLA token-history tensor.

The image records the input cache specs, errors and unmodified upstream
allocation result under `/cache/layout-audit/`. It records PyTorch allocated/reserved bytes, CUDA free
memory and host available memory after loading, cache initialization and
warmup under `/cache/memory-audit/`. These observers do not change the cache
plan, scheduler or sampling. Runtime evidence belongs outside Git.

Hold the per-rank pool budget constant for the first DCP comparison. If more
tokens fit, that is a capacity gain. To demonstrate freed memory, reduce the
pool budget while retaining the same tested workload, then compare both
ranks' allocations and available host memory. Rank 0 also runs the API and
vision path. On a Spark, CPU and GPU use the same physical RAM; CPU cache
offload does not create another independent memory tier.

Source references at the pinned revision:

- [Cache configuration](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/config/cache.py)
- [Allocation and packed grouping](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/v1/core/kv_cache_utils.py)
- [GLM KDA dtype call](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/models/glm5next/common/kda.py)
- [KDA state shapes and dtype calculator](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/model_executor/layers/mamba/mamba_utils.py)
- [GLM kpool and circular tail](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/models/glm5next/common/attention.py)
- [SM120 sparse attention implementation](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py)
- [Backend capability validation](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/v1/attention/backend.py)
