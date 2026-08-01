#!/usr/bin/env python3
"""Parse model.json and output shell variable assignments.

Usage: eval "$(python3 scripts/parse-config.py <model-dir>/model.json)"
"""
import json
import sys


def quote(v):
    """Shell-safe quoting."""
    return str(v).replace("'", "'\\''")


def emit(name, value):
    print(f"{name}='{quote(value)}'")


def main():
    with open(sys.argv[1]) as f:
        m = json.load(f)

    # Common
    print(f"MODEL_NAME='{quote(m['name'])}'")
    print(f"MODEL_ID='{quote(m['id'])}'")
    print(f"MODEL_PORT='{m['port']}'")
    print(f"MODEL_HOST='{m.get('host', '0.0.0.0')}'")
    print(f"MODEL_CONTEXT='{m.get('context', 65536)}'")
    print(f"MODEL_PARALLEL='{m.get('parallel', 1)}'")
    print(f"MODEL_CACHE_TYPE='{m.get('cache_type', 'q8_0')}'")
    print(f"MODEL_MULTIMODAL={'true' if m.get('multimodal') else 'false'}")
    print(f"MODEL_EMBEDDING={'true' if m.get('embedding') else 'false'}")

    # llama
    llama = m.get("llama")
    if llama:
        print(f"LLAMA_REPO='{quote(llama['repo'])}'")
        print(f"LLAMA_QUANT='{quote(llama['quant'])}'")
        emit("LLAMA_CONTEXT", llama.get("context", m.get("context", 65536)))
        emit("LLAMA_PARALLEL", llama.get("parallel", m.get("parallel", 1)))
        if llama.get("mtp"):
            print("LLAMA_MTP=true")
        if llama.get("mtp_n_max"):
            print(f"LLAMA_MTP_N_MAX='{llama['mtp_n_max']}'")
    else:
        print("LLAMA_SUPPORTED=false")

    # mlx
    mlx = m.get("mlx")
    if mlx:
        print(f"MLX_REPO='{quote(mlx['repo'])}'")
        backend = mlx.get("backend", "mlx_vlm")
        print(f"MLX_BACKEND='{backend}'")
        if backend == "mlx_lm":
            emit("MLX_PROMPT_CONCURRENCY", mlx.get("prompt_concurrency", m.get("parallel", 1)))
            emit("MLX_DECODE_CONCURRENCY", mlx.get("decode_concurrency", m.get("parallel", 1)))
        for key, env in (
            ("prefill_step_size", "MLX_PREFILL_STEP_SIZE"),
            ("prompt_cache_size", "MLX_PROMPT_CACHE_SIZE"),
            ("prompt_cache_bytes", "MLX_PROMPT_CACHE_BYTES"),
            ("max_tokens", "MLX_MAX_TOKENS"),
            ("vision_cache_size", "MLX_VISION_CACHE_SIZE"),
            ("max_kv_size", "MLX_MAX_KV_SIZE"),
            ("kv_bits", "MLX_KV_BITS"),
            ("kv_quant_scheme", "MLX_KV_QUANT_SCHEME"),
            ("kv_group_size", "MLX_KV_GROUP_SIZE"),
            ("quantized_kv_start", "MLX_QUANTIZED_KV_START"),
        ):
            if key in mlx:
                emit(env, mlx[key])
        if "draft_enabled" in mlx:
            print(f"MLX_DRAFT_ENABLED={'true' if mlx['draft_enabled'] else 'false'}")
        if mlx.get("draft_model"):
            print(f"MLX_DRAFT_MODEL='{quote(mlx['draft_model'])}'")
        if mlx.get("draft_kind"):
            print(f"MLX_DRAFT_KIND='{quote(mlx['draft_kind'])}'")
        if mlx.get("draft_block_size"):
            print(f"MLX_DRAFT_BLOCK_SIZE='{mlx['draft_block_size']}'")
    else:
        print("MLX_SUPPORTED=false")

    # vllm
    vllm = m.get("vllm")
    if vllm:
        print(f"VLLM_MODEL='{quote(vllm['model'])}'")
        print(f"VLLM_QUANTIZATION='{vllm.get('quantization', 'fp8')}'")
        print(f"VLLM_KV_CACHE_DTYPE='{vllm.get('kv_cache_dtype', 'fp8')}'")
        print(f"VLLM_ATTENTION_BACKEND='{vllm.get('attention_backend', 'TRITON_ATTN')}'")
        emit("VLLM_MAX_MODEL_LEN", vllm.get("max_model_len", m.get("context", 65536)))
        if vllm.get("kv_cache_bytes"):
            print(f"VLLM_KV_CACHE_BYTES='{vllm['kv_cache_bytes']}'")
        if vllm.get("gpu_memory_utilization"):
            print(f"VLLM_GPU_MEMORY_UTILIZATION='{vllm['gpu_memory_utilization']}'")
        if vllm.get("max_num_seqs"):
            print(f"VLLM_MAX_NUM_SEQS='{vllm['max_num_seqs']}'")
        if vllm.get("trust_remote_code"):
            print("VLLM_TRUST_REMOTE_CODE=true")
        if vllm.get("tool_call_parser"):
            print(f"VLLM_TOOL_CALL_PARSER='{quote(vllm['tool_call_parser'])}'")
        if vllm.get("reasoning_parser"):
            print(f"VLLM_REASONING_PARSER='{quote(vllm['reasoning_parser'])}'")
        if vllm.get("reasoning_parser_plugin"):
            print(f"VLLM_REASONING_PARSER_PLUGIN='{quote(vllm['reasoning_parser_plugin'])}'")
        if vllm.get("marlin"):
            print("VLLM_MARLIN=true")
        if vllm.get("enable_prefix_caching"):
            print("VLLM_ENABLE_PREFIX_CACHING=true")
    else:
        print("VLLM_SUPPORTED=false")

    # cpu
    cpu = m.get("cpu")
    if cpu:
        print(f"CPU_QUANT='{quote(cpu['quant'])}'")
        emit("CPU_CONTEXT", cpu.get("context", m.get("context", 65536)))
        emit("CPU_PARALLEL", cpu.get("parallel", m.get("parallel", 1)))
        emit("CPU_CACHE_TYPE", cpu.get("cache_type", "q4_0"))
        print(f"CPU_REPO='{quote(cpu.get('repo', llama['repo'] if llama else ''))}'")
        if "flash_attn" in cpu:
            print(f"CPU_FLASH_ATTN={'true' if cpu['flash_attn'] else 'false'}")
        if "checkpoint_min_step" in cpu:
            print(f"CPU_CHECKPOINT_MIN_STEP='{cpu['checkpoint_min_step']}'")
    else:
        print("CPU_SUPPORTED=false")

    # transformers (tasks that llama.cpp does not support)
    transformers = m.get("transformers")
    if transformers:
        print(f"TRANSFORMERS_MODEL='{quote(transformers['model'])}'")
        emit("TRANSFORMERS_TASK", transformers.get("task", "fill-mask"))
        emit("TRANSFORMERS_MAX_LENGTH", transformers.get("max_length", m.get("context", 8192)))
        emit("TRANSFORMERS_THREADS", transformers.get("threads", 4))
        emit("TRANSFORMERS_TOP_K", transformers.get("top_k", 5))
        if transformers.get("trust_remote_code"):
            print("TRANSFORMERS_TRUST_REMOTE_CODE=true")
    else:
        print("TRANSFORMERS_SUPPORTED=false")


if __name__ == "__main__":
    main()
