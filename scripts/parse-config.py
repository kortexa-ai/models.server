#!/usr/bin/env python3
"""Parse model.json and output shell variable assignments.

Usage: eval "$(python3 scripts/parse-config.py <model-dir>/model.json)"
"""
import json
import sys


def quote(v):
    """Shell-safe quoting."""
    return str(v).replace("'", "'\\''")


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
    else:
        print("LLAMA_SUPPORTED=false")

    # mlx
    mlx = m.get("mlx")
    if mlx:
        print(f"MLX_REPO='{quote(mlx['repo'])}'")
        print(f"MLX_BACKEND='{mlx.get('backend', 'mlx_vlm')}'")
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
        print(f"CPU_CONTEXT='{cpu['context']}'")
        print(f"CPU_REPO='{quote(cpu.get('repo', llama['repo'] if llama else ''))}'")
    else:
        print("CPU_SUPPORTED=false")


if __name__ == "__main__":
    main()
