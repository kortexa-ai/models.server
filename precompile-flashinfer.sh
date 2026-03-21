#!/bin/bash
set -euo pipefail

# Pre-compile FlashInfer CUTLASS SM120 kernels for NVFP4 models on DGX Spark.
# Run this after clearing ~/.cache/flashinfer/ or installing a new vLLM/FlashInfer.
# Single-threaded to avoid OOMing the Spark.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${SCRIPT_DIR}/bench/.venv-vllm"

if [[ ! -d "${VENV}" ]]; then
    echo "Error: vLLM venv not found at ${VENV}" >&2
    exit 1
fi

source "${VENV}/bin/activate"

export MAX_JOBS="${MAX_JOBS:-1}"

echo "Pre-compiling FlashInfer SM120 kernels (MAX_JOBS=${MAX_JOBS})..."
echo "Kernels will be cached at ~/.cache/flashinfer/"
echo ""

python3 -c "
import time

kernels = [
    ('FP4 GEMM (CUTLASS SM120)', lambda: __import__('flashinfer.gemm.gemm_base', fromlist=['get_gemm_sm120_module_cutlass_fp4']).get_gemm_sm120_module_cutlass_fp4()),
    ('Fused MOE (CUTLASS SM120)', lambda: __import__('flashinfer.fused_moe.core', fromlist=['get_cutlass_fused_moe_module']).get_cutlass_fused_moe_module('120')),
    ('FP8 GEMM (CUTLASS SM120)', lambda: __import__('flashinfer.gemm.gemm_base', fromlist=['get_gemm_sm120_module_cutlass_fp8']).get_gemm_sm120_module_cutlass_fp8()),
]

for name, build_fn in kernels:
    print(f'  Compiling {name}...', end=' ', flush=True)
    t0 = time.time()
    try:
        build_fn()
        print(f'done ({time.time() - t0:.1f}s)')
    except Exception as e:
        print(f'FAILED: {e}')

print()
print('All done!')
"
