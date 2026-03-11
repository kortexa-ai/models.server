#!/usr/bin/env bash
# Build TRT-LLM from source on bare metal (DGX Spark / Blackwell aarch64).
#
# Extracts the same logic as Dockerfile.main-source but runs natively.
# Installs into /opt/TensorRT-LLM with a dedicated venv.
#
# Prerequisites (installed by this script if missing):
#   - CUDA toolkit 13.x  (already on DGX Spark)
#   - Python 3.12
#   - TensorRT, NCCL, OpenMPI dev libs (from DGX/CUDA apt repos)
#
# Usage:
#   sudo ./trtllm.spark/build-bare-metal.sh          # full build
#   JOB_COUNT=8 sudo ./trtllm.spark/build-bare-metal.sh   # parallel jobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRTLLM_REPO="${TRTLLM_REPO:-https://github.com/NVIDIA/TensorRT-LLM.git}"
TRTLLM_REF="${TRTLLM_REF:-main}"
INSTALL_DIR="${INSTALL_DIR:-/opt/TensorRT-LLM}"
VENV_DIR="${VENV_DIR:-/opt/trtllm-venv}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-120-real}"
JOB_COUNT="${JOB_COUNT:-4}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==5.3.0}"

echo "=== TRT-LLM Bare Metal Build ==="
echo "Repo:      ${TRTLLM_REPO}"
echo "Branch:    ${TRTLLM_REF}"
echo "Install:   ${INSTALL_DIR}"
echo "Venv:      ${VENV_DIR}"
echo "CUDA arch: ${CUDA_ARCHITECTURES}"
echo "Jobs:      ${JOB_COUNT}"
echo ""

# --- Step 1: System dependencies ---
echo "--- Step 1: System dependencies ---"
apt-get update
apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    git git-lfs \
    cmake \
    build-essential \
    libopenmpi-dev openmpi-bin \
    libnvinfer-dev libnvinfer-plugin10 libnvinfer-headers-dev \
    libnccl-dev libnccl2 \
    libcudnn9-dev-cuda-13

echo ""

# --- Step 2: Create venv ---
echo "--- Step 2: Python venv ---"
if [[ ! -d "${VENV_DIR}" ]]; then
    python3.12 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel

# Install PyTorch (stock 2.10 supports sm_120 on aarch64)
pip install torch==2.10.0

# Install transformers
pip install "${TRANSFORMERS_SPEC}"

echo ""

# --- Step 3: Clone TRT-LLM ---
echo "--- Step 3: Clone TRT-LLM ---"
if [[ -d "${INSTALL_DIR}" ]]; then
    echo "Existing ${INSTALL_DIR} found — pulling latest"
    cd "${INSTALL_DIR}"
    git fetch origin "${TRTLLM_REF}"
    git checkout FETCH_HEAD
else
    GIT_LFS_SKIP_SMUDGE=1 git \
        -c filter.lfs.smudge= \
        -c filter.lfs.required=false \
        clone --depth 1 --branch "${TRTLLM_REF}" "${TRTLLM_REPO}" "${INSTALL_DIR}"
fi
cd "${INSTALL_DIR}"

# Pull LFS files (needed for source build — binary deps like nvshmem)
git lfs install --local
git lfs pull

echo ""

# --- Step 4: Apply transformers 5.3 compatibility patches ---
echo "--- Step 4: Applying compatibility patches ---"
python3 - <<'PY'
from pathlib import Path

repo = Path("/opt/TensorRT-LLM")

replacements = {
    repo / "tensorrt_llm/models/gpt/convert.py": [
        (
            "from transformers import (AutoModelForCausalLM, AutoModelForVision2Seq,\n"
            "                          AutoTokenizer)\n",
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "try:\n"
            "    from transformers import AutoModelForVision2Seq\n"
            "except ImportError:\n"
            "    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq\n",
        ),
    ],
    repo / "tensorrt_llm/tools/multimodal_builder.py": [
        (
            "from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,\n"
            "                          AutoModelForVision2Seq, AutoProcessor,\n",
            "from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor\n"
            "try:\n"
            "    from transformers import AutoModelForVision2Seq\n"
            "except ImportError:\n"
            "    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq\n",
        ),
    ],
    repo / "tensorrt_llm/_torch/models/modeling_clip.py": [
        (
            "from transformers.modeling_utils import (get_parameter_device,\n"
            "                                         get_parameter_dtype)\n",
            "",
        ),
        (
            "from .modeling_utils import _load_weights_impl, register_auto_model\n",
            "from .modeling_utils import _load_weights_impl, register_auto_model\n"
            "\n"
            "\n"
            "def _get_parameter_device(module: nn.Module) -> torch.device:\n"
            "    return next(param.device for param in module.parameters())\n"
            "\n"
            "\n"
            "def _get_parameter_dtype(module: nn.Module) -> torch.dtype:\n"
            "    return next(param.dtype for param in module.parameters()\n"
            "                if param.is_floating_point())\n",
        ),
        (
            "        return get_parameter_dtype(self)\n",
            "        return _get_parameter_dtype(self)\n",
        ),
        (
            "        return get_parameter_device(self)\n",
            "        return _get_parameter_device(self)\n",
        ),
    ],
    repo / "tensorrt_llm/_torch/models/modeling_siglip.py": [
        (
            "from transformers.modeling_utils import (get_parameter_device,\n"
            "                                         get_parameter_dtype)\n",
            "",
        ),
        (
            "from .modeling_utils import _load_weights_impl, register_auto_model\n",
            "from .modeling_utils import _load_weights_impl, register_auto_model\n"
            "\n"
            "\n"
            "def _get_parameter_device(module: nn.Module) -> torch.device:\n"
            "    return next(param.device for param in module.parameters())\n"
            "\n"
            "\n"
            "def _get_parameter_dtype(module: nn.Module) -> torch.dtype:\n"
            "    return next(param.dtype for param in module.parameters()\n"
            "                if param.is_floating_point())\n",
        ),
        (
            "        return get_parameter_dtype(self)\n",
            "        return _get_parameter_dtype(self)\n",
        ),
        (
            "        return get_parameter_device(self)\n",
            "        return _get_parameter_device(self)\n",
        ),
    ],
    repo / "tensorrt_llm/_torch/visual_gen/models/wan/transformer_wan.py": [
        (
            "from transformers.modeling_utils import get_parameter_device\n",
            "",
        ),
        (
            "from tensorrt_llm.models.modeling_utils import QuantConfig\n",
            "from tensorrt_llm.models.modeling_utils import QuantConfig\n"
            "\n"
            "\n"
            "def _get_parameter_device(module: nn.Module) -> torch.device:\n"
            "    return next(param.device for param in module.parameters())\n",
        ),
        (
            "        return get_parameter_device(self)\n",
            "        return _get_parameter_device(self)\n",
        ),
    ],
    repo / "tensorrt_llm/_torch/speculative/suffix_automaton.py": [
        (
            "from tensorrt_llm.bindings.internal import suffix_automaton as _sa_native\n",
            "try:\n"
            "    from tensorrt_llm.bindings.internal import suffix_automaton as _sa_native\n"
            "except ImportError:\n"
            "    _sa_native = None\n",
        ),
    ],
    repo / "tensorrt_llm/_torch/models/modeling_exaone4.py": [
        (
            "    AutoConfig.register(Exaone4Config.model_type, Exaone4Config)\n",
            "    try:\n"
            "        AutoConfig.register(Exaone4Config.model_type, Exaone4Config)\n"
            "    except ValueError:\n"
            "        pass\n",
        ),
    ],
    repo / "tensorrt_llm/_torch/models/modeling_exaone_moe.py": [
        (
            "AutoConfig.register(ExaoneMoEConfig.model_type, ExaoneMoEConfig)\n",
            "try:\n"
            "    AutoConfig.register(ExaoneMoEConfig.model_type, ExaoneMoEConfig)\n"
            "except ValueError:\n"
            "    pass\n",
        ),
    ],
    repo / "tensorrt_llm/_torch/models/modeling_nemotron_h.py": [
        (
            "AutoConfig.register(NemotronHConfig.model_type, NemotronHConfig)\n",
            "try:\n"
            "    AutoConfig.register(NemotronHConfig.model_type, NemotronHConfig)\n"
            "except ValueError:\n"
            "    pass\n",
        ),
    ],
    repo / "tensorrt_llm/_torch/models/modeling_vila.py": [
        (
            "AutoConfig.register(VilaConfig.model_type, VilaConfig)\n"
            "AutoModel.register(VilaConfig, VilaModel)\n",
            "try:\n"
            "    AutoConfig.register(VilaConfig.model_type, VilaConfig)\n"
            "except ValueError:\n"
            "    pass\n"
            "try:\n"
            "    AutoModel.register(VilaConfig, VilaModel)\n"
            "except ValueError:\n"
            "    pass\n",
        ),
    ],
    repo / "tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3_5_moe.py": [
        (
            "AutoConfig.register(\"qwen3_5_moe\", Qwen3_5MoeConfig)\n"
            "AutoConfig.register(\"qwen3_5_moe_text\", Qwen3_5MoeTextConfig)\n",
            "try:\n"
            "    AutoConfig.register(\"qwen3_5_moe\", Qwen3_5MoeConfig)\n"
            "except ValueError:\n"
            "    pass\n"
            "try:\n"
            "    AutoConfig.register(\"qwen3_5_moe_text\", Qwen3_5MoeTextConfig)\n"
            "except ValueError:\n"
            "    pass\n",
        ),
    ],
    repo / "requirements.txt": [
        (
            "transformers==4.57.1\n",
            "transformers==5.3.0\n",
        ),
    ],
    repo / "tensorrt_llm/_torch/models/modeling_llama.py": [
        (
            "from transformers.modeling_utils import load_sharded_checkpoint\n",
            "try:\n"
            "    from transformers.modeling_utils import load_sharded_checkpoint\n"
            "except ImportError:\n"
            "    from transformers.modeling_utils import load_state_dict\n"
            "    from transformers.utils.hub import get_checkpoint_shard_files\n"
            "\n"
            "    def load_sharded_checkpoint(model, folder, strict=False):\n"
            "        index_file = os.path.join(folder, 'model.safetensors.index.json')\n"
            "        if not os.path.exists(index_file):\n"
            "            index_file = os.path.join(folder, 'pytorch_model.bin.index.json')\n"
            "        shard_files, _ = get_checkpoint_shard_files(folder, index_file)\n"
            "        for shard_file in shard_files:\n"
            "            state_dict = load_state_dict(shard_file)\n"
            "            model.load_state_dict(state_dict, strict=strict)\n"
            "            del state_dict\n",
        ),
    ],
    repo / "tensorrt_llm/_torch/autotuner.py": [
        (
            "from tensorrt_llm.bindings.internal.runtime import (delay_kernel,\n"
            "                                                    record_global_timer)\n",
            "try:\n"
            "    from tensorrt_llm.bindings.internal.runtime import (\n"
            "        delay_kernel,\n"
            "        record_global_timer,\n"
            "    )\n"
            "except ImportError:\n"
            "    from tensorrt_llm.bindings.internal.runtime import delay_kernel\n"
            "    record_global_timer = None\n",
        ),
        (
            "        if timer_env == \"globaltimer\":\n"
            "            self._use_global_timer = True\n"
            "        elif timer_env == \"cuda_event\":\n"
            "            self._use_global_timer = False\n"
            "        else:\n"
            "            self._use_global_timer = confidential_compute_enabled()\n",
            "        if timer_env == \"globaltimer\":\n"
            "            self._use_global_timer = record_global_timer is not None\n"
            "        elif timer_env == \"cuda_event\":\n"
            "            self._use_global_timer = False\n"
            "        else:\n"
            "            self._use_global_timer = (\n"
            "                record_global_timer is not None and confidential_compute_enabled()\n"
            "            )\n"
            "\n"
            "        if record_global_timer is None and timer_env == \"globaltimer\":\n"
            "            logger.warning(\n"
            "                \"[Autotuner] record_global_timer binding unavailable; falling back to cuda_event timing.\"\n"
            "            )\n",
        ),
    ],
}

patched = 0
skipped = 0
for path, file_replacements in replacements.items():
    if not path.exists():
        print(f"SKIP (not found): {path}")
        skipped += 1
        continue
    text = path.read_text()
    changed = False
    for old, new in file_replacements:
        if old not in text:
            # Already patched or source changed upstream
            print(f"  SKIP snippet in {path.name}: {old[:60]!r}...")
            continue
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        path.write_text(text)
        print(f"PATCHED: {path}")
        patched += 1

print(f"\nPatches: {patched} files patched, {skipped} skipped")
PY

# DisabledTqdm fix
sed -i 's/super().__init__(\*args, \*\*kwargs, disable=True)/kwargs.pop("disable", None); super().__init__(*args, **kwargs, disable=True)/' \
    "${INSTALL_DIR}/tensorrt_llm/llmapi/utils.py"

# rope_type="default" aliases
sed -i 's/return PositionEmbeddingType\[s\]/return PositionEmbeddingType[{"default":"rope_gpt_neox"}.get(s,s)]/' \
    "${INSTALL_DIR}/tensorrt_llm/functional.py"
sed -i 's/return RotaryScalingType\[s\]/return RotaryScalingType[{"default":"none"}.get(s,s)]/' \
    "${INSTALL_DIR}/tensorrt_llm/functional.py"

echo ""

# --- Step 5: Install TRT-LLM Python deps ---
echo "--- Step 5: Install TRT-LLM Python deps ---"
cd "${INSTALL_DIR}"
pip install -r requirements.txt || echo "WARNING: Some requirements failed (expected — some are NVIDIA-internal)"

echo ""

# --- Step 6: Build C++ libs ---
echo "--- Step 6: Build C++ (this takes a while) ---"
cd "${INSTALL_DIR}"
python3 scripts/build_wheel.py \
    --cuda_architectures "${CUDA_ARCHITECTURES}" \
    --job_count "${JOB_COUNT}" \
    --build_type Release \
    --no-venv \
    --skip-stubs \
    --skip_building_wheel

echo ""

# --- Step 7: Copy compiled .so into source tree ---
echo "--- Step 7: Installing compiled libs ---"
mkdir -p tensorrt_llm/libs
find cpp/build -name '*.so' -type f | while read -r f; do
    name="$(basename "$f")"
    case "$name" in
        libtensorrt_llm.so|libnvinfer_plugin_tensorrt_llm.so|libth_common.so| \
        libdecoder_attention_*.so|libpg_utils.so| \
        libtensorrt_llm_mooncake_wrapper.so|libtensorrt_llm_nixl_wrapper.so| \
        libtensorrt_llm_ucx_wrapper.so)
            echo "  $f -> tensorrt_llm/libs/$name"
            cp "$f" "tensorrt_llm/libs/$name" ;;
    esac
done
echo "=== Libs in tensorrt_llm/libs/ ==="
ls -lh tensorrt_llm/libs/*.so

echo ""

# --- Step 8: Create activation script ---
echo "--- Step 8: Creating activation script ---"
cat > "${INSTALL_DIR}/activate.sh" <<'ACTIVATE'
#!/usr/bin/env bash
# Source this to use TRT-LLM: source /opt/TensorRT-LLM/activate.sh
source /opt/trtllm-venv/bin/activate
export PYTHONPATH="/opt/TensorRT-LLM${PYTHONPATH:+:${PYTHONPATH}}"
echo "TRT-LLM activated (venv + PYTHONPATH)"
ACTIVATE
chmod +x "${INSTALL_DIR}/activate.sh"

echo ""
echo "=== Build Complete ==="
echo ""
echo "To use TRT-LLM:"
echo "  source ${INSTALL_DIR}/activate.sh"
echo ""
echo "To serve a model:"
echo "  python3 -m tensorrt_llm.commands.serve serve Qwen/Qwen3-4B \\"
echo "    --host 0.0.0.0 --port 2250 --backend pytorch --tp_size 1 --max_seq_len 32768"
echo ""
