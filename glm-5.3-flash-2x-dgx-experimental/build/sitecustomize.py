"""Enable only the mixed-weight adapter in the experimental image."""
import qmix
import cache_audit
import os
import sys

qmix.Hook.targets["vllm.v1.core.kv_cache_utils"] = cache_audit.install
qmix.Hook.targets["vllm.v1.worker.gpu_worker"] = cache_audit.install_worker
if os.environ.get("GLM_SM120_DCP") == "1":
    import sm120_dcp
    qmix.Hook.targets[
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120"
    ] = sm120_dcp.install
qmix.register()

# Byte-preserving bounded staging for GB10's slow file-backed host copies.
# Keep this separate from the first unmodified-loader startup measurement.
if os.environ.get("GLM_FAST_LOAD") == "1":
    import glm_fast_load
    glm_fast_load.register()

# A separate, explicitly enabled transport comparison. The first upstream
# baseline leaves this off. It does not install any old model/cache overlays.
if os.environ.get("GLM_ROCE_ALLREDUCE") == "1":
    sys.path.insert(0, "/opt/glm-roce")
    import glm_roce.boot
