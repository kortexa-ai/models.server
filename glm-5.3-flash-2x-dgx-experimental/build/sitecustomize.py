"""Enable only the mixed-weight adapter in the experimental image."""
import qmix
import cache_audit
import os
import sys

qmix.Hook.targets["vllm.v1.core.kv_cache_utils"] = cache_audit.install
qmix.Hook.targets["vllm.v1.worker.gpu_worker"] = cache_audit.install_worker
qmix.register()

# A separate, explicitly enabled transport comparison. The first upstream
# baseline leaves this off. It does not install any old model/cache overlays.
if os.environ.get("GLM_ROCE_ALLREDUCE") == "1":
    sys.path.insert(0, "/opt/glm-roce")
    import glm_roce.boot
