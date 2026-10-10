"""Enable only the mixed-weight adapter in the experimental image."""
import qmix
import cache_audit

qmix.Hook.targets["vllm.v1.core.kv_cache_utils"] = cache_audit.install
qmix.Hook.targets["vllm.v1.worker.gpu_worker"] = cache_audit.install_worker
qmix.register()
