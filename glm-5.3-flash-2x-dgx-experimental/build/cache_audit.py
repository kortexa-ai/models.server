"""Observe the upstream cache plan without changing allocation or scheduling."""
from dataclasses import asdict
import functools
import json
import os
from pathlib import Path
import time


def install(module):
    original = module.get_kv_cache_configs

    @functools.wraps(original)
    def observed(vllm_config, kv_cache_specs, available_memory):
        configs = original(vllm_config, kv_cache_specs, available_memory)
        if not os.environ.get("GLM_EXPERIMENT_FINGERPRINT"):
            return configs
        workers = []
        for rank, config in enumerate(configs):
            workers.append({
                "rank_index": rank,
                "available_bytes": available_memory[rank],
                "allocated_tensor_bytes": sum(t.size for t in config.kv_cache_tensors),
                "pool_bytes_per_block": module._pool_bytes_per_block(
                    vllm_config, config.kv_cache_groups),
                "config": asdict(config),
            })
        record = {"fingerprint": os.environ["GLM_EXPERIMENT_FINGERPRINT"],
                  "dcp": vllm_config.parallel_config.decode_context_parallel_size,
                  "tp": vllm_config.parallel_config.tensor_parallel_size,
                  "max_model_len": vllm_config.model_config.max_model_len,
                  "workers": workers}
        directory = Path("/cache/layout-audit")
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{record['fingerprint']}-{time.time_ns()}.json"
        destination.write_text(json.dumps(record, default=str, indent=2))
        print("glm53-main: cache allocation audit " + str(destination), flush=True)
        return configs

    module.get_kv_cache_configs = observed


def install_worker(module):
    def wrap(original, stage):
        @functools.wraps(original)
        def measured(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            fingerprint = os.environ.get("GLM_EXPERIMENT_FINGERPRINT")
            if not fingerprint:
                return result
            import torch

            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            host = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                name, value = line.split(":", 1)
                if name in ("MemAvailable", "MemFree", "SwapFree", "SwapTotal"):
                    host[name + "_bytes"] = int(value.split()[0]) * 1024
            record = {"fingerprint": fingerprint, "stage": stage, "rank": self.rank,
                      "pid": os.getpid(), "cuda_free_bytes": free, "cuda_total_bytes": total,
                      "torch_allocated_bytes": torch.cuda.memory_allocated(),
                      "torch_reserved_bytes": torch.cuda.memory_reserved(),
                      "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "host": host}
            directory = Path("/cache/memory-audit")
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"{fingerprint}-r{self.rank}-{stage}-{time.time_ns()}.json"
            destination.write_text(json.dumps(record, indent=2))
            print("glm53-main: worker memory audit " + str(destination), flush=True)
            return result

        return measured

    for method in ("load_model", "initialize_from_config", "compile_or_warm_up_model"):
        setattr(module.Worker, method, wrap(getattr(module.Worker, method), method))
