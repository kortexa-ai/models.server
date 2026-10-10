#!/usr/bin/env python3
"""Replay the pinned allocator on observed specs without loading model weights.

The output is a layout estimate, not a tested serving capacity. Run inside the
same image that produced the input audit. No CUDA allocation is required.
"""
import argparse
from collections import Counter
from dataclasses import fields, replace
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from vllm.config import CacheConfig
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backend import MultipleOf
from vllm.v1 import kv_cache_interface as specs_module
from vllm.v1.core import kv_cache_utils as planner


def restore(value):
    if isinstance(value, list):
        return tuple(restore(item) for item in value)
    if isinstance(value, dict):
        if set(value) == {"base"}:
            return MultipleOf(value["base"])
        return {key: restore(item) for key, item in value.items()}
    if isinstance(value, str):
        if value.startswith("torch."):
            return getattr(torch, value.removeprefix("torch."))
        if value.startswith("MambaAttentionBackendEnum."):
            return MambaAttentionBackendEnum[value.split(".")[-1]]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--in-flight-batches", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=6919)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    specs = {name: getattr(specs_module, item["type"])(**restore(item["fields"]))
             for name, item in audit["input_specs"][0].items()}
    accepted_fields = {item.name for item in fields(CacheConfig) if item.init}
    cache = CacheConfig(**{key: value for key, value in audit["cache_config"].items()
                           if key in accepted_fields})
    cache.kv_cache_layout = audit["cache_config"]["kv_cache_layout"]
    config = SimpleNamespace(
        cache_config=cache,
        model_config=SimpleNamespace(max_model_len=audit["max_model_len"],
                                     hf_config=SimpleNamespace(model_type="glm5next")),
        parallel_config=SimpleNamespace(decode_context_parallel_size=audit["dcp"],
                                        tensor_parallel_size=audit["tp"],
                                        prefill_context_parallel_size=1),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=SimpleNamespace(method="dflash", use_eagle=lambda: True,
                                           use_eagle_block_drop=lambda: True),
        max_in_flight_tokens=args.in_flight_batches * args.max_num_batched_tokens)
    for name, spec in specs.items():
        assert spec.max_memory_usage_bytes(config) == audit["input_specs"][0][name]["max_memory_usage_bytes"], (
            f"Input plan mismatch for {name}; check the in-flight batch count")

    def calculate(label, current_specs):
        groups = planner.get_kv_cache_groups(config, dict(current_specs))
        pool_block = planner._pool_bytes_per_block(config, groups)
        needed = planner._max_memory_usage_bytes_from_groups(config, groups)
        details = []
        for group in groups:
            spec = group.kv_cache_spec
            pages = (spec.max_memory_usage_pages(config)
                     if isinstance(spec, specs_module.UniformTypeKVCacheSpecs)
                     else (spec.max_memory_usage_bytes(config) + spec.page_size_bytes - 1)
                     // spec.page_size_bytes)
            details.append({"layers": group.layer_names,
                            "types": dict(Counter(type(s).__name__ for s in
                                                  specs_module.iter_layer_specs(spec))),
                            "page_bytes": spec.page_size_bytes,
                            "request_blocks": pages})
        return {"label": label, "layout": cache.kv_cache_layout,
                "dcp": config.parallel_config.decode_context_parallel_size,
                "context": config.model_config.max_model_len,
                "max_in_flight_tokens": config.max_in_flight_tokens,
                "pool_bytes_per_block": pool_block,
                "minimum_bytes_including_null": needed + pool_block,
                "minimum_gib_including_null": (needed + pool_block) / 1024**3,
                "groups": details}

    results = [calculate("observed", specs)]
    for dcp in (1, 2):
        config.parallel_config.decode_context_parallel_size = dcp
        changed = {name: replace(spec, block_size=4608, block_stride_alignment=None,
                                dcp_sharded=dcp == 1)
                   if isinstance(spec, specs_module.SlidingWindowSpec) else spec
                   for name, spec in specs.items()}
        results.append(calculate(f"triton-draft-estimate-dcp{dcp}", changed))
    for block in (256, 512, 1024, 2304, 4608):
        changed = {name: replace(spec, block_size=block)
                   if isinstance(spec, specs_module.SlidingWindowSpec) else spec
                   for name, spec in specs.items()}
        for dcp in (1, 2):
            config.parallel_config.decode_context_parallel_size = dcp
            current = {name: replace(spec, dcp_sharded=False)
                       if isinstance(spec, specs_module.SlidingWindowSpec) else spec
                       for name, spec in changed.items()} if dcp > 1 else changed
            try:
                results.append(calculate(f"draft-block{block}-dcp{dcp}", current))
            except (ValueError, AssertionError, NotImplementedError) as error:
                results.append({"label": f"draft-block{block}-dcp{dcp}", "error": str(error)})
    for layout in ("BLHNC", "LBHNC"):
        cache.kv_cache_layout = layout
        for dcp in (1, 2):
            config.parallel_config.decode_context_parallel_size = dcp
            for drafts in (7, 3, 1):
                changed = {name: replace(spec, num_speculative_blocks=drafts)
                           if isinstance(spec, specs_module.MambaSpec) else spec
                           for name, spec in specs.items()}
                if dcp > 1:
                    changed = {name: replace(spec, dcp_sharded=False)
                               if isinstance(spec, specs_module.SlidingWindowSpec) else spec
                               for name, spec in changed.items()}
                label = f"{layout}-dcp{dcp}-mamba-lookahead{drafts}"
                try:
                    results.append(calculate(label, changed))
                except (ValueError, AssertionError, NotImplementedError) as error:
                    results.append({"label": label, "error": str(error)})
    args.output.write_text(json.dumps({"fingerprint": audit["fingerprint"],
        "scope": "allocator estimates only; backend and model tests still required",
        "variants": results}, indent=2) + "\n")
    print(json.dumps([{key: value for key, value in result.items() if key != "groups"}
                      for result in results], indent=2))


if __name__ == "__main__":
    main()
