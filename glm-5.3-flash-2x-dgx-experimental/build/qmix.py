"""Minimal lossless8 loader port for pinned vLLM main.

Based on Knapcio's qmix_patch.py at 770d1153062aa916b06591411c61d5c593ea0f03.
Keep the original tensor representation: FP8_BLOCK is block-128 FP8, with
weight_scale_inv, served through vLLM's Fp8LinearMethod. Do not reinterpret it
as ModelOpt's differently named per-block format. No sampling, cache or
scheduler overrides are installed here.
"""
import functools
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys


def patch_modelopt(module):
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    cls = module.ModelOptMixedPrecisionConfig
    original = cls.get_quant_method
    blocked = cls.has_blocked_weights

    def get_method(self, layer, prefix):
        if (isinstance(layer, (LinearBase, ParallelLMHead))
                and not self.is_layer_excluded(prefix)
                and self._resolve_quant_algo(prefix) == "FP8_BLOCK"):
            return Fp8LinearMethod(self.fp8_block_config)
        return original(self, layer, prefix)

    def has_blocked(self):
        return blocked(self) or any(
            value.get("quant_algo") == "FP8_BLOCK"
            for value in self.quantized_layers.values())

    cls.get_quant_method = get_method
    cls.has_blocked_weights = has_blocked
    print("glm53-main: registered FP8_BLOCK dense loader", file=sys.stderr)


def patch_model(module):
    import torch

    original_projection = module._try_load_fp8_attn_proj

    @functools.wraps(original_projection)
    def projection(name, tensor, buf, params_dict, loaded_params, kv_a_pad_size):
        for suffix, (_, target_base, _, _) in module._FP8_ATTN_PROJS.items():
            if suffix in name:
                weight_name = f"{name.rsplit(suffix, 1)[0]}.{target_base}.weight"
                parameter = params_dict.get(weight_name)
                if parameter is not None and parameter.dtype not in (
                        torch.bfloat16, torch.float16, torch.float32):
                    return False
                break
        return original_projection(name, tensor, buf, params_dict,
                                   loaded_params, kv_a_pad_size)

    module._try_load_fp8_attn_proj = projection
    cls = module.Glm5NextForConditionalGeneration
    original_init, original_load = cls.__init__, cls.load_weights

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        object.__setattr__(self, "_qmix_path",
                           kwargs["vllm_config"].model_config.model)

    @functools.wraps(original_load)
    def load_weights(self, weights):
        manifest_path = Path(self._qmix_path) / "qmix-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("zero_copy"):
            raise RuntimeError("Expected the verified zero-copy mixed checkpoint")
        names = set(manifest["quantized_names"])
        counts = {"sentinel": 0, "dropped": 0}

        def filtered():
            for item in weights:
                name, tensor = item[:2]
                if name == manifest["sentinel"]:
                    counts["sentinel"] += 1
                elif name in names and tensor.dtype == torch.bfloat16:
                    counts["dropped"] += 1
                else:
                    yield item

        result = original_load(self, filtered())
        if counts != {"sentinel": 1, "dropped": len(names)}:
            raise RuntimeError(f"Mixed checkpoint filtering mismatch: {counts}")
        print(f"glm53-main: verified mixed checkpoint filtering {counts}",
              file=sys.stderr)
        return result

    cls.__init__, cls.load_weights = initialize, load_weights


class Hook(importlib.abc.MetaPathFinder):
    targets = {
        "vllm.model_executor.layers.quantization.modelopt": patch_modelopt,
        "vllm.models.glm5next.common.model": patch_model,
    }

    def find_spec(self, name, path, target=None):
        if name not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            raise ImportError(f"Required pinned module unavailable: {name}")
        execute = spec.loader.exec_module
        apply = self.targets[name]

        def patched(module):
            execute(module)
            apply(module)

        spec.loader.exec_module = patched
        return spec


def register():
    if not any(isinstance(hook, Hook) for hook in sys.meta_path):
        sys.meta_path.insert(0, Hook())
