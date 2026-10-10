"""Build DFlash attention metadata with the draft's dimensions and precision."""
import copy
import functools


def install_builders(module):
    original = module.AttentionGroup.create_metadata_builders

    @functools.wraps(original)
    def create(self, vllm_config, *args, **kwargs):
        spec = vllm_config.speculative_config
        context = vllm_config.compilation_config.static_forward_context
        if (spec is not None and spec.method == "dflash" and spec.kv_cache_dtype
                and self.layer_names and all(
                    getattr(context[name], "is_draft_layer", False)
                    for name in self.layer_names)):
            vllm_config = copy.copy(vllm_config)
            vllm_config.model_config = spec.draft_model_config
            vllm_config.parallel_config = spec.draft_parallel_config
            vllm_config.cache_config = copy.copy(vllm_config.cache_config)
            vllm_config.cache_config.cache_dtype = spec.kv_cache_dtype
        return original(self, vllm_config, *args, **kwargs)

    module.AttentionGroup.create_metadata_builders = create
