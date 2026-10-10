import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import glm53_main_runtime as runtime

sys.path.insert(0, str(ROOT / "glm-5.3-flash-2x-dgx-experimental/build"))
import draft_cache


def model():
    source = json.loads((ROOT / "glm-5.3-flash-2x-dgx/model.json").read_text())
    c = source.pop("glm53_tp2")
    source.update(id="glm-5.3-flash-2x-dgx-experimental", port=2071,
                  default_engine="vllm-glm53-main")
    c.update(master_port=29670, decode_context_parallel_size=1,
             cp_kv_cache_interleave_size=4, container_memory="112g",
             kv_cache_memory_bytes=6 * 1024 ** 3, max_num_batched_tokens=6919,
             mamba_ssm_cache_dtype="float32", attention_backend="FLASHINFER_MLA_SPARSE",
             moe_backend="marlin", linear_backend="marlin", image_limit=8,
             mm_processor_cache_gb=1, reasoning_parser="glm47", tool_call_parser="glm47",
             speculative_config={"method": "dflash", "model": "/draft", "num_speculative_tokens": 7},
             enforce_eager=True, compilation_config={},
             vllm_revision="276fbcff2717bd934cfa37c8a2e4c391f3e7237b",
             native_revision="187a0eb98aa42341d703f83421d693fa7585581b")
    source["glm53_main"] = c
    return source


class ExperimentalLifecycleTest(unittest.TestCase):
    def test_draft_metadata_preserves_bf16_projections_without_editing_source(self):
        m = model()
        config = {"architectures": ["DFlash2DraftModel"], "num_hidden_layers": 5,
                  "num_target_layers": 45, "quantization_config": {
                      "quant_method": "fp8", "weight_block_size": [128, 128],
                      "ignored_layers": ["model.fc", "fc"]}}
        with tempfile.TemporaryDirectory() as temporary:
            m["glm53_main"]["draft_path"] = temporary
            source = Path(temporary) / "config.json"
            original = json.dumps(config)
            source.write_text(original)
            corrected = json.loads(runtime.draft_config_bytes(m))
            self.assertEqual(source.read_text(), original)
        ignored = corrected["quantization_config"].pop("ignored_layers")
        self.assertEqual(len(ignored), 22)
        self.assertIn("model.layers.45.attention_conv.kernel_projection", ignored)
        self.assertIn("model.layers.49.mlp_conv.kernel_projection", ignored)
        self.assertIn("model.layers.0.attention_conv.kernel_projection", ignored)
        config["quantization_config"].pop("ignored_layers")
        self.assertEqual(corrected, config)

    def test_draft_builder_keeps_own_precision_and_dimensions_without_mutating_target(self):
        class Group:
            def create_metadata_builders(self, vllm_config, device):
                return vllm_config

        module = SimpleNamespace(AttentionGroup=Group)
        draft_cache.install_builders(module)
        draft_model, draft_parallel = object(), object()
        original_cache = SimpleNamespace(cache_dtype="fp8_ds_mla", kv_cache_layout="BLHNC")
        target = SimpleNamespace(
            speculative_config=SimpleNamespace(method="dflash", kv_cache_dtype="fp8_e4m3",
                draft_model_config=draft_model, draft_parallel_config=draft_parallel),
            compilation_config=SimpleNamespace(static_forward_context={
                "draft": SimpleNamespace(is_draft_layer=True),
                "target": SimpleNamespace(is_draft_layer=False)}),
            cache_config=original_cache, model_config=object(), parallel_config=object())
        group = Group()
        group.layer_names = ["draft"]
        selected = group.create_metadata_builders(target, "cuda")
        self.assertEqual(selected.cache_config.cache_dtype, "fp8_e4m3")
        self.assertEqual(selected.cache_config.kv_cache_layout, "BLHNC")
        self.assertIs(selected.model_config, draft_model)
        self.assertIs(selected.parallel_config, draft_parallel)
        self.assertEqual(target.cache_config.cache_dtype, "fp8_ds_mla")
        self.assertIs(target.cache_config, original_cache)
        group.layer_names = ["target"]
        self.assertIs(group.create_metadata_builders(target, "cuda"), target)

    def test_dcp_uses_existing_two_ranks_and_keeps_worker_headless(self):
        m = model()
        m["glm53_main"]["decode_context_parallel_size"] = 2
        for rank in (0, 1):
            args = runtime.render_command(m, rank)
            self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "2")
            self.assertEqual(args[args.index("--decode-context-parallel-size") + 1], "2")
            self.assertEqual(args[args.index("--cp-kv-cache-interleave-size") + 1], "4")
            self.assertEqual("--headless" in args, bool(rank))
            self.assertIn("VLLM_USE_RUST_FRONTEND=0", args)

    def test_setup_rejects_image_with_wrong_source_even_if_id_matches(self):
        m = model()
        image = {"Id": m["glm53_main"]["image_id"], "Config": {"Labels": {}}}
        with patch.object(runtime.stable, "output", return_value=json.dumps([image])):
            with self.assertRaisesRegex(ValueError, "provenance"):
                runtime.image_info(m)

    def test_stable_container_is_not_owned_by_experiment(self):
        m = model()
        info = {"Image": m["glm53_main"]["image_id"], "Config": {"Labels": {
            runtime.LABEL + ".model": "glm-5.3-flash-2x-dgx",
            runtime.LABEL + ".fingerprint": runtime.fingerprint(m),
            runtime.LABEL + ".rank": "0"}}}
        self.assertFalse(runtime.owned(m, info, 0))

    def test_stale_setup_is_rejected_before_container_creation(self):
        m = model()
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "receipt.json"
            receipt.write_text(json.dumps({"fingerprint": "old", "files": {}}))
            with patch.object(runtime, "receipt_path", return_value=receipt), \
                    patch.object(runtime.stable, "node", return_value={"rank": 0}), \
                    patch.object(runtime.stable, "inspect", return_value=None), \
                    patch.object(runtime.stable, "execute") as execute:
                with self.assertRaisesRegex(ValueError, "explicit setup"):
                    runtime.run(m, "run")
                execute.assert_not_called()

    def test_stop_selects_only_experimental_model_and_current_rank(self):
        m = model()
        with patch.object(runtime.stable, "node", return_value={"rank": 1}), \
                patch.object(runtime.stable, "output", return_value="") as output, \
                patch.object(runtime.stable, "execute") as execute:
            runtime.run(m, "stop")
            args = output.call_args.args[0]
            self.assertIn("label=" + runtime.LABEL + ".model=" + m["id"], args)
            self.assertIn("label=" + runtime.LABEL + ".rank=1", args)
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
