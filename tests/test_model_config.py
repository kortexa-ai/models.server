import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "hy-mt2-7b"

EXPECTED_LLAMA_CAPACITY = {
    "qwen-3.5-0.8b": (524288, 8),
    "qwen-3.5-2b": (524288, 8),
    "qwen-3.5-4b": (524288, 8),
    "qwen-3.5-9b": (524288, 8),
    "qwen-3.6-27b": (524288, 8),
    "qwen-3.6-35b-a3b": (524288, 8),
    "qwen-3.8-27b": (524288, 8),
    "qwen-3.8-27b-uncensored": (524288, 8),
    "ornith-1.5-9b": (524288, 8),
    "ornith-1.5-35b-a3b": (524288, 8),
    "gemma-4-e2b": (262144, 8),
    "gemma-4-e4b": (262144, 8),
    "gemma-4-12b": (262144, 8),
    "gemma-4-26b-a4b": (524288, 8),
    "gemma-4-31b": (524288, 8),
    "lfm2-350m-extract": (65536, 16),
    "lfm2.5-230m": (65536, 16),
    "lfm2.5-350m": (65536, 16),
    "lfm2.5-1.2b-instruct": (65536, 16),
    "lfm2.5-1.2b-thinking": (65536, 16),
    "lfm2.5-vl-450m": (65536, 16),
    "lfm2.5-vl-3b": (65536, 16),
    "lfm2.5-2.6b": (131072, 8),
    "lfm2.5-8b-a1b": (128000, 8),
    "qwen3-embedding-0.6b": (131072, 4),
    "embeddinggemma-300m": (8192, 4),
    "lfm2.5-embedding-350m": (2048, 4),
    "hy-mt2-7b": (16384, 8),
}

EXPECTED_CONTEXT_WINDOWS = {
    "audio8-tts-0.6b": 2048,
    "embeddinggemma-300m": 2048,
    "gemma-4-12b": 131072,
    "gemma-4-26b-a4b": 262144,
    "gemma-4-31b": 262144,
    "gemma-4-e2b": 131072,
    "gemma-4-e4b": 131072,
    "hy-mt2-7b": 8192,
    "lfm2-350m-extract": 32768,
    "lfm2.5-1.2b-instruct": 32768,
    "lfm2.5-1.2b-thinking": 32768,
    "lfm2.5-2.6b": 131072,
    "lfm2.5-230m": 32768,
    "lfm2.5-350m": 32768,
    "lfm2.5-8b-a1b": 128000,
    "lfm2.5-embedding-350m": 512,
    "lfm2.5-encoder-350m": 8192,
    "lfm2.5-vl-3b": 32768,
    "lfm2.5-vl-450m": 32768,
    "ornith-1.5-35b-a3b": 262144,
    "ornith-1.5-9b": 262144,
    "qwen-3.5-0.8b": 262144,
    "qwen-3.5-2b": 262144,
    "qwen-3.5-4b": 262144,
    "qwen-3.5-9b": 262144,
    "qwen-3.6-27b": 262144,
    "qwen-3.6-35b-a3b": 262144,
    "qwen-3.8-27b": 262144,
    "qwen-3.8-27b-uncensored": 262144,
    "qwen3-embedding-0.6b": 32768,
    "qwen3-tts-0.6b-customvoice": 32768,
    "qwen3-tts-1.7b-customvoice": 32768,
}


class LlamaCapacityPolicyTest(unittest.TestCase):
    def test_family_capacity_policy(self):
        for model_id, expected in EXPECTED_LLAMA_CAPACITY.items():
            with self.subTest(model_id=model_id):
                config = json.loads((ROOT / model_id / "model.json").read_text())
                llama = config["llama"]
                self.assertEqual((llama["context"], llama["parallel"]), expected)

    def test_every_model_declares_its_advertised_context_window(self):
        config_paths = sorted(ROOT.glob("*/model.json"))
        self.assertEqual({path.parent.name for path in config_paths}, set(EXPECTED_CONTEXT_WINDOWS))
        for config_path in config_paths:
            with self.subTest(model_id=config_path.parent.name):
                config = json.loads(config_path.read_text())
                self.assertEqual(
                    config["context_window"],
                    EXPECTED_CONTEXT_WINDOWS[config_path.parent.name],
                )


class HyMt2ConfigTest(unittest.TestCase):
    def test_translation_profile(self):
        config = json.loads((MODEL_DIR / "model.json").read_text())

        self.assertEqual(config["context"], 8192)
        self.assertEqual(config["parallel"], 1)
        self.assertEqual(config["cache_type"], "q8_0")
        self.assertEqual(config["llama"]["repo"], "tencent/Hy-MT2-7B-GGUF")
        self.assertEqual(config["llama"]["quant"], "Q4_K_M")
        self.assertEqual(
            config["sampling"],
            {
                "temperature": 0.7,
                "top_p": 0.6,
                "top_k": 20,
                "repeat_penalty": 1.05,
                "max_tokens": 4096,
            },
        )

    def test_parser_emits_sampling_defaults(self):
        result = subprocess.run(
            ["python3", str(ROOT / "scripts/parse-config.py"), str(MODEL_DIR / "model.json")],
            check=True,
            capture_output=True,
            text=True,
        )

        for assignment in (
            "LLAMA_TEMPERATURE='0.7'",
            "LLAMA_TOP_K='20'",
            "LLAMA_TOP_P='0.6'",
            "LLAMA_REPEAT_PENALTY='1.05'",
            "LLAMA_MAX_TOKENS='4096'",
        ):
            self.assertIn(assignment, result.stdout)

    def test_service_definitions_exist(self):
        expected = (
            "launchd/ai.kortexa.hy-mt2-7b.plist",
            "launchd/kortexa-hy-mt2-7b.sh",
            "systemd/kortexa-ai-llm-hy-mt2-7b.service",
        )
        for relative_path in expected:
            self.assertTrue((MODEL_DIR / relative_path).is_file(), relative_path)

    def test_llama_runner_uses_sampling_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            launcher = Path(temp_dir) / "llama-server"
            launcher.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@"\n')
            launcher.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{temp_dir}:{environment['PATH']}"

            result = subprocess.run(
                ["bash", str(ROOT / "scripts/run-llama.sh"), str(MODEL_DIR)],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )

        arguments = result.stdout.splitlines()
        self.assertIn("--kv-unified", arguments)
        for flag, value in (
            ("--temp", "0.7"),
            ("--top-k", "20"),
            ("--top-p", "0.6"),
            ("--repeat-penalty", "1.05"),
            ("--n-predict", "4096"),
        ):
            self.assertEqual(arguments[arguments.index(flag) + 1], value)


if __name__ == "__main__":
    unittest.main()
