import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "hy-mt2-7b"


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
