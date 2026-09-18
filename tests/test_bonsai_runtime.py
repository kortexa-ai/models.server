import json
import os
import plistlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "bonsai-2-27b"


class BonsaiRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bonsai runtime ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in (
            "parse-config.py",
            "run-llama.sh",
            "run-mlx.sh",
            "setup-common.sh",
            "setup-llama-runtime.sh",
        ):
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        shutil.copy2(ROOT / "run.sh", self.root / "run.sh")
        shutil.copytree(ROOT / MODEL_ID, self.root / MODEL_ID)
        self.config = json.loads((self.root / MODEL_ID / "model.json").read_text())
        self.engine = self.root / self.config["llama"]["runtime"]["directory"]
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = {
            key: value for key, value in os.environ.items()
            if key not in {
                "PORT", "HOST", "QUANT", "CACHE_TYPE", "CONTEXT", "PARALLEL",
                "TEMPERATURE", "TOP_K", "TOP_P", "REPEAT_PENALTY", "MAX_TOKENS",
            }
        }
        self.env.update(PATH=f"{self.bin}:{self.env['PATH']}", CDPATH="")
        self.write_executable(
            self.bin / "llama-server", "#!/bin/sh\nprintf 'STOCK\\n'; printf '%s\\n' \"$@\"\n"
        )

    def write_executable(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o755)

    def install_fake_server(self):
        self.write_executable(
            self.engine / "build/bin/llama-server",
            "#!/bin/sh\nprintf 'PRISM\\n'; printf '%s\\n' \"$@\"\n",
        )

    def run_script(self, script, *args):
        return subprocess.run(
            ["bash", str(self.root / script), *map(str, args)],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
        )

    def fake_platform(self, platform):
        self.write_executable(self.bin / "uname", f"#!/bin/sh\necho {platform}\n")

    def test_profile_and_port(self):
        self.assertEqual(self.config["default_engine"], "llama")
        self.assertEqual(self.config["context_window"], 262144)
        self.assertEqual(self.config["context"], 262144)
        self.assertEqual(self.config["parallel"], 1)
        self.assertEqual(self.config["cache_type"], "q8_0")
        self.assertTrue(self.config["multimodal"])
        self.assertEqual(self.config["llama"]["quant"], "PQ2_0")
        self.assertEqual(self.config["llama"]["image_min_tokens"], 1024)
        self.assertEqual(self.config["mlx"]["backend"], "prism_hadamard")
        self.assertEqual(self.config["mlx"]["repo"], "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit")
        for path in ROOT.glob("*/model.json"):
            if path.parent.name != MODEL_ID:
                self.assertNotEqual(json.loads(path.read_text())["port"], self.config["port"])

    def test_macos_and_linux_default_to_isolated_server(self):
        self.install_fake_server()
        for platform in ("Darwin", "Linux"):
            with self.subTest(platform=platform):
                self.fake_platform(platform)
                result = self.run_script("run.sh", MODEL_ID)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = result.stdout.splitlines()
                self.assertIn("PRISM", args)
                self.assertNotIn("STOCK", args)
                for flag, value in (
                    ("-hf", "prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0"),
                    ("--alias", MODEL_ID),
                    ("--port", "2062"),
                    ("-c", "262144"),
                    ("--parallel", "1"),
                    ("--cache-type-k", "q8_0"),
                    ("--cache-type-v", "q8_0"),
                    ("--cors-origins", "localhost"),
                    ("--mmproj-url", self.config["llama"]["mmproj_url"]),
                    ("--image-min-tokens", "1024"),
                    ("--temp", "1.0"),
                    ("--top-p", "0.95"),
                    ("--top-k", "20"),
                ):
                    self.assertEqual(args[args.index(flag) + 1], value)
                self.assertIn("--kv-unified", args)
                self.assertNotIn("--no-mmproj", args)
                self.assertNotIn("--spec-type", args)

    def test_quant_override_and_extra_arguments(self):
        self.install_fake_server()
        self.env["QUANT"] = "PTQ1_0"
        result = self.run_script("run.sh", MODEL_ID, "--engine", "llama", "--host", "127.0.0.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prism-ml/Ternary-Bonsai-2-27B-gguf:PTQ1_0", result.stdout)
        self.assertEqual(result.stdout.splitlines()[-2:], ["--host", "127.0.0.1"])

    def test_missing_runtime_never_falls_back_to_stock(self):
        result = self.run_script("run.sh", MODEL_ID)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scripts/setup-llama-runtime.sh", result.stderr)
        self.assertNotIn("STOCK", result.stdout)

    def test_mlx_override_refuses_incompatible_stock_loader(self):
        result = self.run_script("run.sh", MODEL_ID, "--engine", "mlx")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Hadamard-aware MLX loader", result.stderr)
        self.assertIn("--engine llama", result.stderr)

    def test_existing_models_keep_the_stock_runtime(self):
        model_id = "qwen-3.8-27b"
        shutil.copytree(ROOT / model_id, self.root / model_id)
        result = self.run_script("run.sh", model_id, "--engine", "llama")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()
        self.assertIn("STOCK", args)
        self.assertNotIn("PRISM", args)
        self.assertNotIn("--mmproj-url", args)
        self.assertNotIn("--image-min-tokens", args)

    def test_existing_mlx_timeout_environment_is_preserved(self):
        model_id = "qwen-3.8-27b"
        shutil.copytree(ROOT / model_id, self.root / model_id)
        self.write_executable(
            self.bin / "python", '#!/bin/sh\nprintf "%s\\n" "$MLX_VLM_TOKEN_QUEUE_TIMEOUT"\n'
        )
        for override, expected in ((None, "1800"), ("3600", "3600")):
            with self.subTest(override=override):
                self.env.pop("MLX_VLM_TOKEN_QUEUE_TIMEOUT", None)
                if override is not None:
                    self.env["MLX_VLM_TOKEN_QUEUE_TIMEOUT"] = override
                result = self.run_script("run.sh", model_id, "--engine", "mlx")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines()[-1], expected)

    def test_service_definitions_use_default_gguf_engine(self):
        model_dir = ROOT / MODEL_ID
        with (model_dir / f"launchd/ai.kortexa.{MODEL_ID}.plist").open("rb") as stream:
            plist = plistlib.load(stream)
        self.assertEqual(plist["Label"], f"ai.kortexa.{MODEL_ID}")
        self.assertEqual(plist["ProgramArguments"][-1], f"/Users/francip/src/models.server/{MODEL_ID}")
        self.assertFalse(plist["RunAtLoad"])
        self.assertNotIn("--engine", plist["ProgramArguments"])
        service = (model_dir / f"systemd/kortexa-ai-llm-{MODEL_ID}.service").read_text()
        self.assertIn(f"ExecStart=/home/francip/src/models.server/run.sh /home/francip/src/models.server/{MODEL_ID}", service)

    def fake_build_tools(self):
        self.env["BUILD_LOG"] = str(self.root / "build-log.jsonl")
        self.env["EXPECTED_REPO"] = self.config["llama"]["runtime"]["repo"]
        self.write_executable(self.bin / "git", """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["BUILD_LOG"], "a") as log:
    log.write(json.dumps(["git", *args]) + "\\n")
if args[0] == "clone":
    (Path(args[-1]) / ".git").mkdir(parents=True)
elif args[2:5] == ["remote", "get-url", "origin"]:
    print(os.environ["EXPECTED_REPO"])
elif args[2] == "status":
    print(os.environ.get("DIRTY_STATUS", ""), end="")
""")
        self.write_executable(self.bin / "cmake", """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["BUILD_LOG"], "a") as log:
    log.write(json.dumps(["cmake", *args]) + "\\n")
if args[0] == "--build":
    binary = Path(args[1]) / "bin/llama-server"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\\necho PRISM\\n")
    binary.chmod(0o755)
""")

    def test_setup_builds_pinned_metal_runtime_and_can_repeat(self):
        self.fake_platform("Darwin")
        self.fake_build_tools()
        for _ in range(2):
            result = self.run_script("scripts/setup-llama-runtime.sh", MODEL_ID)
            self.assertEqual(result.returncode, 0, result.stderr)
        commands = [
            json.loads(line) for line in Path(self.env["BUILD_LOG"]).read_text().splitlines()
        ]
        self.assertEqual(sum(command[1] == "clone" for command in commands), 1)
        revision = self.config["llama"]["runtime"]["revision"]
        self.assertIn(["git", "-C", str(self.engine), "checkout", "--detach", revision], commands)
        configurations = [command for command in commands if command[:2] == ["cmake", "-S"]]
        self.assertEqual(len(configurations), 2)
        self.assertIn("-DGGML_METAL=ON", configurations[0])
        self.assertIn("-DLLAMA_OPENSSL=ON", configurations[0])
        self.assertNotIn("-DGGML_CUDA=ON", configurations[0])

    def test_setup_builds_both_cuda_architectures(self):
        self.fake_platform("Linux")
        self.fake_build_tools()
        self.write_executable(self.bin / "nvcc", "#!/bin/sh\nexit 0\n")
        result = self.run_script("scripts/setup-llama-runtime.sh", MODEL_ID)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [
            json.loads(line) for line in Path(self.env["BUILD_LOG"]).read_text().splitlines()
        ]
        configure = next(command for command in commands if command[:2] == ["cmake", "-S"])
        self.assertIn("-DGGML_CUDA=ON", configure)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=89;120", configure)

    def test_setup_preserves_local_source_changes(self):
        self.fake_platform("Darwin")
        self.fake_build_tools()
        (self.engine / ".git").mkdir(parents=True)
        self.env["DIRTY_STATUS"] = " M common/arg.cpp\n"
        result = self.run_script("scripts/setup-llama-runtime.sh", MODEL_ID)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("has local changes", result.stderr)
        log = Path(self.env["BUILD_LOG"]).read_text()
        self.assertNotIn('"checkout"', log)
        self.assertNotIn('"cmake"', log)

    def test_setup_rejects_an_unexpected_source_repository(self):
        self.fake_build_tools()
        (self.engine / ".git").mkdir(parents=True)
        self.env["EXPECTED_REPO"] = "https://example.com/other.git"
        result = self.run_script("scripts/setup-llama-runtime.sh", MODEL_ID)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexpected origin", result.stderr)
        log = Path(self.env["BUILD_LOG"]).read_text()
        self.assertNotIn('"checkout"', log)
        self.assertNotIn('"cmake"', log)


if __name__ == "__main__":
    unittest.main()
