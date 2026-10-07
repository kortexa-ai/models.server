import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tensorfold_exl3_runtime", ROOT / "scripts/tensorfold_exl3_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class TensorFoldExl3RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "glm-5.3-flash-exl3/model.json").read_text())
        self.options = argparse.Namespace(host=None, port=None)

    def test_stop_refuses_a_container_owned_by_another_model(self):
        result = runtime.subprocess.CompletedProcess([], 0, json.dumps([
            {"Config": {"Labels": {"kortexa.model": "another-model"}}}
        ]))
        with patch.object(runtime, "check_platform"), \
                patch.object(runtime.subprocess, "run", return_value=result), \
                patch.object(runtime, "run") as stop:
            with self.assertRaisesRegex(ValueError, "another workload"):
                runtime.stop(self.config)
            stop.assert_not_called()

    def test_verified_weights_reject_changes_and_wrong_pins(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runtime, "engine_root", return_value=Path(directory)):
            root = Path(directory)
            (root / "checkpoint").mkdir()
            model = root / "checkpoint/model.safetensors"
            model.write_bytes(b"verified")
            receipt = {"identity": runtime.identity(self.config), "image_id": "sha256:tested-image"}
            runtime.write_json(root / "runtime.json", receipt)
            runtime.write_json(root / "checkpoint.json", {
                "revision": self.config["tensorfold_exl3"]["checkpoint_revision"],
                "files": [{"path": model.name, "bytes": model.stat().st_size, "mtime_ns": model.stat().st_mtime_ns}]
            })
            with patch.object(runtime, "output", return_value=receipt["image_id"]):
                self.assertEqual(runtime.verify_runtime(self.config), receipt)
                model.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "changed since checksum"):
                    runtime.verify_runtime(self.config)
                receipt["identity"]["checkpoint_revision"] = "unverified-revision"
                runtime.write_json(root / "runtime.json", receipt)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    runtime.verify_runtime(self.config)

    def test_busy_gpu_and_low_memory_do_not_start_a_container(self):
        with patch.object(runtime, "check_platform"), \
                patch.object(runtime, "verify_runtime", return_value={"image_id": "sha256:tested-image"}), \
                patch.object(runtime, "container_state", return_value=None), \
                patch.object(runtime.socket, "socket") as probe, \
                patch.object(runtime, "output") as query, \
                patch.object(runtime.Path, "read_text", return_value="MemAvailable: 1024 kB\n"), \
                patch.object(runtime.os, "execvp") as execute:
            probe.return_value.__enter__.return_value.connect_ex.return_value = 1
            query.return_value = "1234"
            with self.assertRaisesRegex(ValueError, "another compute process"):
                runtime.serve(self.config, self.options)
            query.return_value = ""
            with self.assertRaisesRegex(ValueError, "available at startup"):
                runtime.serve(self.config, self.options)
            execute.assert_not_called()
            self.assertIn("--id=" + self.config["tensorfold_exl3"]["gpu_uuid"], query.call_args.args[0])

    def test_launch_uses_verified_image_and_explicit_resource_limits(self):
        with patch.object(runtime, "check_platform"), \
                patch.object(runtime, "verify_runtime", return_value={"image_id": "sha256:tested-image"}), \
                patch.object(runtime, "container_state", return_value=None), \
                patch.object(runtime.socket, "socket") as probe, \
                patch.object(runtime, "output", return_value=""), \
                patch.object(runtime.Path, "read_text", return_value="MemAvailable: 130000000 kB\n"), \
                patch.object(runtime.os, "execvp") as execute:
            probe.return_value.__enter__.return_value.connect_ex.return_value = 1
            runtime.serve(self.config, self.options)
            args = execute.call_args.args[1]
            e = self.config["tensorfold_exl3"]
            self.assertIn("sha256:tested-image", args)
            self.assertEqual(args[args.index("--gpus") + 1], "device=" + e["gpu_uuid"])
            self.assertEqual(args[args.index("--memory") + 1], args[args.index("--memory-swap") + 1])
            self.assertIn("TENSORFOLD_CUDA_MEMORY_LIMIT_GB=" + str(e["container_memory_gib"]), args)
            self.assertIn("kortexa.model=glm-5.3-flash-exl3", args)

    def test_api_identity_and_diagnostic_overrides(self):
        self.options.host, self.options.port = "127.0.0.1", 18100
        args = runtime.serve_args(self.config, self.options)
        for flag, value in {"--name": "glm-5.3-flash-exl3", "--host": "127.0.0.1", "--port": "18100",
                            "--tp": "1", "--parallel": "1", "--kv-dtype": "bf16", "--drafter": "none"}.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertFalse(self.config["multimodal"])
        unit = (ROOT / "glm-5.3-flash-exl3/systemd/kortexa-ai-llm-glm-5.3-flash-exl3.service").read_text()
        self.assertIn("scripts/run-tensorfold-exl3.sh", unit)
        self.assertIn("--stop", unit)


if __name__ == "__main__":
    unittest.main()
