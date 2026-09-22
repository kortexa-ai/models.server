import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cinference_runtime", ROOT / "scripts/cinference_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class CinferenceRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "qwen-3.8-27b-fast/model.json").read_text())
        self.options = argparse.Namespace(host=None, port=None, request_log_jsonl=None)

    def test_launch_profile_and_optional_diagnostic_address(self):
        args = runtime.serve_args(self.config, Path("/model.ninfer"), self.options)
        for flag, value in {
            "--model-id": "qwen-3.8-27b-fast", "--port": "2064",
            "--max-context": "262144", "--kv-capacity": "524288",
            "--max-concurrency": "8", "--kv-dtype": "k8v4",
            "--spec": "dflash2", "--draft-tokens": "7", "--prefill-chunk": "8192",
            "--pending-timeout-ms": "600000", "--default-max-tokens": "32768",
        }.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        for flag in ("--vision", "--lm-head-draft", "--preserve-thinking"):
            self.assertIn(flag, args)
        self.options.host, self.options.port = "127.0.0.1", 18080
        args = runtime.serve_args(self.config, Path("/model.ninfer"), self.options)
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        self.assertEqual(args[args.index("--port") + 1], "18080")

    def test_gpu_override_rejected_before_model_load(self):
        with patch.object(runtime, "check_platform"), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            with self.assertRaisesRegex(ValueError, "pinned"):
                runtime.serve(self.config, self.options)

    def test_memory_gate_and_exec_pin(self):
        receipt = {"runtime_revision": runtime.RUNTIME_REVISION,
                   "recipe_revision": runtime.RECIPE_REVISION, "binary_sha256": "binary"}
        with patch.object(runtime, "check_platform"), patch.object(runtime, "check_checkout"), \
                patch.object(runtime.socket, "socket"), \
                patch.object(runtime, "read_json", return_value=receipt), \
                patch.object(runtime, "sha256", return_value="binary"), \
                patch.object(runtime, "verify_artifact", return_value=Path("/model.ninfer")), \
                patch.object(runtime, "output") as query, patch.object(runtime.os, "execve") as execute, \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": runtime.GPU_UUID}):
            query.return_value = runtime.GPU_UUID + ", 51199"
            with self.assertRaisesRegex(ValueError, "50 GiB"):
                runtime.serve(self.config, self.options)
            execute.assert_not_called()
            query.return_value = runtime.GPU_UUID + ", 80000"
            runtime.serve(self.config, self.options)
            self.assertEqual(execute.call_args.args[2]["CUDA_VISIBLE_DEVICES"], runtime.GPU_UUID)
            self.assertIn(f"--id={runtime.GPU_UUID}", query.call_args.args[0])

    def test_verified_artifact_rejects_corruption_and_wrong_provenance(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runtime, "ENGINE", Path(directory)):
            artifact, source, served, receipt_path = runtime.artifact_paths(self.config)
            source.parent.mkdir(parents=True)
            source.write_bytes(b"test model")
            artifact["sha256"] = runtime.sha256(source)
            artifact["size_bytes"] = source.stat().st_size
            receipt = {"source": runtime.source_identity(artifact),
                       "runtime_revision": runtime.RUNTIME_REVISION,
                       "served_sha256": artifact["sha256"], "served_size_bytes": artifact["size_bytes"]}
            runtime.write_json(receipt_path, receipt)
            self.assertEqual(runtime.verify_artifact(self.config), served)
            source.write_bytes(b"bad! model")
            with self.assertRaisesRegex(ValueError, "checksum"):
                runtime.verify_artifact(self.config)
            receipt["source"]["revision"] = "another revision"
            runtime.write_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "receipt"):
                runtime.verify_artifact(self.config)

    def test_model_aliases_ports_and_service_gpu_match(self):
        ports = {}
        for path in ROOT.glob("*/model.json"):
            config = json.loads(path.read_text())
            if "port" in config:
                self.assertNotIn(config["port"], ports, (path, ports.get(config["port"])))
                ports[config["port"]] = path
        for suffix, port in (("", 2064), ("-abliterated", 2065)):
            model_id = "qwen-3.8-27b-fast" + suffix
            config = json.loads((ROOT / model_id / "model.json").read_text())
            self.assertEqual(config["port"], port)
            self.assertEqual(config["host"], "192.168.2.3")
            self.assertEqual(config["default_engine"], "cinference")
            unit = (ROOT / model_id / "systemd" / f"kortexa-ai-llm-{model_id}.service").read_text()
            self.assertIn("Environment=CUDA_VISIBLE_DEVICES=" + runtime.GPU_UUID, unit)
            self.assertIn("/models.server/" + model_id, unit)

    def test_run_dispatch_uses_cinference_without_platform_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "model").mkdir()
            (root / "run.sh").write_text((ROOT / "run.sh").read_text())
            (root / "model/model.json").write_text(json.dumps(self.config))
            launcher = root / "scripts/run-cinference.sh"
            launcher.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            launcher.chmod(0o755)
            result = subprocess.run(["bash", root / "run.sh", "model", "--port", "18080"],
                                    text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout.splitlines(), [str(root / "model"), "--port", "18080"])


if __name__ == "__main__":
    unittest.main()
