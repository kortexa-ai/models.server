"""Shingi 27B System One model: config, dispatch, setup gate and launcher floors."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "shingi-27b"
MODEL_DIR = ROOT / MODEL_ID
UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"
RTX_4090 = "GPU-afb49bc6-cd89-6584-99cc-a0f03592a010"

# Used only when the real shingi-27b package is not importable. It mirrors the
# package interface the launcher relies on, including the direct name import in
# backend.py. Run this file with the package's Python to test the real code.
FAKE_PACKAGE = {
    "__init__.py": "",
    "gpu.py": """
        import os, re, subprocess

        def selected_gpu():
            uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid):
                raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one full GPU UUID")
            return uuid

        def gpu_snapshot():
            uuid = selected_gpu()
            row = subprocess.check_output(["nvidia-smi", "--id=" + uuid], text=True)
            fields = [s.strip() for s in row.split(",")]
            return {"uuid": fields[0], "total_mib": int(fields[2]), "free_mib": int(fields[3])}

        def gpu_profile():
            gpu = gpu_snapshot()
            if gpu["total_mib"] < 20 * 1024:
                raise RuntimeError("Shingi 27B requires a GPU with at least 20 GiB usable VRAM")
            preload, headroom = (14, 4) if gpu["total_mib"] <= 32 * 1024 else (30, 10)
            return gpu["uuid"], preload * 1024, headroom * 1024

        def gpu_free_mib():
            return gpu_snapshot()["free_mib"]
    """,
    "backend.py": """
        import subprocess
        from .gpu import gpu_free_mib, gpu_profile

        class NativeReadout:
            def __init__(self, executable, model, projector=None):
                _, preload, self.headroom = gpu_profile()
                if gpu_free_mib() < preload:
                    raise RuntimeError(f"requires at least {preload} MiB free on the GPU before loading")
                self.process = subprocess.Popen([executable, model])
    """,
    "server.py": """
        def main():
            raise AssertionError("tests replace server.main")
    """,
}


def load_launcher(test):
    """Import scripts/shingi-server.py against the real or fake shingi package."""
    for name in [n for n in sys.modules if n == "shingi" or n.startswith("shingi.")]:
        test.addCleanup(sys.modules.__setitem__, name, sys.modules.pop(name))
    if importlib.util.find_spec("shingi") is None:
        package_root = Path(tempfile.mkdtemp())
        test.addCleanup(shutil.rmtree, package_root)
        (package_root / "shingi").mkdir()
        for name, source in FAKE_PACKAGE.items():
            (package_root / "shingi" / name).write_text(textwrap.dedent(source))
        sys.path.insert(0, str(package_root))
        test.addCleanup(sys.path.remove, str(package_root))
        test.addCleanup(lambda: [sys.modules.pop(n) for n in list(sys.modules)
                                 if n == "shingi" or n.startswith("shingi.")])
    spec = importlib.util.spec_from_file_location("shingi_server", ROOT / "scripts/shingi-server.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


class ShingiConfigTest(unittest.TestCase):
    def test_system_one_profile(self):
        config = json.loads((MODEL_DIR / "model.json").read_text())
        self.assertEqual(config["type"], "systemone")
        self.assertEqual(config["default_engine"], "shingi")
        self.assertEqual((config["context"], config["context_window"], config["port"]), (16384, 16384, 2068))
        self.assertNotIn("llama", config)
        shingi = config["shingi"]
        self.assertEqual(shingi["runtime"]["directory"], ".engines/llama-prism")
        self.assertEqual(shingi["memory"], {"preload_mib": 10240, "headroom_mib": 1024})
        self.assertEqual(shingi["package"]["revision"], "6a057f2a65a6e95a7099465661c6fc5e34fdfb11")
        self.assertEqual(shingi["weights"]["revision"], "94b97064eae876330cca98a71fa7f74e6696d729")
        self.assertEqual(shingi["weights"]["projector"], "mmproj.gguf")
        # Shares the pinned Prism runtime with Bonsai 2 27B.
        bonsai = json.loads((ROOT / "bonsai-2-27b/model.json").read_text())
        self.assertEqual(shingi["runtime"], bonsai["llama"]["runtime"])
        for pinned in (shingi["package"]["revision"], shingi["weights"]["revision"]):
            self.assertRegex(pinned, r"^[0-9a-f]{40}$")

    def test_parser_emits_shingi_settings_without_llama(self):
        result = subprocess.run(
            ["python3", str(ROOT / "scripts/parse-config.py"), str(MODEL_DIR / "model.json")],
            check=True, capture_output=True, text=True,
        )
        for assignment in (
            "MODEL_TYPE='systemone'",
            "LLAMA_SUPPORTED=false",
            "SHINGI_PACKAGE_REVISION='6a057f2a65a6e95a7099465661c6fc5e34fdfb11'",
            "SHINGI_WEIGHTS_REPO='kortexa-ai/shingi-27b'",
            "SHINGI_WEIGHTS_REVISION='94b97064eae876330cca98a71fa7f74e6696d729'",
            "SHINGI_MODEL_FILE='shingi-27b.gguf'",
            "SHINGI_CALIBRATION_FILE='calibration.json'",
            "SHINGI_PROJECTOR_FILE='mmproj.gguf'",
            "SHINGI_RUNTIME_DIR='.engines/llama-prism'",
            "SHINGI_PRELOAD_MIB='10240'",
            "SHINGI_HEADROOM_MIB='1024'",
        ):
            self.assertIn(assignment, result.stdout)

    def test_parser_emits_empty_projector_when_absent(self):
        config = json.loads((MODEL_DIR / "model.json").read_text())
        del config["shingi"]["weights"]["projector"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(config))
            result = subprocess.run(["python3", str(ROOT / "scripts/parse-config.py"), str(path)],
                                    check=True, capture_output=True, text=True)
        self.assertIn("SHINGI_PROJECTOR_FILE=''", result.stdout)

    def test_other_models_report_shingi_unsupported(self):
        result = subprocess.run(
            ["python3", str(ROOT / "scripts/parse-config.py"), str(ROOT / "bonsai-2-27b/model.json")],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("SHINGI_SUPPORTED=false", result.stdout)
        self.assertIn("MODEL_TYPE=''", result.stdout)

    def test_service_is_pinned_to_the_rtx_4090(self):
        service = (MODEL_DIR / f"systemd/kortexa-ai-llm-{MODEL_ID}.service").read_text()
        reference = (ROOT / "lfm2.5-8b-a1b/systemd/kortexa-ai-llm-lfm2.5-8b-a1b.service").read_text()
        self.assertIn("Description=Shingi 27B System One decision model", service)
        self.assertIn(f"Environment=CUDA_VISIBLE_DEVICES={RTX_4090}", service)
        self.assertIn(f"ExecStartPre=/usr/bin/nvidia-smi --id={RTX_4090} --query-gpu=uuid --format=csv,noheader", service)
        self.assertIn(f"ExecStart=/home/francip/src/models.server/run.sh /home/francip/src/models.server/{MODEL_ID}", service)
        for line in reference.splitlines():
            if line.startswith(("ProtectHome=", "ProtectSystem=", "ReadWritePaths=", "Environment=PATH=")):
                self.assertIn(line, service)


class ShingiScriptsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / "scripts").mkdir()
        for name in ("run.sh", "scripts/parse-config.py", "scripts/run-shingi.sh"):
            shutil.copy2(ROOT / name, self.root / name)
        (self.root / MODEL_ID).mkdir()
        shutil.copy2(MODEL_DIR / "model.json", self.root / MODEL_ID / "model.json")
        self.config = json.loads((MODEL_DIR / "model.json").read_text())["shingi"]
        self.engine = self.root / ".engines/shingi"

    def run_model(self):
        environment = {k: v for k, v in os.environ.items() if k not in ("PORT", "HOST")}
        return subprocess.run(
            ["bash", str(self.root / "run.sh"), MODEL_ID, "--extra"],
            cwd=self.root, capture_output=True, text=True, env=environment,
        )

    def install_fake_setup(self, package=None, runtime=None):
        (self.engine / "venv/bin").mkdir(parents=True)
        (self.engine / "bin").mkdir()
        python = self.engine / "venv/bin/python"
        python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
        python.chmod(0o755)
        readout = self.engine / "bin/readout"
        readout.write_text("#!/bin/sh\n")
        readout.chmod(0o755)
        (self.engine / "package.stamp").write_text((package or self.config["package"]["revision"]) + "\n")
        runtime = runtime or self.config["runtime"]["revision"]
        (self.engine / "bin/readout.stamp").write_text(
            f"{self.config['package']['revision']} {runtime} /somewhere/prism\n")

    def test_missing_setup_fails_without_writing(self):
        result = self.run_model()
        self.assertEqual(result.returncode, 1)
        self.assertIn("scripts/setup-shingi.sh", result.stderr)
        self.assertFalse((self.root / ".engines").exists())

    def test_stale_setup_fails(self):
        for stale in ({"package": "0" * 40}, {"runtime": "0" * 40}):
            with self.subTest(stale=stale):
                shutil.rmtree(self.root / ".engines", ignore_errors=True)
                self.install_fake_setup(**stale)
                result = self.run_model()
                self.assertEqual(result.returncode, 1)
                self.assertIn("scripts/setup-shingi.sh", result.stderr)

    def test_default_engine_starts_the_launcher_with_pinned_settings(self):
        self.install_fake_setup()
        result = self.run_model()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = result.stdout.splitlines()[1:]
        self.assertEqual(args[0], str(self.root / "scripts/shingi-server.py"))
        for flag, value in (
            ("--preload-mib", "10240"),
            ("--headroom-mib", "1024"),
            ("--weights-repo", "kortexa-ai/shingi-27b"),
            ("--weights-revision", self.config["weights"]["revision"]),
            ("--model-file", "shingi-27b.gguf"),
            ("--calibration-file", "calibration.json"),
            ("--projector-file", "mmproj.gguf"),
            ("--executable", str(self.engine / "bin/readout")),
            ("--host", "0.0.0.0"),
            ("--port", "2068"),
        ):
            self.assertEqual(args[args.index(flag) + 1], value, flag)
        self.assertEqual(args[-1], "--extra")

    def test_text_only_config_passes_no_projector(self):
        path = self.root / MODEL_ID / "model.json"
        config = json.loads(path.read_text())
        del config["shingi"]["weights"]["projector"]
        path.write_text(json.dumps(config))
        self.install_fake_setup()
        result = self.run_model()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--projector-file", result.stdout.splitlines())


class ShingiSetupTest(unittest.TestCase):
    """Run setup-shingi.sh with fake tools against a reusable shared Prism runtime."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / "scripts").mkdir()
        for name in ("scripts/parse-config.py", "scripts/setup-common.sh", "scripts/setup-shingi.sh"):
            shutil.copy2(ROOT / name, self.root / name)
        (self.root / MODEL_ID).mkdir()
        shutil.copy2(MODEL_DIR / "model.json", self.root / MODEL_ID / "model.json")
        config = json.loads((MODEL_DIR / "model.json").read_text())["shingi"]
        self.prism = self.root / config["runtime"]["directory"]
        for name in ("include/llama.h", "ggml/include/ggml-backend.h", "vendor/nlohmann/json.hpp",
                     "tools/mtmd/mtmd.h", "build/bin/libllama.so", "build/bin/libggml.so",
                     "build/bin/libggml-base.so", "build/bin/libmtmd.so"):
            (self.prism / name).parent.mkdir(parents=True, exist_ok=True)
            (self.prism / name).touch()
        engine = self.root / ".engines/shingi"
        (engine / "src/.git").mkdir(parents=True)
        (engine / "venv/bin").mkdir(parents=True)
        self.log = self.root / "calls.log"
        self.fake(engine / "venv/bin/python", 'echo "download $4" >> "$LOG"')
        (engine / "package.stamp").write_text(config["package"]["revision"] + "\n")
        self.bin = self.root / "fakebin"
        self.bin.mkdir()
        self.fake(self.bin / "uname", "echo Linux")
        self.fake(self.bin / "uv", "exit 0")
        self.fake(self.bin / "c++", textwrap.dedent('''
            echo "c++ $*" >> "$LOG"
            while [ $# -gt 1 ]; do [ "$1" = -o ] && touch "$2"; shift; done'''))
        self.fake(self.bin / "git", textwrap.dedent(f'''
            case "$*" in
                *" remote get-url origin") echo {config["package"]["repo"]} ;;
                *"/src rev-parse HEAD") echo {config["package"]["revision"]} ;;
                *" rev-parse HEAD") echo {config["runtime"]["revision"]} ;;
                *" status "*) ;;
                *) echo "unexpected git $*" >&2; exit 1 ;;
            esac'''))

    def fake(self, path, body):
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)

    def test_readout_links_mtmd_and_projector_is_downloaded(self):
        environment = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", LOG=str(self.log))
        result = subprocess.run(["bash", str(self.root / "scripts/setup-shingi.sh"), str(self.root / MODEL_ID)],
                                capture_output=True, text=True, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reusing the shared Prism runtime", result.stdout)
        calls = self.log.read_text().splitlines()
        compile_line = next(line for line in calls if line.startswith("c++ "))
        self.assertIn(f"-I{self.prism}/tools/mtmd", compile_line)
        self.assertIn("-lmtmd -lllama -lggml -lggml-base", compile_line)
        stamp = (self.root / ".engines/shingi/bin/readout.stamp").read_text()
        self.assertIn(" mtmd ", stamp)
        downloads = [line.split()[1] for line in calls if line.startswith("download ")]
        self.assertEqual(downloads, ["shingi-27b.gguf", "calibration.json", "mmproj.gguf"])

    def test_shared_runtime_without_mtmd_is_not_reused(self):
        (self.prism / "build/bin/libmtmd.so").unlink()
        environment = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", LOG=str(self.log))
        result = subprocess.run(["bash", str(self.root / "scripts/setup-shingi.sh"), str(self.root / MODEL_ID)],
                                capture_output=True, text=True, env=environment)
        self.assertNotIn("reusing the shared Prism runtime", result.stdout)
        self.assertIn("missing, stale, or static", result.stdout)


class ShingiLauncherTest(unittest.TestCase):
    def setUp(self):
        self.launcher = load_launcher(self)
        from shingi import backend, gpu
        self.gpu, self.backend = gpu, backend
        if hasattr(gpu, "is_macos"):
            # The real package takes a Metal path on macOS; these tests model the CUDA host.
            macos = mock.patch.object(gpu, "is_macos", return_value=False)
            macos.start()
            self.addCleanup(macos.stop)
        environment = mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": UUID})
        environment.start()
        self.addCleanup(environment.stop)

    def fake_nvidia_smi(self, total, free):
        patcher = mock.patch.object(self.gpu.subprocess, "check_output",
                                    return_value=f"{UUID}, NVIDIA GeForce RTX 4090, {total}, {free}\n")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_override_replaces_only_the_floors(self):
        self.fake_nvidia_smi(24564, 11500)
        self.assertEqual(self.gpu.gpu_profile(), (UUID, 14 * 1024, 4 * 1024))
        self.launcher.override_floors(10240, 2048)
        self.assertEqual(self.gpu.gpu_profile(), (UUID, 10240, 2048))
        self.assertEqual(self.backend.gpu_profile(), (UUID, 10240, 2048))

    def test_preload_gate_uses_the_override(self):
        self.fake_nvidia_smi(24564, 11500)
        started = []
        with mock.patch.object(self.backend.subprocess, "Popen", side_effect=lambda *a, **k: started.append(a) or 1 / 0):
            with self.assertRaisesRegex(RuntimeError, "14336 MiB free"):
                self.backend.NativeReadout("readout", "model.gguf")
            self.assertEqual(started, [])
            self.launcher.override_floors(10240, 2048)
            with self.assertRaises(ZeroDivisionError):
                self.backend.NativeReadout("readout", "model.gguf")
            self.assertEqual(len(started), 1)

    def test_package_minimum_size_and_uuid_checks_still_apply(self):
        self.launcher.override_floors(10240, 2048)
        self.fake_nvidia_smi(16384, 15000)
        with self.assertRaisesRegex(RuntimeError, "at least 20 GiB"):
            self.backend.gpu_profile()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(RuntimeError, "exactly one full GPU UUID"):
                self.backend.gpu_profile()

    def test_main_passes_resolved_weights_to_the_server(self):
        calls, seen = [], {}
        from shingi import server

        def fake_resolve(repo, revision, filename):
            calls.append((repo, revision, filename))
            return Path("/cache") / filename

        def fake_main():
            seen["argv"] = list(sys.argv)
        with mock.patch.object(self.launcher, "resolve", fake_resolve), \
                mock.patch.object(server, "main", fake_main), \
                mock.patch.object(sys, "argv", ["shingi-server.py"]):
            self.launcher.main([
                "--preload-mib", "10240", "--headroom-mib", "2048",
                "--weights-repo", "kortexa-ai/shingi-27b", "--weights-revision", "abc",
                "--model-file", "shingi-27b.gguf", "--calibration-file", "calibration.json",
                "--executable", "/engines/readout", "--host", "0.0.0.0", "--port", "2068",
            ])
        self.assertEqual(calls, [("kortexa-ai/shingi-27b", "abc", "shingi-27b.gguf"),
                                 ("kortexa-ai/shingi-27b", "abc", "calibration.json")])
        self.assertEqual(seen["argv"], [
            "shingi-27b", "--model", "/cache/shingi-27b.gguf", "--calibration", "/cache/calibration.json",
            "--executable", "/engines/readout", "--host", "0.0.0.0", "--port", "2068",
        ])

    def test_main_passes_the_resolved_projector_as_mmproj(self):
        calls, seen = [], {}
        from shingi import server

        def fake_resolve(repo, revision, filename):
            calls.append((repo, revision, filename))
            return Path("/cache") / filename

        def fake_main():
            seen["argv"] = list(sys.argv)
        with mock.patch.object(self.launcher, "resolve", fake_resolve), \
                mock.patch.object(server, "main", fake_main), \
                mock.patch.object(sys, "argv", ["shingi-server.py"]):
            self.launcher.main([
                "--preload-mib", "10240", "--headroom-mib", "1024",
                "--weights-repo", "kortexa-ai/shingi-27b", "--weights-revision", "abc",
                "--model-file", "shingi-27b.gguf", "--calibration-file", "calibration.json",
                "--projector-file", "mmproj.gguf",
                "--executable", "/engines/readout", "--port", "2068",
            ])
        self.assertEqual(calls[-1], ("kortexa-ai/shingi-27b", "abc", "mmproj.gguf"))
        self.assertEqual(seen["argv"], [
            "shingi-27b", "--model", "/cache/shingi-27b.gguf", "--calibration", "/cache/calibration.json",
            "--mmproj", "/cache/mmproj.gguf", "--executable", "/engines/readout", "--port", "2068",
        ])

    @unittest.skipIf(importlib.util.find_spec("huggingface_hub") is None, "huggingface_hub not installed")
    def test_resolve_prefers_the_local_cache(self):
        import huggingface_hub
        from huggingface_hub.utils import LocalEntryNotFoundError

        calls = []

        def fake_download(repo, filename, revision, local_files_only=False):
            calls.append(local_files_only)
            if local_files_only and filename == "missing":
                raise LocalEntryNotFoundError("not cached")
            return f"/cache/{filename}"
        with mock.patch.object(huggingface_hub, "hf_hub_download", fake_download):
            self.assertEqual(self.launcher.resolve("r", "sha", "cached"), "/cache/cached")
            self.assertEqual(calls, [True])
            self.assertEqual(self.launcher.resolve("r", "sha", "missing"), "/cache/missing")
            self.assertEqual(calls, [True, True, False])


if __name__ == "__main__":
    unittest.main()
