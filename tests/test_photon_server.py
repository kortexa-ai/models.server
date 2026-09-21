import importlib.util
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("photon_server", ROOT / "scripts/photon-server.py")
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


class FakeSpeech:
    def __init__(self, *args, **kwargs):
        self.options = kwargs
        self.paths = []
        self.active = 0
        self.peak = 0
        self.closed = False
        self.lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def transcribe(self, audio, timestamps):
        assert timestamps in ("none", "segment", "word")
        self.paths.append(audio)
        data = Path(audio).read_bytes()
        if data == b"bad":
            raise ValueError("bad fixture")
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        result = {"text": "Hello world."}
        if timestamps != "none":
            result["segments"] = [{"text": "Hello world.", "start": 0.0, "end": 1.0}]
            if timestamps == "word":
                result["segments"][0]["words"] = [{"word": "Hello", "start": 0.0, "end": 0.4}]
        return result


class PhotonApiTest(unittest.TestCase):
    def setUp(self):
        self.speech = FakeSpeech()
        self.app = SERVER.create_app("moondream/parakeet-redux", "parakeet-redux",
                                     model_factory=lambda *a, **k: self.speech)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.assertTrue(self.speech.closed)
        self.assertTrue(all(not Path(path).exists() for path in self.speech.paths))

    def upload(self, data=None, audio=b"audio"):
        return self.client.post("/v1/audio/transcriptions", data=data or {},
                                files={"file": ("../../speech.wav", audio, "audio/wav")})

    def test_metadata_and_formats(self):
        self.assertEqual(self.client.get("/health").json()["device"], "cpu")
        model = self.client.get("/v1/models").json()["data"][0]
        self.assertEqual(model["type"], "transcription")
        self.assertFalse(model["capabilities"]["streaming"])
        self.assertEqual(self.upload().json(), {"text": "Hello world."})
        response = self.upload({"response_format": "text"})
        self.assertEqual(response.text, "Hello world.")
        self.assertIn("text/plain", response.headers["content-type"])
        response = self.upload({"model": "moondream/parakeet-redux",
                                "response_format": "verbose_json",
                                "timestamp_granularities[]": "word"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["words"][0]["word"], "Hello")
        self.assertEqual(response.json()["segments"][0]["end"], 1.0)

    def test_rejects_invalid_requests_without_inference(self):
        for data in ({"model": "qwen"}, {"response_format": "srt"},
                     {"timestamp_granularities[]": "word"},
                     {"response_format": "verbose_json", "timestamp_granularities[]": "token"},
                     {"language": "en"}):
            with self.subTest(data=data):
                self.assertEqual(self.upload(data).status_code, 400)
        self.assertEqual(self.client.post("/v1/audio/transcriptions", json={}).status_code, 400)
        self.assertEqual(self.speech.paths, [])

    def test_empty_oversized_and_bad_audio_cleanup(self):
        self.assertEqual(self.upload(audio=b"").status_code, 400)
        with patch.object(SERVER, "MAX_UPLOAD_BYTES", 4):
            self.assertEqual(self.upload(audio=b"12345").status_code, 413)
        self.assertEqual(self.upload(audio=b"bad").status_code, 400)

    def test_serializes_inference(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(lambda _: self.upload(), range(3)))
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(self.speech.peak, 1)


class PhotonLauncherTest(unittest.TestCase):
    def test_dispatches_cpu_configuration_and_port_override(self):
        config = json.loads((ROOT / "parakeet-redux/model.json").read_text())
        self.assertEqual(config["type"], "transcription")
        self.assertNotIn("context_window", config)
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "model.json").write_text(json.dumps(config))
            binary = model_dir / ".venv/bin/python"
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/bash\nprintf "CUDA=%s\\n" "$CUDA_VISIBLE_DEVICES"\nprintf "%s\\n" "$@"\n')
            binary.chmod(0o755)
            env = {**os.environ, "PORT": "21999", "CUDA_VISIBLE_DEVICES": "0"}
            result = subprocess.run([str(ROOT / "run.sh"), str(model_dir)],
                                    check=True, capture_output=True, text=True, env=env)
        args = result.stdout.splitlines()
        self.assertIn("CUDA=", args)
        for flag, value in (("--model", "moondream/parakeet-redux"),
                            ("--device", "cpu"), ("--threads", "8"), ("--port", "21999")):
            self.assertEqual(args[args.index(flag) + 1], value)


if __name__ == "__main__":
    unittest.main()
