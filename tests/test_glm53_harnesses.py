import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("switch", ROOT / "scripts/configure-glm53-harnesses.py")
switch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(switch)


class HarnessSwitchTest(unittest.TestCase):
    def test_targeted_preservation_and_idempotence(self):
        for host in ["snappy", "smarty", "scrappy"]:
            with self.subTest(host=host):
                providers = {"kortexa.ai": {"apiKey": "synthetic-private", "models": [{"id": "old-model"}]},
                             "openai-codex": {"auth": "oauth", "models": []},
                             "other": {"models": [{"id": "untouched"}]}}
                data = {".omp/agent/models.yml": {"providers": providers},
                        ".pi/agent/models.json": {"providers": providers},
                        ".omp/agent/config.yml": {"modelRoles": {"default": "kortexa.ai/old:high", "tiny": "tiny", "advisor": "advisor"}},
                        ".pi/agent/settings.json": {"defaultModel": "old", "defaultProvider": "old", "defaultThinkingLevel": "medium", "theme": "dark"}}
                files = {n: {"path": "/synthetic/" + n, "text": switch.migration.serialize(n, v)} for n, v in data.items()}
                result, changes = switch.plan(host, files)
                for n in [".omp/agent/models.yml", ".pi/agent/models.json"]:
                    p = result[n]["providers"]
                    self.assertEqual(p["kortexa.ai"]["apiKey"], "synthetic-private")
                    self.assertEqual(p["other"], providers["other"])
                    self.assertEqual(p["kortexa.ai"]["models"][0]["id"], "old-model")
                    self.assertEqual([m["id"] for m in p[switch.DUAL]["models"]], [switch.MODEL])
                roles = result[".omp/agent/config.yml"]["modelRoles"]
                self.assertEqual((roles["tiny"], roles["advisor"]), ("tiny", "advisor"))
                expected = "kortexa.ai/" + switch.MODEL + ":high" if host == "snappy" else "openai-codex/gpt-6-luna"
                self.assertEqual(roles["default"], expected)
                settings = result[".pi/agent/settings.json"]
                self.assertEqual((settings["theme"], settings["defaultThinkingLevel"]), ("dark", "medium"))
                updated = {n: {"path": v["path"], "text": changes.get(n, {}).get("after", v["text"])} for n, v in files.items()}
                self.assertEqual(switch.plan(host, updated)[1], {})

    def test_native_windows_and_mira_keep_identity_and_auxiliary(self):
        for name in ["windows-hermes/config.yaml", ".hermes/profiles/mira/config.yaml"]:
            data = {"model": {"provider": "custom:old", "default": "old"},
                    "providers": {"kortexa.ai": {"api_key": "synthetic-private", "models": {"old": {}}}},
                    "auxiliary": {"title": {"model": "unchanged"}}, "identity": "mira"}
            files = {name: {"path": "/synthetic/config.yaml", "text": switch.migration.serialize(name, data)}}
            result, _ = switch.plan("scrappy", files)
            actual = result[name]
            self.assertEqual(actual["identity"], "mira")
            self.assertEqual(actual["auxiliary"], data["auxiliary"])
            self.assertEqual(actual["model"]["provider"], "openai-codex")
            self.assertEqual(actual["model"]["api_mode"], "responses")
            self.assertIn(switch.MODEL, actual["model_overrides"]["custom:kortexa-dual"])


if __name__ == "__main__":
    unittest.main()
