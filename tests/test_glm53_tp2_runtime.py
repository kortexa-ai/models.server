import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("glm53", ROOT / "scripts/glm53_tp2_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class GlmPairTest(unittest.TestCase):
    def setUp(self):
        self.model = runtime.load(ROOT / "glm-5.3-flash-2x-dgx")

    def test_port_and_capacity_contract(self):
        ports = [json.loads(p.read_text())["port"] for p in ROOT.glob("*/model.json")]
        self.assertEqual(ports.count(2070), 1)
        self.assertEqual(self.model["context_window"], 524288)
        c = self.model["glm53_tp2"]
        env = c["environment"]
        self.assertEqual(env["BATCHED_TOKENS"], 6919)
        self.assertEqual(env["KV_BYTES"], 6 * 1024 ** 3)
        self.assertEqual(env["MAX_SEQS"], 8)
        self.assertEqual({row[2] for row in env["SPEC_TABLE"]}, set(range(1, 8)))
        for batch in range(1, 9):
            for length in range(1, 8):
                self.assertIn(batch * (length + 1), env["CAPTURE_SIZES"])

    def test_tp4_flags_replaced_and_ambient_profile_knobs_cannot_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / "profiles").mkdir()
            (source / "profiles/current.env").write_text(
                'test "$PF3_ARM" = off && test "$GATHER_ROUTE" = 0\n'
                'MAX_SEQS=32\nEXTRA_ENV="GLM_PREFILL_SHARD=1 GLM_PREFILL_SHARD_PAD=1 KEEP_ME=1"\n')
            script = runtime.environment(self.model, source)
            result = subprocess.check_output(["bash", "-c", script + '\nprintf "%s\\n%s" "$MAX_SEQS" "$EXTRA_ENV"'], text=True)
            self.assertTrue(result.startswith("8\n"))
            self.assertNotIn("GLM_PREFILL_SHARD=1", result)
            self.assertNotIn("GLM_PREFILL_SHARD_PAD=1", result)
            self.assertIn("KEEP_ME=1", result)
            self.assertIn("MAX_JOBS=2", result)

    def test_rank_ownership_is_not_name_only(self):
        m = self.model
        labels = {runtime.LABEL + ".model": m["id"], runtime.LABEL + ".rank": "0",
                  runtime.LABEL + ".fingerprint": runtime.fingerprint(m)}
        info = {"Config": {"Labels": labels}, "Image": m["glm53_tp2"]["image_id"]}
        self.assertTrue(runtime.owned(m, info, 0))
        self.assertFalse(runtime.owned(m, info, 1))
        labels[runtime.LABEL + ".fingerprint"] = "foreign"
        self.assertFalse(runtime.owned(m, info, 0))

    def test_foreign_container_is_never_stopped(self):
        with patch.object(runtime, "node", return_value={"rank": 0}), \
             patch.object(runtime, "output", return_value="") as query, \
             patch.object(runtime, "execute") as execute:
            runtime.run(self.model, stop=True)
            filters = query.call_args.args[0]
            self.assertIn("label=" + runtime.LABEL + ".model=" + self.model["id"], filters)
            self.assertIn("label=" + runtime.LABEL + ".rank=0", filters)
            execute.assert_not_called()

    def test_invalid_host_and_missing_setup_fail_before_start(self):
        with patch.object(runtime.platform, "system", return_value="Darwin"):
            with self.assertRaises(ValueError):
                runtime.node(self.model)
        with tempfile.TemporaryDirectory() as tmp, patch.object(runtime, "locations", return_value=(Path(tmp), Path(tmp), Path(tmp))):
            with self.assertRaises(FileNotFoundError):
                runtime.validate_ready(self.model)


if __name__ == "__main__":
    unittest.main()
