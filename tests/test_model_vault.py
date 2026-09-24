import hashlib
import contextlib
import io
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("vault", Path(__file__).parents[1] / "vault/run.py")
vault = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vault)


def item(data, lfs=False):
    sha = hashlib.sha256(data).hexdigest()
    blob = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {len(data)}\n".encode() if lfs else data
    return {"path": "weights.bin", "bytes": len(data), "sha256": sha if lfs else None,
            "git_oid": hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest()}


class VaultTests(unittest.TestCase):
    def manifests(self):
        a = {"repo": "owner/old", "revision": "a" * 40, "bytes": 1,
             "files": [item(b"a")], "categories": ["original"]}
        b = {"repo": "owner/new", "revision": "b" * 40, "bytes": 1,
             "files": [item(b"b")], "categories": ["original"]}
        return {"repos": [a], "total_bytes": 1}, {"repos": [a, b], "total_bytes": 2}

    def test_append_recovery_preserves_receipts_at_each_commit_stage(self):
        before, after = self.manifests()
        for manifest, state_manifest in ((before, before), (after, before), (after, after)):
            with self.subTest(stage=len(manifest["repos"]), state=len(state_manifest["repos"])), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                receipts = {"owner/old": {"done": {"weights.bin": {"bytes": 1, "sha256": "kept"}}, "retry_at": 123}}
                state = {"manifest_sha256": vault.manifest_hash(state_manifest), "repos": receipts}
                vault.atomic_json(root / "manifest.json", manifest)
                vault.atomic_json(root / "state.json", state)
                vault.atomic_json(root / "manifest-update.json", {"before": before, "after": after})
                vault.recover_update(root)
                self.assertEqual(json.loads((root / "manifest.json").read_text()), after)
                saved = vault.load_state(root, vault.manifest_hash(after))
                self.assertEqual(saved["repos"], receipts)
                self.assertFalse((root / "manifest-update.json").exists())
                vault.recover_update(root)  # Recovery is idempotent after completion.

    def test_append_rejects_replacement_of_existing_pinned_files(self):
        before, after = self.manifests()
        after = json.loads(json.dumps(after))
        after["repos"][0]["revision"] = "c" * 40
        with self.assertRaises(ValueError):
            vault.validate_append(before, after)

    def test_append_capacity_check_and_explicit_queue_overcommit(self):
        before, after = self.manifests()
        additions = {"repos": [after["repos"][1]], "total_bytes": 1}
        with tempfile.TemporaryDirectory() as d, patch.object(vault, "guard"), \
             patch.object(vault.shutil, "disk_usage", return_value=SimpleNamespace(free=0)), \
             contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)
            vault.atomic_json(root / "manifest.json", before)
            vault.atomic_json(root / "state.json", {"manifest_sha256": vault.manifest_hash(before), "repos": {}})
            vault.atomic_json(root / "additions.json", additions)
            with self.assertRaises(RuntimeError):
                vault.append_queue(root, root / "additions.json")
            self.assertEqual(json.loads((root / "manifest.json").read_text()), before)
            vault.append_queue(root, root / "additions.json", allow_capacity_shortfall=True)
            saved = json.loads((root / "manifest.json").read_text())
            self.assertEqual(len(saved["repos"]), 2)
            vault.load_state(root, vault.manifest_hash(saved))
            vault.append_queue(root, root / "additions.json", allow_capacity_shortfall=True)
            self.assertEqual(json.loads((root / "manifest.json").read_text()), saved)

    def test_xet_transport_configuration_keeps_hdd_and_cache_bounds(self):
        with patch.dict(os.environ, {}, clear=True):
            vault.configure(Path("/vault"), use_xet=True)
            self.assertEqual(os.environ["HF_HUB_DISABLE_XET"], "0")
            self.assertEqual(os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"], "1")
            self.assertEqual(os.environ["HF_XET_NUM_CONCURRENT_RANGE_GETS"], "4")
            self.assertEqual(os.environ["HF_XET_CACHE"], "/vault/.cache/xet")
            vault.configure(Path("/vault"))
            self.assertEqual(os.environ["HF_HUB_DISABLE_XET"], "1")

    def test_sparse_partial_writes_advance_progress_without_changing_length(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "weights.incomplete"
            with p.open("wb") as out:
                out.truncate(1024**3)
            size, before = vault.partial_progress(root)
            self.assertLess(size, p.stat().st_size)
            with p.open("r+b") as out:
                out.write(b"x" * 4096)
            os.utime(p, ns=(p.stat().st_atime_ns, before[0][2] + 1000000))
            size, after = vault.partial_progress(root)
            self.assertNotEqual(before, after)
            self.assertEqual(p.stat().st_size, 1024**3)
            self.assertGreaterEqual(size, 4096)

    def test_checksum_rejects_corruption_with_same_size(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "weights.bin"
            p.write_bytes(b"good")
            for lfs in (True, False):
                receipt = vault.verify(p, item(b"good", lfs))
                self.assertTrue(vault.receipt_valid(p, receipt))
            p.write_bytes(b"evil")
            with self.assertRaises(ValueError):
                vault.verify(p, item(b"good", True))
            self.assertFalse(vault.receipt_valid(p, receipt))

    def test_masked_lfs_checksum_uses_pointer_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "weights.bin"
            p.write_bytes(b"original")
            f = item(b"original", True)
            f["sha256"] = None
            vault.verify(p, f)

    def test_duplicate_repo_merges_files_without_double_counting(self):
        a, b = item(b"a"), item(b"b")
        b["path"] = "quant.bin"
        base = {"repo": "owner/repo", "revision": "a" * 40, "files": [a]}
        quant = dict(base, files=[b])
        source = {"original_groups": [{"repos": [base]}], "nvfp4_packages": [quant]}
        merged = vault.merged_packages(source)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["files"]), 2)
        quant["revision"] = "b" * 40
        with self.assertRaises(ValueError):
            vault.merged_packages(source)

    def test_path_traversal_rejected(self):
        for p in ("/absolute", "../parent", "a/../b", "a//b", "./relative"):
            with self.assertRaises(ValueError):
                vault.safe_path(p)

    def test_missing_or_wrong_disk_stops_before_writes(self):
        with patch.object(Path, "is_mount", return_value=False):
            with self.assertRaises(RuntimeError):
                vault.guard()

    def test_progress_survives_restart_and_manifest_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = vault.load_state(root, "abc")
            state["repos"]["owner/repo"] = {"done": {"weights.bin": {"bytes": 4}}, "retry_at": 123}
            vault.atomic_json(root / "state.json", state)
            self.assertEqual(vault.load_state(root, "abc"), state)
            with self.assertRaises(ValueError):
                vault.load_state(root, "changed")

    def test_modelscope_http200_range_resumes_existing_partial(self):
        class Response:
            status_code = 200
            headers = {"content-range": "bytes 2-5/6"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_bytes(self, size): return iter([b"cdef"])
        def stream(method, url, **kwargs):
            self.assertEqual(kwargs["headers"], {"Range": "bytes=2-"})
            self.assertNotIn("Authorization", kwargs["headers"])
            return Response()
        with tempfile.TemporaryDirectory() as d, patch.dict("sys.modules", {"httpx": SimpleNamespace(stream=stream)}):
            root = Path(d)
            (root / "weights.bin.mirror.incomplete").write_bytes(b"ab")
            result = vault.sam_mirror(item(b"abcdef", True), root)
            self.assertEqual(result.read_bytes(), b"abcdef")

    def test_mirror_bad_checksum_quarantines_partial_for_fresh_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            partial = root / "file.incomplete"
            partial.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                vault.publish_mirror(partial, root / "weights.bin", item(b"yes", True))
            self.assertFalse(partial.exists())
            self.assertEqual(len(list(root.glob("*.corrupt-*"))), 1)

    def test_failed_repo_yields_and_restart_preserves_verified_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            repos = [{"repo": name, "revision": "a" * 40, "files": [item(b"x")], "bytes": 1}
                     for name in ("owner/blocked", "owner/working")]
            vault.atomic_json(root / "manifest.json", {"repos": repos, "total_bytes": 2})
            calls = []
            def worker(*args, **kwargs):
                current = json.loads((root / "current.json").read_text())
                repo = current["repo"]
                calls.append(repo["repo"])
                if repo["repo"] == "owner/blocked":
                    result = {"ok": False, "error": "GatedRepoError", "http_status": 403}
                else:
                    p = root / "huggingface" / repo["repo"] / repo["revision"] / "weights.bin"
                    p.parent.mkdir(parents=True)
                    p.write_bytes(b"x")
                    result = {"ok": True, "receipt": vault.verify(p, item(b"x"))}
                vault.atomic_json(root / "result.json", result)
                return SimpleNamespace(poll=lambda: 0, wait=lambda **kw: 0)
            with patch.object(vault, "guard"), patch.object(vault.subprocess, "Popen", side_effect=worker), \
                 patch.object(vault.time, "sleep", side_effect=lambda _: vault.stop()), \
                 patch.object(vault, "STOP", False), contextlib.redirect_stdout(io.StringIO()):
                vault.run(root)
                self.assertEqual(calls, ["owner/blocked", "owner/working"])
                state = json.loads((root / "state.json").read_text())
                self.assertIn("weights.bin", state["repos"]["owner/working"]["done"])
                self.assertGreater(state["repos"]["owner/blocked"]["retry_at"], 0)
                vault.STOP = False
                vault.run(root)
                self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
