import hashlib
import importlib.util
import json
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


if __name__ == "__main__":
    unittest.main()
