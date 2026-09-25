#!/usr/bin/env python3
"""Bounded HTTP/Xet comparison using fresh pinned shards as real archive progress."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import run as vault


def evaluate(trials):
    if len(trials) != 4 or [r["transport"] for r in trials] != ["http", "xet", "xet", "http"]:
        raise ValueError("Four completed HTTP-Xet-Xet-HTTP trials are required")
    rates = {}
    individual = {}
    for transport in ("http", "xet"):
        selected = [r for r in trials if r["transport"] == transport]
        rates[transport] = sum(r["bytes"] for r in selected) / sum(r["elapsed_seconds"] for r in selected)
        individual[transport] = [r["bytes"] / r["elapsed_seconds"] for r in selected]
    ratio = rates["xet"] / rates["http"]
    # Require an improvement larger than modest variations in competing traffic.
    better = ratio >= 1.2 and min(individual["xet"]) >= max(individual["http"]) * 1.1
    return {"prefer_xet": better, "xet_speedup": ratio, "bytes_per_second": rates}


def fresh_shards(root, manifest, state):
    from huggingface_hub._local_folder import get_local_download_paths
    current_path = root / "current.json"
    current = json.loads(current_path.read_text()) if current_path.exists() else {}
    for repo in manifest["repos"]:
        directory = root / "huggingface" / repo["repo"] / repo["revision"]
        candidates = []
        for item in repo["files"]:
            if not (4_000_000_000 <= item["bytes"] <= 6_000_000_000) or not item["path"].endswith(".safetensors"):
                continue
            if item["path"] in state["repos"].get(repo["repo"], {}).get("done", {}):
                continue
            if current.get("repo", {}).get("repo") == repo["repo"] and current.get("file", {}).get("path") == item["path"]:
                continue
            paths = get_local_download_paths(directory, item["path"])
            prefix = paths.incomplete_path(item.get("sha256") or item["git_oid"]).name.split(".", 1)[0]
            if paths.file_path.exists() or any(paths.metadata_path.parent.glob(prefix + ".*.incomplete")):
                continue
            candidates.append(item)
        # Same repository and effectively equal sizes limit comparison differences.
        for start in range(len(candidates) - 3):
            selected = candidates[start:start + 4]
            sizes = [f["bytes"] for f in selected]
            if max(sizes) <= min(sizes) * 1.01:
                return repo, selected
    raise ValueError("No four fresh, similarly sized pinned shards available")


def compare(root, report_path):
    vault.guard(root)
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        vault.recover_update(root)
        manifest = json.loads((root / "manifest.json").read_text())
        state = vault.load_state(root, vault.manifest_hash(manifest))
        repo, items = fresh_shards(root, manifest, state)
        report = {"started": time.time(), "manifest_sha256": state["manifest_sha256"],
                  "repo": repo["repo"], "revision": repo["revision"], "trials": [], "phase": "running",
                  "xet_range_gets": 16}
        vault.atomic_json(report_path, report)
        rs = state["repos"].setdefault(repo["repo"], {"done": {}, "attempts": 0, "retry_at": 0})
        try:
            for transport, item in zip(("http", "xet", "xet", "http"), items):
                vault.guard(root, vault.RESERVE + item["bytes"])
                vault.atomic_json(root / "current.json", {"repo": {k: repo[k] for k in ("repo", "revision")},
                                                          "file": item, "transport": transport, "xet_range_gets": 16})
                vault.atomic_json(root / "result.json", {"ok": False, "error": "BenchmarkInterrupted"})
                vault.atomic_json(root / "activity.json", {"phase": "downloading", "time": time.time()})
                report["current"] = {"transport": transport, "file": item["path"], "started": time.time()}
                vault.atomic_json(report_path, report)
                print("Trial", transport, item["path"], item["bytes"], flush=True)
                started = time.monotonic()
                child = subprocess.Popen([sys.executable, str(Path(vault.__file__)), "fetch", "--vault", str(root)],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    # Bound each trial even if the network or storage stops responding.
                    while child.poll() is None:
                        vault.guard(root)
                        if time.monotonic() - started > 900:
                            raise TimeoutError("Benchmark trial exceeded 15 minutes")
                        try:
                            child.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                elapsed = time.monotonic() - started
                result = json.loads((root / "result.json").read_text())
                if child.returncode != 0 or not result["ok"]:
                    raise RuntimeError("Benchmark worker failed: " + result.get("error", "unknown"))
                path = root / "huggingface" / repo["repo"] / repo["revision"] / item["path"]
                if not vault.receipt_valid(path, result["receipt"]):
                    raise ValueError("Benchmark result does not match the archived file")
                rs["done"][item["path"]] = result["receipt"]
                vault.atomic_json(root / "state.json", state)
                trial = {"transport": transport, "file": item["path"], "bytes": item["bytes"],
                         "elapsed_seconds": elapsed, "download_seconds": result["download_seconds"],
                         "verification_seconds": result["verification_seconds"],
                         "xet_range_gets": 16 if transport == "xet" else None}
                report["trials"].append(trial)
                vault.atomic_json(report_path, report)
                print("Verified", transport, round(item["bytes"] / elapsed / 10**6, 2), "MB/s", flush=True)
            decision = evaluate(report["trials"])
            report.update(phase="complete", finished=time.time(), decision=decision)
            vault.atomic_json(report_path, report)
            if decision["prefer_xet"]:
                vault.atomic_json(root / "transport-policy.json", {"prefer_xet": True, "benchmark": str(report_path),
                                  "measured_speedup": decision["xet_speedup"], "xet_range_gets": 16,
                                  "updated": time.time()})
            print("Decision", json.dumps(decision), flush=True)
        except BaseException as exc:
            report.update(phase="failed", error=type(exc).__name__, finished=time.time())
            vault.atomic_json(report_path, report)
            raise


def interrupted(signum, frame):
    raise InterruptedError("Benchmark interrupted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=vault.VAULT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    vault.configure(args.vault)
    compare(args.vault, args.report)
