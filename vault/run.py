#!/usr/bin/env python3
"""Pinned archival queue; no inference, remote code, or shared Hub cache writes."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlencode

VAULT = Path.home() / "storage/models/vault"
MOUNT = Path("/mnt/storage")
DISK = Path("/dev/disk/by-uuid/a2dcdf68-f962-4af5-bc1c-b90123cc9cf0")
RESERVE = 512 * 1024**3
MAX_HTTP_BYTES = 50_000_000_000  # huggingface-hub 1.31.0 transport limit.
STOP = False
CHILD = None


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as out:
        json.dump(data, out, indent=2, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    sync_dir(path.parent)


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def safe_path(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or any(x in ("", ".", "..") for x in value.split("/")):
        raise ValueError("Unsafe archive path")
    return value


def guard(vault=VAULT, required=RESERVE):
    if not MOUNT.is_mount() or MOUNT.stat().st_dev != DISK.stat().st_rdev:
        raise RuntimeError("Expected external disk is not mounted")
    if not vault.resolve().is_relative_to(MOUNT):
        raise RuntimeError("Vault is outside the external disk")
    if shutil.disk_usage(MOUNT).free < required:
        raise RuntimeError("External disk free-space reserve reached")


def configure(vault, use_xet=False):
    # Preserve the existing login token location. Keep every download/cache here.
    for key, value in {
        "HF_HUB_CACHE": str(vault / ".cache/hub"),
        "HF_XET_CACHE": str(vault / ".cache/xet"),
        "HF_ASSETS_CACHE": str(vault / ".cache/assets"),
        "HF_HUB_DISABLE_XET": "0" if use_xet else "1",
        "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY": "1",
        "HF_XET_NUM_CONCURRENT_RANGE_GETS": "4",
        "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
        "HF_XET_HIGH_PERFORMANCE": "0",
        "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        "HF_HUB_ETAG_TIMEOUT": "30",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_UPDATE_CHECK": "1",
        "HF_HUB_VERBOSITY": "critical",
    }.items():
        os.environ[key] = value
    logging.disable(logging.CRITICAL)


def merged_packages(source):
    packages = {}
    inputs = [(r, "original") for g in source["original_groups"] for r in g["repos"]]
    inputs += [(r, "nvfp4") for r in source["nvfp4_packages"]]
    for repo, category in inputs:
        name = safe_path(repo["repo"])
        if len(name.split("/")) != 2:
            raise ValueError("Expected owner/repository")
        entry = packages.setdefault(name, {"repo": name, "revision": repo["revision"],
                                           "categories": [], "files": {}})
        if entry["revision"] != repo["revision"]:
            raise ValueError("Conflicting revisions for " + name)
        entry["categories"].append(category)
        for original in repo["files"]:
            f = dict(original)
            safe_path(f["path"])
            if f["bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{40}", f["git_oid"]):
                raise ValueError("Invalid file metadata")
            previous = entry["files"].setdefault(f["path"], f)
            if previous != f:
                raise ValueError("Conflicting file metadata")
    return list(packages.values())


def prepare(source_path, vault):
    """Pin main once, and reject changed inventory entries before any weights download."""
    from huggingface_hub import HfApi
    guard(vault)
    vault.mkdir(parents=True, exist_ok=True)
    destination = vault / "manifest.json"
    if destination.exists():
        raise RuntimeError("Manifest exists; do not replace an active queue")
    packages = merged_packages(json.loads(Path(source_path).read_text()))
    api = HfApi()
    for repo in packages:
        info = api.model_info(repo["repo"], revision=repo["revision"] or "main", files_metadata=True)
        available = {f.rfilename: f for f in info.siblings}
        for f in repo["files"].values():
            actual = available[f["path"]]
            if actual.size != f["bytes"] or actual.blob_id != f["git_oid"]:
                raise RuntimeError("Inventory changed: " + repo["repo"] + "/" + f["path"])
            if actual.lfs and re.fullmatch(r"[0-9a-f]{64}", actual.lfs.sha256):
                if f["sha256"] and f["sha256"] != actual.lfs.sha256:
                    raise RuntimeError("Inventory hash changed")
                f["sha256"] = actual.lfs.sha256
        repo["revision"] = info.sha
        repo["files"] = sorted(repo["files"].values(), key=lambda f: (f["bytes"], f["path"]))
        repo["bytes"] = sum(f["bytes"] for f in repo["files"])
        print("Pinned", repo["repo"], len(repo["files"]), flush=True)
    packages.sort(key=lambda r: (r["bytes"], r["repo"]))
    manifest = {"schema": 1, "created": time.time(), "source": str(source_path),
                "total_bytes": sum(r["bytes"] for r in packages), "repos": packages}
    guard(vault, manifest["total_bytes"] + RESERVE)
    atomic_json(destination, manifest)
    print("Prepared", len(packages), "repositories;", manifest["total_bytes"], "bytes")


def verify(path, item):
    if path.stat().st_size != item["bytes"]:
        raise ValueError("File size mismatch")
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {item['bytes']}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            sha.update(chunk)
            git.update(chunk)
        os.fsync(stream.fileno())
    digest = sha.hexdigest()
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {item['bytes']}\n".encode()
    pointer_oid = hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()
    if item.get("sha256"):
        valid = digest == item["sha256"]
    else:
        valid = item["git_oid"] in (git.hexdigest(), pointer_oid)
    if not valid:
        raise ValueError("File checksum mismatch")
    sync_dir(path.parent)
    return {"bytes": item["bytes"], "mtime_ns": path.stat().st_mtime_ns, "sha256": digest}


def sam_mirror(item, local_dir):
    """Public publisher mirror. The original HF Git/LFS checksum remains mandatory."""
    import httpx
    path = local_dir / item["path"]
    partial = path.with_name(path.name + ".mirror.incomplete")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            verify(path, item)
            return path
        except ValueError:
            os.replace(path, path.with_name(path.name + f".corrupt-{time.time_ns()}"))
    offset = partial.stat().st_size if partial.exists() else 0
    if offset == item["bytes"]:
        return publish_mirror(partial, path, item)
    url = "https://modelscope.cn/api/v1/models/facebook/sam3/repo?" + urlencode(
        {"Revision": "f8ee5d16d5c30d0e87b1d22cbd358ce97761df56", "FilePath": item["path"]})
    # No HF credentials are passed to the mirror or its redirect hosts.
    with httpx.stream("GET", url, headers={"Range": f"bytes={offset}-"},
                      follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        content_range = response.headers.get("content-range")
        if content_range:
            # ModelScope's CDN also returns valid byte ranges with HTTP 200.
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match or int(match[1]) != offset or int(match[3]) != item["bytes"]:
                raise ValueError("Invalid mirror range")
        elif response.status_code == 200:
            offset = 0
        else:
            raise ValueError("Invalid mirror response")
        with partial.open("ab" if offset else "wb") as out:
            for chunk in response.iter_bytes(1024**2):
                if out.tell() + len(chunk) > item["bytes"]:
                    raise ValueError("Mirror sent excessive data")
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
    if partial.stat().st_size < item["bytes"]:
        raise OSError("Mirror transfer incomplete; keep bytes for resume")
    return publish_mirror(partial, path, item)


def publish_mirror(partial, path, item):
    try:
        verify(partial, item)
    except ValueError:
        os.replace(partial, partial.with_name(partial.name + f".corrupt-{time.time_ns()}"))
        raise
    os.replace(partial, path)
    sync_dir(path.parent)
    return path


def fetch(vault):
    from huggingface_hub import hf_hub_download
    request = json.loads((vault / "current.json").read_text())
    repo, item = request["repo"], request["file"]
    local_dir = vault / "huggingface" / repo["repo"] / repo["revision"]
    local_dir.mkdir(parents=True, exist_ok=True)
    guard(vault, RESERVE + item["bytes"])
    try:
        if repo["repo"] == "facebook/sam3" and item["path"] in ("sam3.pt", "model.safetensors"):
            path = sam_mirror(item, local_dir)
        else:
            path = Path(hf_hub_download(repo["repo"], item["path"], revision=repo["revision"],
                                        local_dir=local_dir))
        atomic_json(vault / "activity.json", {"phase": "verifying", "time": time.time()})
        try:
            result = verify(path, item)
        except ValueError:
            # Keep evidence and force the next attempt to fetch fresh content.
            os.replace(path, path.with_name(path.name + f".corrupt-{time.time_ns()}"))
            raise
        atomic_json(vault / "result.json", {"ok": True, "receipt": result})
    except Exception as exc:
        # Avoid signed URLs/tokens and unbounded dependency tracebacks in logs.
        response = getattr(exc, "response", None)
        atomic_json(vault / "result.json", {"ok": False, "error": type(exc).__name__,
                                            "http_status": getattr(response, "status_code", None)})
        return 1
    return 0


def partial_progress(local_dir):
    # Xet can preallocate a sparse file at its final length. Actual writes still
    # change mtime, so a fixed apparent size must not trigger the stall watchdog.
    stats = [(str(p), p.stat()) for p in sorted(local_dir.rglob("*.incomplete")) if p.is_file()]
    allocated = sum(min(s.st_size, s.st_blocks * 512) for _, s in stats)
    fingerprint = tuple((name, s.st_size, s.st_mtime_ns) for name, s in stats)
    return allocated, fingerprint


def stop(signum=None, frame=None):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.terminate()


def load_state(vault, manifest_hash):
    path = vault / "state.json"
    state = json.loads(path.read_text()) if path.exists() else {"manifest_sha256": manifest_hash, "repos": {}}
    if state["manifest_sha256"] != manifest_hash:
        raise ValueError("Queue manifest changed; review it before resuming")
    return state


def receipt_valid(path, receipt):
    return path.is_file() and path.stat().st_size == receipt["bytes"] and path.stat().st_mtime_ns == receipt["mtime_ns"]


def run(vault):
    global CHILD
    guard(vault)
    with (vault / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_raw = (vault / "manifest.json").read_bytes()
        manifest = json.loads(manifest_raw)
        state = load_state(vault, hashlib.sha256(manifest_raw).hexdigest())
        for repo in manifest["repos"]:
            rs = state["repos"].setdefault(repo["repo"], {"done": {}, "attempts": 0, "retry_at": 0})
            directory = vault / "huggingface" / repo["repo"] / repo["revision"]
            rs["done"] = {p: receipt for p, receipt in rs["done"].items() if receipt_valid(directory / p, receipt)}
        atomic_json(vault / "state.json", state)

        def status(phase, current=None, active_bytes=0):
            done_bytes = sum(f["bytes"] for rs in state["repos"].values() for f in rs["done"].values())
            complete = sum(len(state["repos"][r["repo"]]["done"]) == len(r["files"]) for r in manifest["repos"])
            atomic_json(vault / "status.json", {
                "updated": time.time(), "phase": phase, "current": current, "active_bytes": active_bytes,
                "verified_bytes": done_bytes, "total_bytes": manifest["total_bytes"],
                "completed_repos": complete, "total_repos": len(manifest["repos"]),
                "deferred": {k: {x: v.get(x) for x in ("error", "http_status", "retry_at")}
                             for k, v in state["repos"].items() if v.get("error")},
            })

        while not STOP:
            guard(vault)
            pending = [r for r in manifest["repos"] if len(state["repos"][r["repo"]]["done"]) < len(r["files"])]
            if not pending:
                status("complete")
                print("Archive complete; all selected files verified.", flush=True)
                # Stay inert under ktxsvc's Restart=always, including after reboot.
                while not STOP:
                    time.sleep(10)
                return
            repo = next((r for r in pending if state["repos"][r["repo"]]["retry_at"] <= time.time()), None)
            if repo is None:
                status("waiting-for-retry")
                time.sleep(10)
                continue
            rs = state["repos"][repo["repo"]]
            item = next(f for f in repo["files"] if f["path"] not in rs["done"])
            current = {"repo": repo["repo"], "file": item["path"], "bytes": item["bytes"],
                       "transport": "xet" if item["bytes"] > MAX_HTTP_BYTES else "http"}
            print("Downloading", current, flush=True)
            atomic_json(vault / "current.json", {"repo": {k: repo[k] for k in ("repo", "revision")}, "file": item})
            atomic_json(vault / "result.json", {"ok": False, "error": "WorkerInterrupted"})
            atomic_json(vault / "activity.json", {"phase": "downloading", "time": time.time()})
            directory = vault / "huggingface" / repo["repo"] / repo["revision"]
            CHILD = subprocess.Popen([sys.executable, __file__, "fetch", "--vault", str(vault)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            last_progress, last_change = None, time.monotonic()
            try:
                while CHILD.poll() is None and not STOP:
                    guard(vault)
                    count, progress = partial_progress(directory)
                    activity = json.loads((vault / "activity.json").read_text())
                    phase = activity["phase"]
                    if progress != last_progress:
                        last_progress, last_change = progress, time.monotonic()
                    # A 100 GB file can take a long time to read on a busy HDD.
                    timeout = 12 * 3600 if phase == "verifying" else 15 * 60
                    if time.monotonic() - last_change > timeout:
                        print("Worker stalled; retaining partial download for retry", flush=True)
                        CHILD.kill()
                        break
                    status(phase, current, min(count, item["bytes"]))
                    try:
                        CHILD.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if CHILD.poll() is None:
                    CHILD.terminate()
                try:
                    CHILD.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    CHILD.kill()
                    CHILD.wait()
                CHILD = None
            if STOP:
                status("paused", current)
                return
            result = json.loads((vault / "result.json").read_text())
            if result["ok"]:
                rs["done"][item["path"]] = result["receipt"]
                rs.update(attempts=0, retry_at=0, error=None, http_status=None)
                print("Verified", repo["repo"], item["path"], flush=True)
            else:
                rs["attempts"] += 1
                delay = min(21600, 60 * 2 ** min(rs["attempts"], 9))
                rs.update(retry_at=time.time() + delay, error=result.get("error"), http_status=result.get("http_status"))
                print("Deferred", repo["repo"], rs["error"], rs["http_status"], "retry in", delay, "seconds", flush=True)
            atomic_json(vault / "state.json", state)
            status("running")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "fetch", "status"))
    parser.add_argument("--vault", type=Path, default=VAULT)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    use_xet = False
    if args.command == "fetch":
        use_xet = json.loads((args.vault / "current.json").read_text())["file"]["bytes"] > MAX_HTTP_BYTES
    configure(args.vault, use_xet=use_xet)
    if args.command == "prepare":
        prepare(args.source, args.vault)
    elif args.command == "fetch":
        return fetch(args.vault)
    elif args.command == "status":
        status = json.loads((args.vault / "status.json").read_text())
        stamp = datetime.fromtimestamp(status["updated"]).astimezone().isoformat(timespec="seconds")
        print(f"{status['phase']} — updated {stamp}")
        print(f"Verified {status['verified_bytes'] / 10**9:,.2f} GB / {status['total_bytes'] / 10**12:.3f} TB; "
              f"{status['completed_repos']} / {status['total_repos']} repositories complete")
        if current := status.get("current"):
            print(f"{current['repo']}: {current['file']}")
            print(f"Current file: {status['active_bytes'] / 10**9:.3f} / {current['bytes'] / 10**9:.3f} GB")
        for repo, failure in status["deferred"].items():
            print(f"Retry pending: {repo}: {failure['error']} (HTTP {failure['http_status']})")
    else:
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        run(args.vault)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("Vault stopped safely:", type(exc).__name__, flush=True)
        sys.exit(1)
