#!/usr/bin/env python3
"""Prepare pinned NInfer artifacts, or serve them offline on Smarty's 6000."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-a71210ca-e14a-755a-88bb-77f53a2102f6"
RECIPE_REPO = "https://github.com/satellitedown/fast-long-context-cinference.git"
RECIPE_REVISION = "104b236e557741b990ff084ef6512801689c9613"
RUNTIME_REPO = "https://github.com/satellitedown/cinference.git"
RUNTIME_REVISION = "b74044fb0a319cd2a737cb7108012e6344b96dac"
ENGINE = ROOT / ".engines/cinference"
NATIVE = ENGINE / "runtime/ninfer"
BINARY = NATIVE / "build/apps/ninfer-serve"


def command(args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def output(args, **kwargs):
    return command(args, text=True, capture_output=True, **kwargs).stdout.strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def check_platform():
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise ValueError("Cinference requires Linux x86_64 and Smarty's RTX PRO 6000.")


def check_checkout(path, repo, revision):
    if output(["git", "-C", path, "remote", "get-url", "origin"]) != repo:
        raise ValueError(f"Unexpected origin in {path}; leaving it unchanged.")
    if output(["git", "-C", path, "rev-parse", "HEAD"]) != revision:
        raise ValueError(f"Unexpected revision in {path}; leaving it unchanged.")
    if output(["git", "-C", path, "status", "--porcelain"]):
        raise ValueError(f"Local changes in {path}; leaving it unchanged.")


def artifact_paths(config):
    artifact = config["cinference"]["artifact"]
    directory = ENGINE / "models" / artifact["directory"]
    source = directory / artifact["filename"]
    served = source if artifact["format_version"] == 3 else source.with_name(source.stem + "_v3.ninfer")
    return artifact, source, served, directory / "models-server-verified.json"


def source_identity(artifact):
    return {key: artifact[key] for key in ("repo", "revision", "filename", "sha256", "size_bytes", "format_version")}


def verify_artifact(config):
    artifact, source, served, receipt_path = artifact_paths(config)
    if not receipt_path.is_file() or not served.is_file():
        raise ValueError("Prepared artifact missing. Run scripts/setup-cinference.sh for this model first.")
    receipt = read_json(receipt_path)
    if receipt["source"] != source_identity(artifact) or receipt["runtime_revision"] != RUNTIME_REVISION:
        raise ValueError("Artifact receipt does not match the pinned model/runtime. Run setup again.")
    if served.stat().st_size != receipt["served_size_bytes"] or sha256(served) != receipt["served_sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {served}")
    if artifact["format_version"] == 3 and receipt["served_sha256"] != artifact["sha256"]:
        raise ValueError("Artifact receipt does not match the pinned source checksum.")
    return served


def prepare(config):
    check_platform()
    if not ENGINE.exists():
        ENGINE.mkdir(parents=True)
        command(["git", "init", ENGINE])
        command(["git", "-C", ENGINE, "remote", "add", "origin", RECIPE_REPO])
        command(["git", "-C", ENGINE, "fetch", "--depth", "1", "origin", RECIPE_REVISION])
        command(["git", "-C", ENGINE, "checkout", "--detach", RECIPE_REVISION])
    check_checkout(ENGINE, RECIPE_REPO, RECIPE_REVISION)
    manifest = read_json(ENGINE / "runtime-manifest.json")["runtime"]
    if manifest["revision"] != RUNTIME_REVISION or manifest["repo_id"] != "satellitedown/cinference":
        raise ValueError("Recipe runtime manifest differs from the tested runtime.")
    # This upstream installer installs only isolated tools/CUDA and builds with two
    # jobs. It does not change drivers, system packages, or managed services.
    if not BINARY.is_file():
        command(["bash", ENGINE / "scripts/install.sh"])
    check_checkout(NATIVE, RUNTIME_REPO, RUNTIME_REVISION)
    artifact, source, served, receipt_path = artifact_paths(config)
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        command([ENGINE / ".venv/bin/hf", "download", artifact["repo"],
                 artifact["filename"], "README.md", "--revision", artifact["revision"],
                 "--local-dir", source.parent])
    if source.stat().st_size != artifact["size_bytes"] or sha256(source) != artifact["sha256"]:
        raise ValueError(f"Source checksum/size mismatch: {source}; no conversion performed.")
    if artifact["format_version"] == 2:
        if served.exists():
            # Never bless or overwrite an unverified prior conversion.
            verify_artifact(config)
        else:
            command([sys.executable, NATIVE / "tools/upgrade_ninfer_v2_to_v3.py", source, served])
    summary = json.loads(output([sys.executable, "-m", "tools.artifact.inspect", served, "--json"], cwd=NATIVE))
    for component in ("text", "vision", "dflash2"):
        if component not in summary["components"]:
            raise ValueError(f"Artifact is missing the required {component} component.")
    # The v2 upgrader preserves weights but generates a random artifact UUID.
    # Pin the input globally and record the verified output digest locally.
    write_json(receipt_path, {
        "source": source_identity(artifact), "runtime_revision": RUNTIME_REVISION,
        "served_sha256": sha256(served), "served_size_bytes": served.stat().st_size,
        "components": summary["components"],
    })
    write_json(ENGINE / "models/models-server-runtime.json", {
        "recipe_revision": RECIPE_REVISION, "runtime_revision": RUNTIME_REVISION,
        "binary_sha256": sha256(BINARY),
    })
    print(f"Prepared {config['id']}: {served}\nNo service was installed or started.")


def serve_args(config, served, options):
    engine = config["cinference"]
    args = [str(BINARY), str(served), "--model-id", config["id"],
            "--host", options.host or config["host"],
            "--port", str(options.port or config["port"]), "--device", "0",
            "--max-context", str(config["context_window"]),
            "--kv-capacity", str(config["context"]),
            "--max-concurrency", str(config["parallel"]),
            "--kv-dtype", config["cache_type"],
            "--prefill-chunk", str(engine["prefill_chunk"]),
            "--spec", engine["spec"], "--draft-tokens", str(engine["draft_tokens"]),
            "--pending-timeout-ms", str(engine["pending_timeout_ms"]),
            "--default-max-tokens", str(engine["default_max_tokens"]),
            "--presence-penalty", "0", "--frequency-penalty", "0",
            "--lm-head-draft", "--preserve-thinking"]
    if config["multimodal"]:
        args.append("--vision")
    if options.request_log_jsonl:
        args.extend(["--request-log-jsonl", options.request_log_jsonl])
    return args


def serve(config, options):
    check_platform()
    if os.environ.get("CUDA_VISIBLE_DEVICES", GPU_UUID) != GPU_UUID:
        raise ValueError(f"This model is pinned to {GPU_UUID}; refusing a different CUDA_VISIBLE_DEVICES.")
    # Native serving uses SO_REUSEPORT. Do not join an existing listener.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((options.host or config["host"], options.port or config["port"]))
    check_checkout(NATIVE, RUNTIME_REPO, RUNTIME_REVISION)
    receipt = read_json(ENGINE / "models/models-server-runtime.json")
    if (receipt["runtime_revision"] != RUNTIME_REVISION or receipt["recipe_revision"] != RECIPE_REVISION
            or sha256(BINARY) != receipt["binary_sha256"]):
        raise ValueError("Runtime receipt/checksum mismatch. Run setup again.")
    served = verify_artifact(config)
    row = output(["nvidia-smi", f"--id={GPU_UUID}", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"])
    uuid, free = (field.strip() for field in row.split(","))
    if uuid != GPU_UUID or int(free) < 51200:
        raise ValueError("Need at least 50 GiB free on the RTX PRO 6000 (about 40 GiB model + 10 GiB headroom). Check running services first.")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    libraries = str(ENGINE / ".cuda-toolkit/nvidia/cu13/lib")
    environment["LD_LIBRARY_PATH"] = libraries + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    args = serve_args(config, served, options)
    print(f"Serving {config['id']} on {args[args.index('--host') + 1]}:{args[args.index('--port') + 1]} using {GPU_UUID}", flush=True)
    os.execve(BINARY, args, environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("setup", "run"))
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--request-log-jsonl")
    options = parser.parse_args()
    config = read_json(options.model_dir / "model.json")
    if options.operation == "setup":
        prepare(config)
    else:
        serve(config, options)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"Cinference: {error}")
