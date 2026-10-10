#!/usr/bin/env python3
"""Separate managed lifecycle for the pinned-main GLM experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import glm53_tp2_runtime as stable

ROOT = Path(__file__).resolve().parents[1]
LABEL = stable.LABEL


def load(path):
    path = Path(path)
    if path.is_dir():
        path /= "model.json"
    model = json.loads(path.read_text())
    c = model["glm53_main"]
    if model["default_engine"] != "vllm-glm53-main":
        raise ValueError("Wrong engine")
    if sorted(n["rank"] for n in c["nodes"].values()) != [0, 1]:
        raise ValueError("Exactly two ranks required")
    if model["context"] != model["context_window"]:
        raise ValueError("Context fields disagree")
    if c["decode_context_parallel_size"] not in (1, 2):
        raise ValueError("DCP must divide TP2")
    if c["cp_kv_cache_interleave_size"] % 4:
        raise ValueError("GLM kpool DCP interleave must be a multiple of four")
    return model


def adapter(model):
    # Reuse the stable artifact and hardware checks without editing its source,
    # manifest, receipt or fingerprint. No stable launcher is invoked here.
    c = model["glm53_main"]
    return {**model, "glm53_tp2": {**c, "environment": {"MASTER_PORT": c["master_port"]}}}


def fingerprint(model):
    payload = json.dumps(model, sort_keys=True).encode()
    payload += Path(__file__).read_bytes() + Path(stable.__file__).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def receipt_path(model):
    return ROOT / ".engines/glm53-main" / model["id"] / "receipt.json"


def container_name(model, rank):
    return model["id"] + "-" + fingerprint(model)[:12] + f"-r{rank}"


def image_info(model):
    c = model["glm53_main"]
    image = json.loads(stable.output(["docker", "image", "inspect", c["image_id"]]))[0]
    labels = image["Config"].get("Labels") or {}
    if image["Id"] != c["image_id"]:
        raise ValueError("Wrong image")
    for field in ("vllm_revision", "native_revision"):
        label = "ai.kortexa.glm53." + field.replace("_", "-")
        if labels.get(label) != c[field]:
            raise ValueError(f"Wrong image provenance: {field}")
    return image


def setup(model):
    stable.node(adapter(model))
    image_info(model)
    # A still-valid stable receipt proves the same bytes without re-reading
    # 100+ GiB every time an experimental cache knob changes.
    prior = stable.load(ROOT / "glm-5.3-flash-2x-dgx")
    stable.validate_ready(prior)
    prior_receipt = json.loads((stable.locations(prior)[0] / "receipt.json").read_text())
    prior_hashes = dict(stable.artifact_files(prior))
    states = {}
    for path, expected in stable.artifact_files(adapter(model)):
        before = stable.file_state(path)
        reused = (prior_hashes.get(path) == expected
                  and prior_receipt["files"].get(str(path)) == before)
        if not reused and stable.digest(path) != expected:
            raise ValueError(f"Artifact checksum mismatch: {path}")
        if stable.file_state(path) != before:
            raise ValueError(f"Artifact changed during validation: {path}")
        states[str(path)] = before
    index = json.loads((Path(model["glm53_main"]["model_path"])
                        / "model.safetensors.index.json").read_text())
    if not set(index["weight_map"].values()) <= set(model["glm53_main"]["target_sha256"]):
        raise ValueError("Index references an unpinned shard")
    Path(model["glm53_main"]["cache_path"]).mkdir(parents=True, exist_ok=True)
    destination = receipt_path(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"fingerprint": fingerprint(model), "files": states}
    with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False) as stream:
        json.dump(receipt, stream, indent=2)
        temporary = stream.name
    os.replace(temporary, destination)
    print("Experimental setup verified; stable recipe untouched; no service started.")


def validate_ready(model):
    receipt = json.loads(receipt_path(model).read_text())
    if receipt["fingerprint"] != fingerprint(model):
        raise ValueError("Experimental recipe changed: run explicit setup again")
    for path, _ in stable.artifact_files(adapter(model)):
        if receipt["files"].get(str(path)) != stable.file_state(path):
            raise ValueError(f"Artifact changed since setup: {path}")
    image_info(model)


def render_command(model, rank):
    c = model["glm53_main"]
    nodes = sorted(c["nodes"].values(), key=lambda n: n["rank"])
    n = nodes[rank]
    args = ["docker", "create", "--name", container_name(model, rank),
            "--gpus", "device=" + n["gpu_uuid"], "--network", "host", "--ipc", "host",
            "--memory", c["container_memory"], "--memory-swap", c["container_memory"],
            "--ulimit", "memlock=-1:-1", "--cap-add", "IPC_LOCK",
            "--device", "/dev/infiniband:/dev/infiniband"]
    for key, value in {"model": model["id"], "fingerprint": fingerprint(model),
                       "rank": str(rank)}.items():
        args += ["--label", LABEL + "." + key + "=" + value]
    for source, destination in ((c["model_path"], "/model:ro"),
                                (c["draft_path"], "/draft:ro"),
                                (c["cache_path"], "/cache")):
        args += ["-v", source + ":" + destination]
    env = {**c["environment"], "VLLM_HOST_IP": n["ip"], "VLLM_USE_RUST_FRONTEND": "0",
           "GLM_EXPERIMENT_FINGERPRINT": fingerprint(model)}
    for key, value in env.items():
        args += ["-e", key + "=" + str(value)]
    args += ["--entrypoint", "vllm", c["image_id"], "serve", "/model",
             "--served-model-name", model["id"], "--host", model["host"],
             "--port", str(model["port"]), "--trust-remote-code",
             "--tensor-parallel-size", "2", "--distributed-executor-backend", "mp",
             "--nnodes", "2", "--node-rank", str(rank),
             "--master-addr", nodes[0]["ip"], "--master-port", str(c["master_port"]),
             "--max-model-len", str(model["context"]), "--max-num-seqs", str(model["parallel"]),
             "--max-num-batched-tokens", str(c["max_num_batched_tokens"]),
             "--kv-cache-memory-bytes", str(c["kv_cache_memory_bytes"]),
             "--kv-cache-dtype", model["cache_type"],
             "--decode-context-parallel-size", str(c["decode_context_parallel_size"]),
             "--cp-kv-cache-interleave-size", str(c["cp_kv_cache_interleave_size"]),
             "--mamba-ssm-cache-dtype", c["mamba_ssm_cache_dtype"],
             "--attention-backend", c["attention_backend"],
             "--moe-backend", c["moe_backend"], "--linear-backend", c["linear_backend"],
             "--enable-prefix-caching", "--enable-chunked-prefill",
             "--limit-mm-per-prompt", json.dumps({"image": c["image_limit"]}),
             "--mm-processor-cache-gb", str(c["mm_processor_cache_gb"]),
             "--reasoning-parser", c["reasoning_parser"], "--enable-auto-tool-choice",
             "--tool-call-parser", c["tool_call_parser"],
             "--default-chat-template-kwargs", json.dumps({"reasoning_effort": "low"}),
             "--no-enable-flashinfer-autotune"]
    if c["speculative_config"]:
        args += ["--speculative-config", json.dumps(c["speculative_config"])]
    if c["enforce_eager"]:
        args += ["--enforce-eager"]
    else:
        args += ["--compilation-config", json.dumps(c["compilation_config"])]
    if rank:
        args += ["--headless"]
    return args


def owned(model, info, rank):
    labels = info["Config"].get("Labels") or {}
    return (labels.get(LABEL + ".model") == model["id"]
            and labels.get(LABEL + ".fingerprint") == fingerprint(model)
            and labels.get(LABEL + ".rank") == str(rank)
            and info["Image"] == model["glm53_main"]["image_id"])


def run(model, action):
    rank = stable.node(adapter(model))["rank"]
    if action == "dry-run":
        print(shlex.join(render_command(model, rank)))
        return
    if action == "stop":
        names = stable.output(["docker", "ps", "--format", "{{.Names}}", "--filter",
                               "label=" + LABEL + ".model=" + model["id"], "--filter",
                               "label=" + LABEL + ".rank=" + str(rank)]).splitlines()
        if len(names) > 1:
            raise ValueError("Multiple experimental ranks found; inspect ownership")
        for name in names:
            stable.execute(["docker", "stop", "--time", "60", name])
        return
    name = container_name(model, rank)
    info = stable.inspect(name)
    if info and not owned(model, info, rank):
        raise ValueError("Container ownership mismatch")
    validate_ready(model)
    if info and info["State"]["Running"]:
        raise ValueError("Experimental container already running")
    stable.preflight(adapter(model), rank)
    if info is None:
        stable.execute(render_command(model, rank))
    os.execvp("docker", ["docker", "start", "--attach", name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "run", "stop", "dry-run"))
    parser.add_argument("model")
    args = parser.parse_args()
    model = load(args.model)
    if args.action == "setup":
        setup(model)
    else:
        run(model, args.action)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"GLM experimental: {exc}", file=sys.stderr)
        sys.exit(1)
