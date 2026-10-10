#!/usr/bin/env python3
"""Pinned two-node GLM lifecycle. Setup prepares; run never builds or downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
LABEL = "ai.kortexa.models-server"


def execute(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def output(args, **kwargs):
    return execute(args, capture_output=True, **kwargs).stdout.strip()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    path = Path(path)
    if path.is_dir():
        path /= "model.json"
    model = json.loads(path.read_text())
    config = model["glm53_tp2"]
    if model["default_engine"] != "vllm-glm53-tp2":
        raise ValueError("Wrong engine")
    if sorted(n["rank"] for n in config["nodes"].values()) != [0, 1]:
        raise ValueError("Exactly two ranks required")
    if config["environment"]["MAX_MODEL_LEN"] != model["context"]:
        raise ValueError("Context fields disagree")
    if config["environment"]["MAX_SEQS"] != model["parallel"]:
        raise ValueError("Parallel fields disagree")
    return model


def fingerprint(model):
    payload = json.dumps(model, sort_keys=True).encode() + Path(__file__).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def locations(model):
    config = model["glm53_tp2"]
    base = ROOT / ".engines" / "glm53-tp2" / model["id"]
    # Revision-specific source trees never mutate an existing container's mounts.
    return base, base / config["runtime_revision"], base / config["recipe_revision"]


def node(model):
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("This recipe requires Linux aarch64 on the registered Spark pair")
    name = socket.gethostname().split(".")[0]
    if name not in model["glm53_tp2"]["nodes"]:
        raise ValueError("Host is not in this recipe's node inventory")
    return model["glm53_tp2"]["nodes"][name]


def checkout(path, url, revision, setup=False):
    if not path.exists() and setup:
        execute(["git", "clone", "--no-checkout", url, str(path)])
        execute(["git", "-C", str(path), "checkout", "--detach", revision])
    if output(["git", "-C", str(path), "rev-parse", "HEAD"]) != revision:
        raise ValueError(f"Wrong source revision: {path}")
    if output(["git", "-C", str(path), "remote", "get-url", "origin"]) != url:
        raise ValueError(f"Wrong source origin: {path}")
    execute(["git", "-C", str(path), "diff", "--quiet", "HEAD", "--"])
    if output(["git", "-C", str(path), "ls-files", "--others", "--exclude-standard"]):
        raise ValueError(f"Untracked files in pinned source: {path}")


def file_state(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def artifact_files(model):
    c = model["glm53_tp2"]
    for kind, field in [("target", "model_path"), ("draft", "draft_path")]:
        root = Path(c[field])
        for name, sha in c[kind + "_sha256"].items():
            if Path(name).name != name:
                raise ValueError("Artifact names must be plain filenames")
            yield root / name, sha


def adopt(source, target):
    """Local hardlinks only: never relay weights or overwrite an artifact."""
    source, target = Path(source), Path(target)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, copy_function=os.link)


def setup(model, args):
    node(model)
    c = model["glm53_tp2"]
    base, source, recipe = locations(model)
    base.mkdir(parents=True, exist_ok=True)
    checkout(source, c["runtime_url"], c["runtime_revision"], setup=True)
    checkout(recipe, c["recipe_url"], c["recipe_revision"], setup=True)
    if args.adopt_target:
        adopt(args.adopt_target, c["model_path"])
    if args.adopt_draft:
        adopt(args.adopt_draft, c["draft_path"])
    image = json.loads(output(["docker", "image", "inspect", c["image_id"]]))[0]
    if image["Id"] != c["image_id"]:
        raise ValueError("Pinned image is missing")
    states = {}
    for path, expected in artifact_files(model):
        before = file_state(path)
        if digest(path) != expected or file_state(path) != before:
            raise ValueError(f"Artifact checksum mismatch or concurrent change: {path}")
        states[str(path)] = before
        print(f"Verified {path.name}", flush=True)
    index = json.loads((Path(c["model_path"]) / "model.safetensors.index.json").read_text())
    if not set(index["weight_map"].values()) <= set(c["target_sha256"]):
        raise ValueError("Index references an unpinned shard")
    Path(c["cache_path"]).mkdir(parents=True, exist_ok=True)
    receipt = {"fingerprint": fingerprint(model), "files": states}
    with tempfile.NamedTemporaryFile("w", dir=base, delete=False) as stream:
        json.dump(receipt, stream, indent=2)
        tmp = stream.name
    os.replace(tmp, base / "receipt.json")
    print("Setup verified; no service started.")


def validate_ready(model):
    c = model["glm53_tp2"]
    base, source, recipe = locations(model)
    receipt = json.loads((base / "receipt.json").read_text())
    if receipt["fingerprint"] != fingerprint(model):
        raise ValueError("Recipe changed: run explicit setup again")
    checkout(source, c["runtime_url"], c["runtime_revision"])
    checkout(recipe, c["recipe_url"], c["recipe_revision"])
    for path, _ in artifact_files(model):
        if receipt["files"].get(str(path)) != file_state(path):
            raise ValueError(f"Artifact changed since setup: {path}")
    if output(["docker", "image", "inspect", "--format", "{{.Id}}", c["image_id"]]) != c["image_id"]:
        raise ValueError("Wrong image")


def shell_value(value):
    return shlex.quote(json.dumps(value) if isinstance(value, (list, dict)) else str(value))


def environment(model, source):
    c = model["glm53_tp2"]
    lines = [f"{k}={shell_value(v)}" for k, v in c["environment_before_profile"].items()]
    lines.append("source " + shlex.quote(str(source / c["profile"])))
    env = dict(c["environment"])
    env.update(HOSTS="local local", IPS=" ".join(n["ip"] for n in sorted(c["nodes"].values(), key=lambda n: n["rank"])),
               IMAGE=c["image_id"], MODEL_DIR=c["model_path"], DRAFT_DIR=c["draft_path"],
               OVERLAY_REMOTE=str(source), CTN=model["id"] + "-" + fingerprint(model)[:12],
               SERVED_NAME=model["id"], HOST_BIND=model["host"], PORT=model["port"])
    lines.extend(f"{k}={shell_value(v)}" for k, v in env.items())
    # Replace these keys instead of relying on duplicate Docker -e semantics.
    extra = " ".join(f"{k}={v}" for k, v in c["extra_environment"].items())
    pattern = "|".join(k + "=*" for k in c["extra_environment"])
    lines.append('FILTERED_ENV=""; for kv in $EXTRA_ENV; do case "$kv" in '
                 + pattern + ') ;; *) FILTERED_ENV="$FILTERED_ENV $kv" ;; esac; done')
    lines.append('EXTRA_ENV="$FILTERED_ENV ' + extra + '"')
    return "\n".join(lines) + "\n"


def container_name(model, rank):
    return model["id"] + "-" + fingerprint(model)[:12] + f"-r{rank}"


def render_command(model, rank):
    """Render the pinned upstream run_rank without executing its cluster lifecycle."""
    _, source, recipe = locations(model)
    c = model["glm53_tp2"]
    script = (recipe / c["launcher"]).read_text()
    marker = "\ncase $CMD in\n"
    if script.count(marker) != 1:
        raise ValueError("Pinned launcher structure changed")
    # Drop its entry point (including rsync/stop/start), retain pinned command construction.
    script = script.split(marker)[0]
    script = script.replace('cd "$(dirname "$0")"', "cd " + shlex.quote(str(source)), 1)
    # Bash 3 (local command review on macOS) expands one local statement at once.
    script = script.replace('local r=$1 h=${HOSTS[$r]} ip=${IPS[$r]}',
                            'local r=$1; local h=${HOSTS[$r]} ip=${IPS[$r]}')
    script += '\nrssh() { printf "%s\\n" "${@:2}"; }\nrun_rank "$1"\n'
    with tempfile.TemporaryDirectory() as tmp:
        env_file = Path(tmp) / "runtime.env"
        env_file.write_text(environment(model, source))
        driver = Path(tmp) / "render.sh"
        driver.write_text(script)
        # Do not let ambient developer overrides alter the pinned launch.
        env = {"PATH": os.environ["PATH"], "HOME": str(Path.home()), "ENV_FILE": str(env_file)}
        command = output(["bash", str(driver), str(rank)], env=env)
    argv = shlex.split(command.replace("\\\n", ""))
    if argv[:3] != ["docker", "run", "-d"]:
        raise ValueError("Unexpected pinned launcher command")
    argv[1:3] = ["create"]
    gpu = next(n["gpu_uuid"] for n in c["nodes"].values() if n["rank"] == rank)
    argv[argv.index("--gpus") + 1] = "device=" + gpu
    for i, arg in enumerate(argv):
        if arg == str(source) + "/cache:/cache":
            argv[i] = c["cache_path"] + ":/cache"
    argv[2:2] = ["--label", LABEL + ".model=" + model["id"],
                 "--label", LABEL + ".fingerprint=" + fingerprint(model),
                 "--label", LABEL + ".rank=" + str(rank)]
    return argv


def inspect(name):
    result = subprocess.run(["docker", "inspect", name], text=True, capture_output=True)
    if result.returncode:
        # Distinguish an absent container from a dead Docker daemon.
        execute(["docker", "info", "--format", "{{.ServerVersion}}"], stdout=subprocess.DEVNULL)
        return None
    return json.loads(result.stdout)[0]


def owned(model, info, rank):
    labels = info["Config"].get("Labels") or {}
    return (labels.get(LABEL + ".model") == model["id"]
            and labels.get(LABEL + ".fingerprint") == fingerprint(model)
            and labels.get(LABEL + ".rank") == str(rank)
            and info["Image"] == model["glm53_tp2"]["image_id"])


def preflight(model, rank):
    c = model["glm53_tp2"]
    gpu = next(n["gpu_uuid"] for n in c["nodes"].values() if n["rank"] == rank)
    if output(["nvidia-smi", "--id=" + gpu, "--query-gpu=uuid", "--format=csv,noheader"]) != gpu:
        raise ValueError("GPU identity mismatch")
    processes = output(["nvidia-smi", "--id=" + gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    if processes:
        raise ValueError("GPU has compute processes; inspect and stop the owning service first")
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].split()[0]) * 1024
    if available < c["startup_free_gib"] * 1024 ** 3:
        raise ValueError("Insufficient available memory for the pinned startup budget")
    ports = [model["port"], c["environment"]["MASTER_PORT"]] if rank == 0 else []
    for port in ports:
        with socket.socket() as probe:
            probe.bind((model["host"], port))


def run(model, stop=False, dry=False):
    rank = node(model)["rank"]
    name = container_name(model, rank)
    if dry:
        print(shlex.join(render_command(model, rank)))
        return
    if stop:
        # A later manifest/source edit must still be able to stop its old owned rank.
        names = output(["docker", "ps", "--format", "{{.Names}}", "--filter",
                        "label=" + LABEL + ".model=" + model["id"], "--filter",
                        "label=" + LABEL + ".rank=" + str(rank)]).splitlines()
        if len(names) > 1:
            raise ValueError("Multiple owned ranks found; inspect them before stopping")
        for existing in names:
            execute(["docker", "stop", "--time", "60", existing])
        return
    info = inspect(name)
    if info and not owned(model, info, rank):
        raise ValueError("Container ownership mismatch; refusing to change it")
    validate_ready(model)
    if info and info["State"]["Running"]:
        raise ValueError("Container already running; refusing a duplicate supervisor")
    preflight(model, rank)
    if info is None:
        execute(render_command(model, rank))
    # Attached Docker remains systemd's foreground process. ExecStop stops only this rank.
    os.execvp("docker", ["docker", "start", "--attach", name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "run", "stop", "dry-run"])
    parser.add_argument("model", nargs="?", default=str(ROOT / "glm-5.3-flash-2x-dgx"))
    parser.add_argument("--adopt-target")
    parser.add_argument("--adopt-draft")
    args = parser.parse_args()
    model = load(args.model)
    if args.action == "setup":
        setup(model, args)
    else:
        run(model, stop=args.action == "stop", dry=args.action == "dry-run")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"GLM TP2: {exc}", file=sys.stderr)
        sys.exit(1)
