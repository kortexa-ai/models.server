"""Build the experimental image explicitly; never called during service start."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parent
EXPECTED_TRANSPORT = "sha256:881716c0f7d0904d3aa2c4f138a15e916c4c1aaca48070b39785032ed485863a"
TRANSPORT_TAG = "local/glm53-tp2:770d115-20261009"


def main():
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise SystemExit("Build on the registered ARM64 Spark pair")
    actual = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", TRANSPORT_TAG],
        text=True).strip()
    if actual != EXPECTED_TRANSPORT:
        raise SystemExit("Preserved transport image does not match its immutable pin")
    available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < 16 * 1024 ** 2:
        raise SystemExit("Build needs 16 GiB available; coordinate the pair before stopping the model")
    digest = hashlib.sha256()
    for path in sorted(ROOT.iterdir()):
        if path.is_file():
            digest.update(path.name.encode() + b"\0" + path.read_bytes())
    tag = "local/glm53-main:" + digest.hexdigest()[:16]
    subprocess.run(["docker", "build", "--progress=plain", "--tag", tag, str(ROOT)], check=True)
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", tag]))[0]
    receipt = {"build_inputs_sha256": digest.hexdigest(), "tag": tag,
               "image_id": image["Id"], "labels": image["Config"].get("Labels", {})}
    output = ROOT.parents[1] / ".engines/glm53-main/build-receipts"
    output.mkdir(parents=True, exist_ok=True)
    (output / (digest.hexdigest() + ".json")).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
