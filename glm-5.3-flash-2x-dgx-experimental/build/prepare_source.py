"""Check native-wheel compatibility and enable the verified mixed checkpoint."""
from pathlib import Path
import json
import subprocess
import sys


def replace_once(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"Pinned source changed: {path}, expected one match")
    path.write_text(text.replace(old, new))


def prepare(root, source_revision, native_revision):
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != source_revision:
        raise RuntimeError("Wrong source revision")
    changed = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-only", native_revision,
         source_revision, "--"], text=True).splitlines()
    native_paths = ("csrc/", "cmake/", "CMakeLists.txt", "pyproject.toml",
                    "vllm/_custom_ops.py", "requirements/", "setup.py")
    if any(name.startswith(native_paths) for name in changed):
        raise RuntimeError("Native inputs differ: build a matching native wheel")
    # The intervening Rust change removes a deprecated HTTP argument. This
    # recipe uses the Python API server, not the optional compiled Rust server.
    manifest = {"vllm_revision": source_revision, "native_revision": native_revision,
                "delta_files": changed, "frontend": "python"}
    model = root / "vllm/models/glm5next/common/model.py"
    replace_once(model,
                 "quant_config=None,  # MLA projections are BF16 in checkpoint",
                 "quant_config=quant_config,  # Mixed checkpoint supplies projection scales")
    kda = root / "vllm/models/glm5next/common/kda.py"
    replace_once(kda,
                 "# KDA projections remain BF16 because fp8 checkpoints omit their scales.",
                 "# Ordinary FP8 checkpoints omit KDA scales; our mixed checkpoint has them.")
    replace_once(kda,
                 "            vllm_config.quant_config = None\n",
                 "            vllm_config.quant_config = (\n"
                 "                saved_quant_config\n"
                 "                if type(saved_quant_config).__name__ == 'ModelOptMixedPrecisionConfig'\n"
                 "                else None\n"
                 "            )\n")
    Path("/opt/glm53-main-build.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    prepare(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
