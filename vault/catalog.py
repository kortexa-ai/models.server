#!/usr/bin/env python3
"""Export one compact Markdown row per model, combining its archive formats."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


GROUPS = ["Language models", "Gemma 4", "LiquidAI", "Modified language models",
          "Embeddings", "Speech", "Music and audio", "Image and video", "Vision"]
ALIASES = {
    "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit": "prism-ml/Ternary-Bonsai-2-27B-gguf",
    "zai-org/GLM-5.3-BF16": "zai-org/GLM-5.3",
    "nvidia/Gemma-4-26B-A4B-NVFP4": "google/gemma-4-26B-A4B-it",
    "rockerBOO/minimax-h3-nvfp4-convrot": "MiniMaxAI/MiniMax-H3",
    "UrocyonF/Qwen3-TTS-12Hz-1.7B-NVFP4": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "m-a-p/YuE2-Vae": "m-a-p/YuE2-3B",
    "m-a-p/YuE2-Vae-legacy": "m-a-p/YuE2-3B",
    "BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF":
        "BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-GGUF",
}
LABELS = {
    "Qwen/Qwen3.8-2.4T-A95B": "Qwen3.8-Max 2.4T",
    "meta-models/Muse-Glimmer-30B": "Muse Glimmer 30B",
    "huihui-ai/Huihui-Qwen3.8-27B-abliterated": "Qwen3.8-27B abliterated",
    "orcarouter/Qwen3.8-27B-Uncensored-FP8": "Qwen3.8-27B uncensored",
    "dealignai/GLM-5.3-Flash-UNCENSORED-FP8": "GLM-5.3-Flash uncensored",
    "audreyt/DeepSeek-V4.1-Flash-Abliterated-GGUF": "DeepSeek-V4.1-Flash abliterated",
    "BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-GGUF": "Bonsai 2 27B abliterated",
    "openai/whisper-large-v3": "Whisper large-v3",
    "hexgrad/Kokoro-82M": "Kokoro-82M",
    "m-a-p/YuE2-3B": "YuE2-3B",
}
FORMATS = {
    "moonshotai/Kimi-K3": "MXFP4",
    "deepseek-ai/DeepSeek-V4.1-Flash": "FP4/FP8",
    "zai-org/GLM-5.3": "FP8",
    "zai-org/GLM-5.3-Flash": "FP8",
    "Qwen/Qwen3.8-27B": "BF16",
    "Qwen/Qwen3.8-2.4T-A95B": "BF16",
    "zai-org/GLM-5.3-BF16": "BF16",
    "meta-models/Muse-Glimmer-30B": "BF16",
    "huihui-ai/Huihui-Qwen3.8-27B-abliterated": "BF16",
    "prism-ml/Ternary-Bonsai-2-27B-gguf": "F16, ternary GGUF",
    "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit": "ternary MLX",
    "BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PTQ1_0-GGUF": "PTQ1_0 GGUF",
    "BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF": "PQ2_0 GGUF, MTP",
    "audreyt/DeepSeek-V4.1-Flash-Abliterated-GGUF": "Q2 GGUF, Q2 imatrix",
    "Lightricks/LTX-2.5": "BF16, NVFP4",
    "m-a-p/YuE2-Vae": "VAE",
    "m-a-p/YuE2-Vae-legacy": "legacy VAE",
}


def normalized(name):
    name = name.split("/", 1)[1].lower().removeprefix("openai_")
    return re.sub(r"-(?:nvfp4(?:-fp8-mixed)?|fp8)$", "", name)


def render(vault):
    raw = (vault / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    state = json.loads((vault / "state.json").read_text())
    if state["manifest_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("State belongs to a different manifest")
    source_path = vault / "source-inventory.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else {"original_groups": []}
    labels, groups = dict(LABELS), {}
    for group in source["original_groups"]:
        for repo in group["repos"]:
            name = repo["repo"]
            labels.setdefault(name, group["label"] if len(group["repos"]) == 1 else name.split("/", 1)[1])
            groups[name] = group["group"]
    active = manifest["repos"]
    deferred = manifest.get("deferred_repos", [])
    originals = {normalized(r["repo"]): r["repo"] for r in active + deferred if "original" in r["categories"]}
    rows = defaultdict(list)
    for repo in active + deferred:
        name = repo["repo"]
        key = ALIASES.get(name, originals.get(normalized(name), name))
        rows[key].append(repo)
        if repo.get("group") and repo["group"] != "NVFP4 copies":
            groups.setdefault(key, repo["group"])
    deferred_names = {r["repo"] for r in deferred}
    lines = ["| Status | Model | Size (GB, active) | Notes | Repositories |",
             "| :---: | --- | ---: | --- | --- |"]
    def sort_key(key):
        group = groups.get(key, "")
        return (GROUPS.index(group) if group in GROUPS else len(GROUPS), labels.get(key, key).casefold())
    for key in sorted(rows, key=sort_key):
        repos = sorted(rows[key], key=lambda r: (r["repo"] in deferred_names, "original" not in r["categories"], r["repo"]))
        complete = all(r["repo"] not in deferred_names and
                       len(state["repos"].get(r["repo"], {}).get("done", {})) == len(r["files"]) for r in repos)
        notes, links = [], []
        for repo in repos:
            name = repo["repo"]
            fmt = FORMATS.get(name)
            if fmt is None:
                fmt = ("NVFP4/FP8" if "nvfp4-fp8-mixed" in name.lower() else
                       "NVFP4" if "nvfp4" in repo["categories"] else
                       "FP8" if name.lower().endswith("-fp8") else "Original")
                if name.startswith("mlx-community/"):
                    fmt += " MLX"
            if name in deferred_names:
                fmt += " deferred"
            for note in fmt.split(", "):
                if note not in notes:
                    notes.append(note)
            links.append(f"[{name}](https://huggingface.co/{name}/tree/{repo['revision']})")
            if name == "facebook/sam3":
                notes.append("ModelScope weights")
                links.append("[ModelScope/facebook/sam3](https://modelscope.cn/models/facebook/sam3)")
        size = sum(r["bytes"] for r in repos if r["repo"] not in deferred_names) / 10**9
        label = labels.get(key, key.split("/", 1)[1])
        lines.append(f"| {'✅' if complete else '🕒'} | {label} | {size:,.2f} | {', '.join(notes)} | {'<br>'.join(links)} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=Path.home() / "storage/models/vault")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.vault))
    print(args.output)
