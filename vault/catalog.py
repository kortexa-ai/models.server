#!/usr/bin/env python3
"""Export the exact vault queue as a readable Markdown repository catalog."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path


GROUPS = ["Language models", "Gemma 4", "LiquidAI", "Modified language models",
          "Embeddings", "Speech", "Music and audio", "Image and video", "Vision", "NVFP4 copies"]
SPECIAL_NOTES = {
    "Lightricks/LTX-2.5": "Selected original BF16 dev/distilled transformers, text encoder, VAEs, upscalers and LoRA, plus the distilled NVFP4 transformer. INT8 files and other duplicate representations are excluded.",
    "MiniMaxAI/MiniMax-H3": "Only the original FL2VA/ and Ref2VA/ bundles. Duplicate root Diffusers layout is excluded.",
    "rockerBOO/minimax-h3-nvfp4-convrot": "Only the two unpruned NVFP4 transformer files (FL2VA and Ref2VA) and small metadata. Use the encoders/VAEs from the archived original release.",
    "mlx-community/MiniMax-Music3-nvfp4": "MLX-format NVFP4 archival copy; not validated as a DGX Spark serving package.",
    "mlx-community/sam3-nvfp4": "MLX-format NVFP4 archival copy; not validated as a DGX Spark serving package.",
    "DKmode22/YuE2-3B-NVFP4": "NVFP4 AR weights only. The original NAR and VAE components are separately included.",
    "UrocyonF/Qwen3-TTS-12Hz-1.7B-NVFP4": "NVFP4 CustomVoice variant; other original TTS variants are listed separately.",
    "prism-ml/Ternary-Bonsai-2-27B-gguf": "Full official GGUF release: F16 reference, both ternary packs (PTQ1_0/PQ2_0), and both vision projectors.",
    "zai-org/GLM-5.3": "Official FP8 release (E4M3 quantization). See the separate BF16 precision check below.",
}


def read_optional(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def render(vault):
    raw = (vault / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    state = json.loads((vault / "state.json").read_text())
    if state["manifest_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("State belongs to a different manifest")
    source = read_optional(vault / "source-inventory.json", {"original_groups": []})
    progress = read_optional(vault / "status.json", {})
    capacity = read_optional(vault / "capacity.json", {})
    precision = read_optional(vault / "glm-precision-check.json", {})
    labels, groups, notes = {}, {}, {}
    for group in source["original_groups"]:
        for repo in group["repos"]:
            name = repo["repo"]
            groups[name] = group["group"]
            labels[name] = group["label"] if len(group["repos"]) == 1 else name.split("/", 1)[1]
            notes[name] = group.get("note", "")
    by_group = defaultdict(list)
    for repo in manifest["repos"]:
        group = repo.get("group", groups.get(repo["repo"], "NVFP4 copies"))
        by_group[group].append(repo)
    verified = sum(f["bytes"] for rs in state["repos"].values() for f in rs["done"].values())
    complete = sum(len(state["repos"].get(r["repo"], {}).get("done", {})) == len(r["files"])
                   for r in manifest["repos"])
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = ["# Model vault — complete download list", "", f"Generated: **{stamp}**.", "",
             "Target: `smarty:~/storage/models/vault` (`/mnt/storage/models/vault`).",
             "This catalog includes completed downloads and every repository still queued. Status is a snapshot.", "",
             f"- **Repositories: {len(manifest['repos'])}.** Each is listed once below.",
             f"- **Selected files: {sum(len(r['files']) for r in manifest['repos']):,}.**",
             f"- **Total selected bytes: {manifest['total_bytes']:,} — {manifest['total_bytes'] / 10**12:.3f} TB.**",
             f"- Verified at export: **{verified / 10**12:.3f} TB**, with **{complete} repositories complete**.",
             "- Sizes use decimal GB/TB and include selected metadata, alternate formats and bundled dependencies.",
             "- Original and NVFP4 copies are separate archives. No cross-repository deduplication is assumed.",
             f"- Manifest SHA-256: `{hashlib.sha256(raw).hexdigest()}`.", ""]
    if capacity:
        lines += ["## Storage capacity", "",
                  f"At the last queue update, the disk had **{capacity['free_bytes'] / 10**12:.3f} TB free**.",
                  f"The downloader retains a **{capacity['reserve_bytes'] / 1024**3:.0f} GiB reserve**.",
                  f"The remaining queue plus that reserve exceeds current free space by approximately **{capacity['approx_shortfall_bytes'] / 10**12:.3f} TB**.",
                  "All selections remain queued. More capacity is required to finish; the runtime disk guard stays enabled.",
                  "This estimate excludes credit for partial downloads and can change with other disk use.", ""]
    lines += ["## Sources and selection rules", "",
              "The repository link names the exact source. The revision link opens the frozen commit used by the queue.",
              "The selected file count and byte total come from the active manifest, not a parameter-count estimate.",
              "Some original repositories contain several framework formats or quant alternatives; their selected copies are all counted.",
              "LFM selections are restricted to the 2.5 family. Gemma thinking and non-thinking modes use the same selected IT weights.",
              "SAM 3 weights come from ModelScope; its original small metadata files come from Hugging Face. See its entry for both sources.", ""]
    for group in GROUPS + sorted(set(by_group) - set(GROUPS)):
        if group not in by_group:
            continue
        repos = sorted(by_group[group], key=lambda r: r["repo"].casefold())
        lines += [f"## {group}", "",
                  f"{len(repos)} repositories; **{sum(r['bytes'] for r in repos) / 10**9:,.3f} GB** selected.", ""]
        for repo in repos:
            name, revision = repo["repo"], repo["revision"]
            label = repo.get("label", labels.get(name, name.split("/", 1)[1]))
            rs = state["repos"].get(name, {})
            done = len(rs.get("done", {}))
            if done == len(repo["files"]):
                status = "Complete; all selected files verified"
            elif (progress.get("current") or {}).get("repo") == name:
                status = f"{progress.get('phase', 'Active')}; {done}/{len(repo['files'])} files verified"
            elif rs.get("error"):
                status = f"Queued for retry ({rs['error']}); {done}/{len(repo['files'])} files verified"
            else:
                status = f"Queued; {done}/{len(repo['files'])} files verified"
            note = repo.get("note") or SPECIAL_NOTES.get(name) or notes.get(name)
            scope = repo.get("scope", "Exact selected files in the pinned manifest")
            lines += [f"### {label}", "",
                      f"- Repository: [{name}](https://huggingface.co/{name}).",
                      f"- Revision: [`{revision}`](https://huggingface.co/{name}/tree/{revision}).",
                      f"- Selected: **{repo['bytes'] / 10**9:,.3f} GB** ({repo['bytes']:,} bytes), **{len(repo['files'])} files**.",
                      f"- Archive type: {', '.join(repo['categories'])}. {scope}.",
                      f"- Status: {status}."]
            if note:
                lines.append(f"- Scope note: {note}")
            if name == "facebook/sam3":
                lines += ["- Weight download source: [ModelScope facebook/sam3](https://modelscope.cn/models/facebook/sam3).",
                          "- ModelScope revision: `f8ee5d16d5c30d0e87b1d22cbd358ce97761df56`.",
                          "- Mirror files: `sam3.pt` and `model.safetensors`, checked against original HF SHA-256 values."]
            lines.append("")
    if precision:
        lines += ["## GLM-5.3 BF16 check", "",
                  "The BF16 release is real and is a separate official repository. The existing GLM-5.3 archive is the FP8 release.",
                  "Its config declares BF16 as the general dtype but also enables FP8 E4M3 weight quantization; the BF16 repository has no quantization config.", ""]
        queued = {r["repo"] for r in manifest["repos"]}
        for name, info in precision.items():
            revision = info["revision"]
            state_label = "Included in this queue" if name in queued else "Documented only; not added to the queue"
            lines += [f"- [{name}](https://huggingface.co/{name}): **{info['bytes'] / 10**9:,.3f} GB** ({info['bytes']:,} bytes). {state_label}.",
                      f"  Verified revision: [`{revision}`](https://huggingface.co/{name}/blob/{revision}/config.json)."]
        lines.append("")
    lines += ["## Progress and control", "", "From Snappy:", "", "```bash",
              "ssh smarty 'python3 ~/src/models.server/vault/run.py status'", "```", "",
              "On Smarty:", "", "```bash", "ktxsvc stop models/vault   # pause; partial downloads remain",
              "ktxsvc start models/vault  # resume",
              "journalctl -u kortexa-ai-model-vault.service -n 30 --no-pager", "```", "",
              "The service is enabled at boot. It resumes partial files, verifies checksums, and retries failures indefinitely.",
              "The exact file manifest, verification receipts and selection-update history are stored in the vault.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=Path.home() / "storage/models/vault")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.vault))
    print(args.output)
