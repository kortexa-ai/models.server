#!/usr/bin/env python3
import argparse
import json
import re
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "models.json"


def load_catalog():
    payload = json.loads(CATALOG_PATH.read_text())
    return payload["models"]


def alias_index():
    index = {}
    for model in load_catalog():
        for alias in [model["key"], *model.get("aliases", [])]:
            index[alias.lower()] = model
    return index


def sanitize_model_id(model_path):
    tail = model_path.rsplit("/", 1)[-1]
    normalized = re.sub(r"[^a-z0-9]+", "-", tail.lower()).strip("-")
    return normalized or "model"


def choose_sglang_profile(model, profile_name=None):
    profiles = model.get("sglang_profiles", {})
    resolved = profile_name or model.get("default_sglang_profile")
    if resolved not in profiles:
        choices = ", ".join(sorted(profiles))
        raise SystemExit(f"Unknown SGLang profile '{resolved}' for model '{model['key']}'. Choices: {choices}")
    merged = dict(model)
    merged["selected_sglang_profile"] = resolved
    merged["sglang_model_path"] = profiles[resolved]["model_path"]
    merged["quantization"] = profiles[resolved].get("quantization")
    return merged


def resolve_model(name, profile_name=None):
    index = alias_index()
    lowered = name.lower()
    if lowered in index:
        return choose_sglang_profile(index[lowered], profile_name)
    if "/" in name:
        model_id = sanitize_model_id(name)
        quantization = "modelopt_fp4" if name.upper().endswith("-NVFP4") else None
        return {
            "key": model_id,
            "display_name": name,
            "model_id": model_id,
            "selected_sglang_profile": profile_name or "custom",
            "sglang_model_path": name,
            "vllm_model_path": name,
            "quantization": quantization,
            "sglang_port": 2231,
            "vllm_port": 2241,
            "mlx_port": 2031,
            "llama_port": 2031,
            "context_length_default": 32768,
            "tp": 1,
            "reasoning_parser": "qwen3",
        }
    choices = ", ".join(model["key"] for model in load_catalog())
    raise SystemExit(f"Unknown model '{name}'. Known presets: {choices}")


def to_shell_vars(model):
    payload = {
        "MODEL_KEY": model["key"],
        "DISPLAY_NAME": model["display_name"],
        "MODEL_ID": model["model_id"],
        "SGLANG_PROFILE": model.get("selected_sglang_profile", ""),
        "SGLANG_MODEL_PATH": model.get("sglang_model_path", ""),
        "VLLM_MODEL_PATH": model.get("vllm_model_path", ""),
        "MLX_MODEL_PATH": model.get("mlx_model_path", ""),
        "MLX_PORT": model.get("mlx_port", ""),
        "LLAMA_GGUF_REPO": model.get("llama_gguf_repo", ""),
        "LLAMA_PORT": model.get("llama_port", ""),
        "QUANTIZATION_DEFAULT": model.get("quantization"),
        "SGLANG_PORT": model.get("sglang_port", ""),
        "VLLM_PORT": model.get("vllm_port", ""),
        "CONTEXT_LENGTH_DEFAULT": model.get("context_length_default", 32768),
        "TP_DEFAULT": model.get("tp", 1),
        "REASONING_PARSER_DEFAULT": model.get("reasoning_parser", "qwen3"),
    }
    lines = []
    for key, value in payload.items():
        lines.append(f"{key}={shlex.quote(str(value or ''))}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Resolve a model preset from the bench catalog.")
    parser.add_argument("model", nargs="?", help="Preset key like 0.8b or a direct Hugging Face repo ID.")
    parser.add_argument("--profile", choices=("standard", "nvfp4"), help="SGLang profile for known presets.")
    parser.add_argument("--format", choices=("json", "shell"), default="json", help="Output format.")
    parser.add_argument("--list", action="store_true", help="List known presets and exit.")
    args = parser.parse_args()

    if args.list:
        for model in load_catalog():
            aliases = ", ".join(model.get("aliases", []))
            profiles = ", ".join(sorted(model.get("sglang_profiles", {})))
            print(f"{model['key']:>4}  profiles: {profiles}  aliases: {aliases}")
        return

    if not args.model:
        parser.error("the following arguments are required: model")

    model = resolve_model(args.model, args.profile)
    if args.format == "shell":
        print(to_shell_vars(model))
        return
    print(json.dumps(model, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
