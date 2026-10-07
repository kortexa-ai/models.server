#!/usr/bin/env python3
"""Consolidate installed harness configs; run on Snappy with Hermes's Python.

Default is a credential-free preview. --apply backs up exact affected private
files on their owning machine, checks for concurrent edits, and writes 0600
files atomically. SSH carries private contents only on stdin/stdout; logs show
paths, provider/model names and backup locations, never credential values.
"""
import argparse
import copy
import datetime
import hashlib
import io
import json
import os
import pathlib
import shlex
import subprocess
import urllib.request

from ruamel.yaml import YAML

PUBLIC = "kortexa.ai"
STATIC = "kortexa-static"
PUBLIC_URL = "https://api.kortexa.ai/v1"
STATIC_URL = "http://192.168.2.101:2067/v1"
FLASH = "qwen-3.8-flash-next-fast"
QWEN = "qwen-3.8-27b"
BONSAI = "bonsai-2-27b"
CODEX_MODEL = {
    "id": "gpt-6-luna", "name": "GPT-6-Luna", "api": "openai-codex-responses",
    "contextWindow": 272000, "maxTokens": 128000, "reasoning": True, "input": ["text", "image"],
    "cost": {"input": 0.1, "output": 0.5, "cacheRead": 0.01, "cacheWrite": 0.125},
}
yaml = YAML()
yaml.preserve_quotes = True
yaml.width = 120

# These helpers operate only on private harness files; repository files travel through Git.
READ = '''import json,pathlib
p=pathlib.Path.home()
names=['.hermes/config.yaml','.omp/agent/models.yml','.omp/agent/config.yml','.pi/agent/models.json','.pi/agent/settings.json']
names += [str(f.relative_to(p)) for f in (p/'.hermes/profiles').glob('*/config.yaml')]
print(json.dumps({n:(p/n).read_text() if (p/n).is_file() else None for n in names}))
'''
WRITE = '''import json,pathlib,hashlib,sys,os,tempfile
data=json.load(sys.stdin); root=pathlib.Path.home()
for n,item in data['files'].items():
 f=root/n
 if f.is_symlink() or not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=item['before']:
  raise SystemExit('Concurrent change or unsafe target: '+n)
backup=root/'.local/share/kortexa/provider-migration'/data['stamp']
backup.mkdir(parents=True,exist_ok=False,mode=0o700)
for n,item in data['files'].items():
 f=root/n; b=backup/n
 b.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with b.open('xb') as out: out.write(f.read_bytes())
 os.chmod(b,0o600)
 with tempfile.NamedTemporaryFile(dir=f.parent,delete=False) as out:
  out.write(item['after'].encode()); out.flush(); os.fsync(out.fileno()); tmp=out.name
 os.chmod(tmp,0o600); os.replace(tmp,f)
print(json.dumps({'backup':str(backup),'files':list(data['files'])}))
'''


def remote(host, code, payload=None):
    if host == "snappy":
        command = ["python3", "-c", code]
    else:
        command = ["ssh", "-o", "BatchMode=yes", host, "python3 -c " + shlex.quote(code)]
    result = subprocess.run(command, input=payload, text=True, capture_output=True)
    if result.returncode:
        # Exceptions from private config parsing can include values; report only the host.
        raise RuntimeError(f"Private configuration operation failed on {host}")
    return json.loads(result.stdout)


def parsed(name, text):
    return json.loads(text) if name.endswith(".json") else yaml.load(text)


def serialize(name, data):
    if name.endswith(".json"):
        return json.dumps(data, indent=2) + "\n"
    output = io.StringIO()
    yaml.dump(data, output)
    return output.getvalue()


def model_entry(model_id, catalog):
    info = catalog.get(model_id, {})
    caps = info.get("capabilities", {})
    entry = {
        "id": model_id, "name": model_id,
        "contextWindow": info.get("context_window", 262144), "maxTokens": 32768,
        "input": [m for m in info.get("input_modalities", ["text", "image"]) if m in ("text", "image")],
        "reasoning": caps.get("reasoning", model_id in (QWEN, BONSAI, FLASH)),
        "compat": {"supportsStrictMode": False, "supportsDeveloperRole": False,
                   "supportsReasoningEffort": caps.get("reasoning_effort", model_id == QWEN),
                   "maxTokensField": "max_tokens"},
    }
    if info.get("provider") == "openai":
        entry["api"] = "openai-responses"
        entry["maxTokens"] = 128000
    elif info.get("provider") == "anthropic":
        entry["api"] = "anthropic-messages"
        entry["maxTokens"] = 32768
    return entry


def remap(value, old_providers):
    if isinstance(value, dict):
        mapped = copy.deepcopy(value)
        for key, item in value.items():
            new_key = remap(key, old_providers)
            if new_key != key:
                del mapped[key]
                if hasattr(mapped, "ca") and key in mapped.ca.items:
                    mapped.ca.items[new_key] = mapped.ca.items.pop(key)
            mapped[new_key] = remap(item, old_providers)
        return mapped
    if isinstance(value, list):
        return [remap(v, old_providers) for v in value]
    if not isinstance(value, str):
        return value
    if value.startswith("lfm"):
        return BONSAI
    for old in sorted(old_providers, key=len, reverse=True):
        for prefix, replacement in [("custom:" + old, "custom:" + PUBLIC), (old, PUBLIC)]:
            if value == prefix:
                return replacement
            if value.startswith(prefix + "/"):
                model = value[len(prefix) + 1:]
                return replacement + "/" + (BONSAI if model.startswith("lfm") else model)
    return value


def migrate_models(data, harness, key, catalog):
    providers = data.setdefault("providers", {})
    old = [p for p in providers if p.startswith("kortexa") and p != STATIC]
    models = {}
    overrides = {}
    for provider in old:
        definition = providers[provider]
        entries = definition.get("models", {})
        if harness == "hermes":
            entries = [{"id": k, **v} for k, v in entries.items()]
        for entry in entries:
            if not entry["id"].startswith("lfm"):
                copied = copy.deepcopy(entry)
                if harness == "pi":
                    copied["compat"] = {**definition.get("compat", {}), **copied.get("compat", {})}
                models[copied["id"]] = copied
        if harness == "hermes":
            overrides.update(data.get("model_overrides", {}).pop("custom:" + provider, {}))
    for model_id in [QWEN, BONSAI, FLASH] + [k for k, v in catalog.items() if v.get("provider") in ("openai", "anthropic")]:
        if model_id not in models:
            models[model_id] = model_entry(model_id, catalog)
    for provider in old:
        del providers[provider]
    providers.pop(STATIC, None)
    for entry in models.values():
        if "input" in entry:
            # The catalog also advertises video; OMP/pi schemas currently support text/image only.
            entry["input"] = [m for m in entry["input"] if m in ("text", "image")]

    if harness == "hermes":
        public_models = {}
        for model_id, entry in models.items():
            public_models[model_id] = {"context_length": entry.get("contextWindow", entry.get("context_length", 262144))}
            overrides[model_id] = {**overrides.get(model_id, {}), "context_window": public_models[model_id]["context_length"],
                                   "supports_vision": "image" in entry.get("input", ["text", "image"]),
                                   "supports_reasoning": entry.get("reasoning", True), "supports_tools": True}
        overrides = {k: v for k, v in overrides.items() if not k.startswith("lfm")}
        providers[PUBLIC] = {"name": PUBLIC, "api": PUBLIC_URL, "api_key": key,
                             "default_model": BONSAI, "discover_models": False, "models": public_models}
        providers[STATIC] = {"name": STATIC, "api": STATIC_URL, "default_model": FLASH,
                             "discover_models": False, "models": {FLASH: {"context_length": 262144}}}
        data.setdefault("model_overrides", {}).pop("custom:" + STATIC, None)
        data["model_overrides"]["custom:" + PUBLIC] = overrides
        data["model_overrides"]["custom:" + STATIC] = {FLASH: copy.deepcopy(overrides[FLASH])}
        for role, entry in data.get("auxiliary", {}).items():
            if entry.get("provider", "").removeprefix("custom:") in old:
                entry["provider"] = "custom:" + PUBLIC
                # The old Qwen and LFM auxiliary ports are retired; Bonsai is running.
                if entry.get("model", "").startswith("lfm") or entry.get("model") == QWEN:
                    entry["model"] = BONSAI
                entry["base_url"] = PUBLIC_URL
                entry["api_key"] = key
        model = data.setdefault("model", {})
        model.update(default="gpt-6-luna", provider="openai-codex",
                     base_url="https://chatgpt.com/backend-api/codex", api_mode="responses")
    else:
        # Older installed catalogs lack Luna. Merge only this model into the native
        # provider, retaining its OAuth auth and inherited Codex endpoint.
        native = providers.setdefault("openai-codex", {})
        native.setdefault("baseUrl", "https://chatgpt.com/backend-api")
        if harness == "omp":
            native.setdefault("auth", "oauth")
        native_models = native.setdefault("models", [])
        if not any(m["id"] == CODEX_MODEL["id"] for m in native_models):
            native_models.append(copy.deepcopy(CODEX_MODEL))
        for model_id, entry in models.items():
            entry.setdefault("compat", {})["supportsStrictMode"] = False
            entry["compat"].setdefault("maxTokensField", "max_tokens")
            if harness == "omp":
                entry.setdefault("supportsTools", True)
        public = {"baseUrl": PUBLIC_URL, "api": "openai-completions", "apiKey": key, "models": list(models.values())}
        if harness == "pi":
            public["authHeader"] = True
        providers[PUBLIC] = public
        fallback = {"baseUrl": STATIC_URL, "api": "openai-completions", "models": [copy.deepcopy(models[FLASH])]}
        if harness == "pi":
            fallback["apiKey"] = "no-key-needed"
        else:
            fallback["auth"] = "none"
        providers[STATIC] = fallback
    return old


def plan(files, key, catalog):
    result = {}
    changes = {}
    for name, text in files.items():
        if text is not None:
            result[name] = parsed(name, text) or {}
    for name, data in result.items():
        if name.endswith("models.yml") or name.endswith("models.json") or name.endswith("config.yaml"):
            harness = "hermes" if name.endswith("config.yaml") else "omp" if name.endswith("models.yml") else "pi"
            old = migrate_models(data, harness, key, catalog)
            if harness == "omp":
                settings = result.get(".omp/agent/config.yml")
                if settings is not None:
                    settings["modelRoles"] = remap(settings.get("modelRoles", {}), old)
                    settings["modelRoles"]["default"] = "openai-codex/gpt-6-luna"
            elif harness == "pi":
                settings = result.get(".pi/agent/settings.json")
                if settings is not None:
                    settings.update(remap(settings, old))
                    settings.update(defaultProvider="openai-codex", defaultModel="gpt-6-luna")
    for name, data in result.items():
        updated = serialize(name, data)
        if updated != files[name]:
            changes[name] = {"before": hashlib.sha256(files[name].encode()).hexdigest(), "after": updated}
    return result, changes


def self_test():
    old = {"providers": {"kortexa-primary": {"baseUrl": "http://old/v1", "models": [{"id": QWEN}]},
                         "kortexa-alt": {"models": [{"id": "lfm2.5-vl-3b"}]},
                         "unrelated": {"apiKey": "synthetic-preserve"}}}
    files = {".pi/agent/models.json": json.dumps(old), ".pi/agent/settings.json": json.dumps({"theme": "dark", "defaultProvider": "kortexa-primary"}),
             ".omp/agent/models.yml": serialize("models.yml", old), ".omp/agent/config.yml": "modelRoles:\n  tiny: kortexa-alt/lfm2.5-vl-3b\n",
             ".hermes/config.yaml": "model:\n  provider: openai-codex\nproviders:\n  kortexa-alt:\n    models:\n      lfm2.5-vl-3b: {}\nauxiliary:\n  title_generation:\n    provider: kortexa-alt\n    model: lfm2.5-vl-3b\n"}
    result, changes = plan(files, "synthetic-key", {})
    assert result[".pi/agent/models.json"]["providers"]["unrelated"]["apiKey"] == "synthetic-preserve"
    assert result[".pi/agent/settings.json"]["theme"] == "dark"
    assert result[".pi/agent/settings.json"]["defaultModel"] == "gpt-6-luna"
    assert result[".omp/agent/config.yml"]["modelRoles"]["tiny"] == PUBLIC + "/" + BONSAI
    assert result[".hermes/config.yaml"]["auxiliary"]["title_generation"]["model"] == BONSAI
    for name in [".pi/agent/models.json", ".omp/agent/models.yml"]:
        providers = result[name]["providers"]
        assert set(p["id"] for p in providers[PUBLIC]["models"]) == {QWEN, BONSAI, FLASH}
        assert providers[STATIC]["baseUrl"] == STATIC_URL
        assert providers["openai-codex"]["models"][0]["api"] == "openai-codex-responses"
        assert "apiKey" not in providers["openai-codex"], "Native OAuth must remain independent"
    assert plan({".hermes/config.yaml": None}, "synthetic-key", {}) == ({}, {})
    repeated = {name: serialize(name, data) for name, data in result.items()}
    assert not plan(repeated, "synthetic-key", {})[1], "Migration must be idempotent"
    assert changes
    video = model_entry(BONSAI, {BONSAI: {"input_modalities": ["text", "image", "video"]}})
    assert video["input"] == ["text", "image"]
    commented = yaml.load("tiny: kortexa-alt/lfm2.5-vl-3b # Keep this role comment\n")
    assert "# Keep this role comment" in serialize("config.yml", remap(commented, ["kortexa-alt"]))
    print("migration self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", nargs="+", choices=["snappy", "smarty", "scrappy", "moodymoose"], default=["snappy", "smarty", "scrappy", "moodymoose"])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    key = os.environ.get("KORTEXA_API_KEY", "")
    if not key:
        raise SystemExit("KORTEXA_API_KEY is required")
    request = urllib.request.Request(PUBLIC_URL + "/models", headers={"x-api-key": key})
    with urllib.request.urlopen(request, timeout=15) as response:
        catalog = {m["id"]: m for m in json.load(response)["data"] if m.get("type") == "language"}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for host in args.hosts:
        files = remote(host, READ)
        result, changes = plan(files, key, catalog)
        summary = {"host": host, "files": list(changes), "installed": list(result), "applied": args.apply}
        if args.apply and changes:
            summary.update(remote(host, WRITE, json.dumps({"stamp": stamp, "files": changes})))
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
