#!/usr/bin/env python3
"""Targeted October 2026 client switch; preview by default, private backups on apply.

Run on Snappy with Hermes's Python (ruamel.yaml). Existing credentials, auxiliary
roles and unrelated providers are retained. This is not a provider consolidation.
"""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("migration", Path(__file__).with_name("migrate-harness-providers.py"))
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)

MODEL = "glm-5.3-flash-2x-dgx"
PUBLIC = "kortexa.ai"
DUAL = "kortexa-dual"
DIRECT = "http://192.168.2.101:2070/v1"
SOL = "gpt-6.1-sol"
WINDOW = 262144

READ = '''import json,pathlib
root=pathlib.Path.home()
files={n:str(root/n) for n in ['.omp/agent/models.yml','.omp/agent/config.yml','.pi/agent/models.json','.pi/agent/settings.json']}
if str(root).startswith('/Users/'):
 files.update({n:str(root/n) for n in ['.hermes/config.yaml','.hermes/profiles/mira/config.yaml']})
win=pathlib.Path('/mnt/c/Users/francip/AppData/Local/hermes/config.yaml')
if win.is_file(): files['windows-hermes/config.yaml']=str(win)
print(json.dumps({n:{'path':p,'text':pathlib.Path(p).read_text()} for n,p in files.items() if pathlib.Path(p).is_file()}))
'''

WRITE = '''import json,pathlib,hashlib,sys,os,tempfile
data=json.load(sys.stdin); root=pathlib.Path.home()
for name,item in data['files'].items():
 f=pathlib.Path(item['path'])
 if name.startswith('/') or '..' in pathlib.Path(name).parts:
  raise SystemExit('Invalid backup path')
 expected=pathlib.Path('/mnt/c/Users/francip/AppData/Local/hermes/config.yaml') if name=='windows-hermes/config.yaml' else root/name
 if f!=expected or f.is_symlink() or not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=item['before']:
  raise SystemExit('Concurrent change or unsafe target: '+name)
backup=root/'.local/share/kortexa/glm53-client-switch'/data['stamp']
backup.mkdir(parents=True,exist_ok=False,mode=0o700)
for name,item in data['files'].items():
 f=pathlib.Path(item['path']); b=backup/name
 b.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with b.open('xb') as out: out.write(f.read_bytes())
 os.chmod(b,0o600)
 # Recheck immediately before replacement, after backup I/O.
 if hashlib.sha256(f.read_bytes()).hexdigest()!=item['before']:
  raise SystemExit('Concurrent change before replacement: '+name)
 with tempfile.NamedTemporaryFile(dir=f.parent,delete=False) as out:
  out.write(item['after'].encode()); out.flush(); os.fsync(out.fileno()); tmp=out.name
 os.chmod(tmp,0o600); os.replace(tmp,f)
 if f.read_text()!=item['after']: raise SystemExit('Readback mismatch: '+name)
print(json.dumps({'backup':str(backup),'files':list(data['files'])}))
'''


def entry(harness):
    value = {"id": MODEL, "name": "GLM-5.3 Flash NVFP4 DFlash2 (dual DGX Spark)",
             "api": "openai-completions", "contextWindow": WINDOW, "maxTokens": 32768,
             "input": ["text", "image"], "reasoning": True,
             "compat": {"supportsStrictMode": False, "supportsDeveloperRole": False,
                        "supportsReasoningEffort": True, "maxTokensField": "max_tokens"}}
    if harness == "omp":
        value["supportsTools"] = True
    return value


def upsert(models, value):
    for index, old in enumerate(models):
        if old["id"] == value["id"]:
            models[index] = value
            return
    models.append(value)


def update_providers(data, harness):
    providers = data["providers"]
    public = providers[PUBLIC]
    if harness == "hermes":
        public["models"][MODEL] = {"context_length": WINDOW}
        overrides = {"context_window": WINDOW, "supports_vision": True,
                     "supports_reasoning": True, "supports_tools": True}
        data.setdefault("model_overrides", {}).setdefault("custom:" + PUBLIC, {})[MODEL] = copy.deepcopy(overrides)
        providers[DUAL] = {"name": DUAL, "api": DIRECT, "default_model": MODEL,
                           "discover_models": False, "models": {MODEL: {"context_length": WINDOW}}}
        data["model_overrides"]["custom:" + DUAL] = {MODEL: overrides}
        data["model"].update(default=SOL, provider="openai-codex",
                             base_url="https://chatgpt.com/backend-api/codex", api_mode="responses")
    else:
        upsert(public["models"], entry(harness))
        providers[DUAL] = {"baseUrl": DIRECT, "api": "openai-completions", "models": [entry(harness)]}
        if harness == "omp":
            providers[DUAL]["auth"] = "none"
        else:
            providers[DUAL]["apiKey"] = "no-key-needed"
            # All three installed pi catalogs already contain native Codex Sol.
            # Keep its catalog metadata and the machine's existing OAuth login.


def plan(host, files):
    parsed = {name: migration.parsed(name, item["text"]) for name, item in files.items()}
    for name, data in parsed.items():
        if name.endswith("config.yaml"):
            update_providers(data, "hermes")
        elif name.endswith("models.yml"):
            update_providers(data, "omp")
        elif name.endswith("models.json"):
            update_providers(data, "pi")
        elif name.endswith("config.yml"):
            roles = data.setdefault("modelRoles", {})
            previous = roles.get("default", "")
            # Preserve the user's explicit effort suffix when changing the model.
            effort = ":" + previous.rsplit(":", 1)[1] if ":" in previous else ""
            roles["default"] = PUBLIC + "/" + MODEL + effort if host == "snappy" else "openai-codex/gpt-6-luna"
        elif name.endswith("settings.json"):
            data.update(defaultProvider="openai-codex", defaultModel=SOL)
    changes = {}
    for name, data in parsed.items():
        # Avoid serialization-only writes and unnecessary backups.
        if data == migration.parsed(name, files[name]["text"]):
            continue
        changes[name] = {"path": files[name]["path"], "before": hashlib.sha256(files[name]["text"].encode()).hexdigest(),
                         "after": migration.serialize(name, data)}
    return parsed, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--host", choices=["snappy", "smarty", "scrappy"], action="append")
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for host in args.host or ["snappy", "smarty", "scrappy"]:
        files = migration.remote(host, READ)
        result, changes = plan(host, files)
        print(json.dumps({"host": host, "changed_files": list(changes), "public_model": MODEL,
                          "fallback": DUAL, "default_omp": result.get(".omp/agent/config.yml", {}).get("modelRoles", {}).get("default"),
                          "default_pi": result.get(".pi/agent/settings.json", {}).get("defaultModel")}))
        if args.apply and changes:
            print(json.dumps({"host": host, **migration.remote(host, WRITE, json.dumps({"stamp": stamp, "files": changes}))}))
        if args.apply:
            _, remaining = plan(host, migration.remote(host, READ))
            if remaining:
                raise RuntimeError(f"Post-write verification failed on {host}")


if __name__ == "__main__":
    main()
