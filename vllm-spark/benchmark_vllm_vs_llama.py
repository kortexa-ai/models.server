#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"
LLAMA_CACHE_DIR = Path.home() / ".cache" / "llama.cpp"
LOG_DIR = ROOT / "bench-logs"
RESULTS_PATH = ROOT / "bench-results-small-models.json"

PROMPTS = [
    "Reply with the numbers 1 through 40, separated by spaces, and nothing else.",
    "Now continue with 41 through 80, separated by spaces, and nothing else.",
    "Now continue with 81 through 120, separated by spaces, and nothing else.",
]

MODELS = {
    "2B": {
        "alias": "qwen-3.5-2b",
        "vllm_repo": "Qwen/Qwen3.5-2B",
        "llama_repo": "unsloth/Qwen3.5-2B-GGUF",
        "llama_quant": "Q8_0",
        "llama_manifest": LLAMA_CACHE_DIR / "manifest=unsloth=Qwen3.5-2B-GGUF=Q8_0.json",
        "port_base": 2130,
    },
    "4B": {
        "alias": "qwen-3.5-4b",
        "vllm_repo": "Qwen/Qwen3.5-4B",
        "llama_repo": "unsloth/Qwen3.5-4B-GGUF",
        "llama_quant": "Q8_0",
        "llama_manifest": LLAMA_CACHE_DIR / "manifest=unsloth=Qwen3.5-4B-GGUF=Q8_0.json",
        "port_base": 2140,
    },
    "27B": {
        "alias": "qwen-3.5-27b",
        "vllm_repo": "Qwen/Qwen3.5-27B",
        "llama_repo": "unsloth/Qwen3.5-27B-GGUF",
        "llama_quant": "Q4_K_M",
        "llama_manifest": LLAMA_CACHE_DIR / "manifest=unsloth=Qwen3.5-27B-GGUF=Q4_K_M.json",
        "port_base": 2170,
    },
}


def run_cmd(cmd, env=None, cwd=None, check=True):
    return subprocess.run(cmd, env=env, cwd=cwd, check=check, text=True)


def http_json(method, url, payload=None, timeout=180):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        if not body.strip():
            return {}
        return json.loads(body)


def wait_for_api(port, timeout_s=600):
    deadline = time.monotonic() + timeout_s
    urls = [
        f"http://127.0.0.1:{port}/health",
        f"http://127.0.0.1:{port}/v1/models",
    ]
    while time.monotonic() < deadline:
        for url in urls:
            try:
                http_json("GET", url, timeout=5)
                return
            except Exception:
                pass
        time.sleep(1)
    raise TimeoutError(f"API on port {port} was not ready within {timeout_s}s")


def read_log_tail(path, lines=80):
    if not path.exists():
        return ""
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def parse_first_float(text, pattern):
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def gpu_compute_memory_mib():
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        pid, process_name, used = [part.strip() for part in line.split(",", 2)]
        rows.append(
            {
                "pid": int(pid),
                "process_name": process_name,
                "used_gpu_memory_mib": int(used),
            }
        )
    return rows


def gpu_compute_total_mib():
    return sum(row["used_gpu_memory_mib"] for row in gpu_compute_memory_mib())


def llama_loaded_gpu_memory_mib(log_path):
    text = log_path.read_text(errors="replace")
    patterns = [
        r"CUDA0 model buffer size =\s+([0-9.]+) MiB",
        r"CUDA0 KV buffer size =\s+([0-9.]+) MiB",
        r"CUDA0 RS buffer size =\s+([0-9.]+) MiB",
        r"CUDA0 compute buffer size =\s+([0-9.]+) MiB",
    ]
    values = [parse_first_float(text, pattern) for pattern in patterns]
    if any(value is None for value in values):
        return None
    return round(sum(values))


def prefetch_vllm(model_repo, image_tag):
    marker = HF_CACHE_DIR / "hub" / f"models--{model_repo.replace('/', '--')}"
    if marker.exists():
        return
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python",
        "-v",
        f"{HF_CACHE_DIR}:/root/.cache/huggingface",
        image_tag,
        "-c",
        (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download('{model_repo}')"
        ),
    ]
    run_cmd(cmd, cwd=ROOT)


def prefetch_llama(manifest_path, repo):
    manifest = json.loads(manifest_path.read_text())
    filename = manifest["ggufFile"]["rfilename"]
    target = LLAMA_CACHE_DIR / f"{repo.replace('/', '_')}_{filename}"
    if target.exists() and target.stat().st_size == manifest["ggufFile"]["size"]:
        return
    LLAMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    cmd = [
        "curl",
        "-L",
        "--fail",
        "-C",
        "-",
        "-o",
        str(target),
        url,
    ]
    run_cmd(cmd, cwd=ROOT)


def normalize_reply(payload):
    choice = payload["choices"][0]
    if "message" in choice:
        message = choice["message"]
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            content = "".join(parts).strip()
        if content:
            return content.strip()
        reasoning = message.get("reasoning")
        if reasoning:
            return reasoning.strip()
    text = choice.get("text")
    if text:
        return text.strip()
    return ""


def run_chat_sequence(port, model_name, max_tokens):
    messages = []
    turns = []
    for prompt in PROMPTS:
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        t0 = time.monotonic()
        response = http_json(
            "POST",
            f"http://127.0.0.1:{port}/v1/chat/completions",
            payload=payload,
            timeout=240,
        )
        elapsed_s = time.monotonic() - t0
        reply = normalize_reply(response)
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        tok_s = None
        if completion_tokens and elapsed_s > 0:
            tok_s = completion_tokens / elapsed_s
        turns.append(
            {
                "prompt": prompt,
                "elapsed_s": round(elapsed_s, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "completion_tok_s": round(tok_s, 2) if tok_s else None,
                "reply_excerpt": reply[:120],
            }
        )
        messages.append({"role": "assistant", "content": reply})
    return turns


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def stop_container(name):
    subprocess.run(
        ["docker", "rm", "-f", name],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def benchmark_vllm(model_key, cfg, args):
    container_name = f"bench-vllm-{model_key.lower()}"
    log_path = LOG_DIR / f"vllm-{model_key.lower()}.log"
    stop_container(container_name)
    env = os.environ.copy()
    env.update(
        {
            "IMAGE_TAG": args.image_tag,
            "CONTAINER_NAME": container_name,
            "MODEL_REPO": cfg["vllm_repo"],
            "SERVED_MODEL_NAME": cfg["alias"],
            "PORT": str(cfg["port_base"]),
            "LANGUAGE_MODEL_ONLY": "1",
            "MAX_MODEL_LEN": str(args.context),
            "MAX_NUM_SEQS": str(args.max_num_seqs),
            "GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
        }
    )
    cmd = [str(ROOT / "run-vllm-docker.sh")]
    if args.kv_cache_memory_bytes:
        cmd.extend(["--kv-cache-memory-bytes", args.kv_cache_memory_bytes])
    if args.kv_cache_dtype:
        cmd.extend(["--kv-cache-dtype", args.kv_cache_dtype])

    baseline_mib = gpu_compute_total_mib()
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            t0 = time.monotonic()
            wait_for_api(cfg["port_base"], timeout_s=args.startup_timeout)
            ready_s = time.monotonic() - t0
            loaded_mib = gpu_compute_total_mib() - baseline_mib
            turns = run_chat_sequence(cfg["port_base"], cfg["alias"], args.max_tokens)
            return {
                "engine": "vllm",
                "model": model_key,
                "context": args.context,
                "max_tokens": args.max_tokens,
                "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
                "kv_cache_dtype": args.kv_cache_dtype,
                "startup_s": round(ready_s, 3),
                "loaded_gpu_memory_mib_delta": loaded_mib,
                "turns": turns,
                "log_path": str(log_path),
            }
        except Exception as exc:
            raise RuntimeError(f"vLLM benchmark failed for {model_key}: {exc}\n{read_log_tail(log_path)}")
        finally:
            stop_process(proc)
            stop_container(container_name)


def benchmark_llama(model_key, cfg, args):
    log_path = LOG_DIR / f"llama-{model_key.lower()}.log"
    cmd = [
        "llama-server",
        "-hf",
        f"{cfg['llama_repo']}:{cfg['llama_quant']}",
        "--alias",
        cfg["alias"],
        "--host",
        "127.0.0.1",
        "--port",
        str(cfg["port_base"] + 1),
        "--jinja",
        "--no-mmproj",
        "-c",
        str(args.context),
        "-ngl",
        "99",
        "--threads",
        "-1",
        "--parallel",
        "1",
        "--flash-attn",
        "on",
        "--cache-type-k",
        args.llama_cache_type,
        "--cache-type-v",
        args.llama_cache_type,
        "--no-mmap",
    ]

    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            t0 = time.monotonic()
            wait_for_api(cfg["port_base"] + 1, timeout_s=args.startup_timeout)
            ready_s = time.monotonic() - t0
            loaded_mib = llama_loaded_gpu_memory_mib(log_path)
            turns = run_chat_sequence(cfg["port_base"] + 1, cfg["alias"], args.max_tokens)
            return {
                "engine": "llama",
                "model": model_key,
                "context": args.context,
                "max_tokens": args.max_tokens,
                "llama_quant": cfg["llama_quant"],
                "llama_cache_type": args.llama_cache_type,
                "startup_s": round(ready_s, 3),
                "loaded_gpu_memory_mib_delta": loaded_mib,
                "turns": turns,
                "log_path": str(log_path),
            }
        except Exception as exc:
            raise RuntimeError(
                f"llama benchmark failed for {model_key}: {exc}\n{read_log_tail(log_path)}"
            )
        finally:
            stop_process(proc)


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM vs llama-server on Qwen 3.5 small models.")
    parser.add_argument("--models", nargs="+", default=["2B", "4B"], choices=sorted(MODELS))
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--image-tag", default="kortexa/qwen-3.5-27b-vllm:ngc-26.02")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--kv-cache-memory-bytes", default="16G")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--llama-cache-type", default="q4_0")
    parser.add_argument("--startup-timeout", type=int, default=900)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("Warming local caches...")
    for model_key in args.models:
        cfg = MODELS[model_key]
        prefetch_vllm(cfg["vllm_repo"], args.image_tag)
        prefetch_llama(cfg["llama_manifest"], cfg["llama_repo"])

    results = []
    for model_key in args.models:
        cfg = MODELS[model_key]
        print(f"Benchmarking vLLM {model_key}...")
        results.append(benchmark_vllm(model_key, cfg, args))
        print(f"Benchmarking llama {model_key}...")
        results.append(benchmark_llama(model_key, cfg, args))

    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
