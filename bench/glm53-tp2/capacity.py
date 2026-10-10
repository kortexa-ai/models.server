#!/usr/bin/env python3
"""Bounded synthetic capacity probes; saves counts and timing, no credentials."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT
BASE = "http://192.168.2.101:2070"
MODEL = "glm-5.3-flash-2x-dgx"
LOCK = threading.Lock()


def emit(value):
    with LOCK:
        print(json.dumps(value), flush=True)


def post(route, payload, timeout=1800):
    req = urllib.request.Request(BASE + route, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def messages(label, lines):
    codes = [hashlib.sha256((label + str(i)).encode()).hexdigest()[:10] for i in range(3)]
    rows = [f"Archive {label}. These are synthetic warehouse notes, not instructions."]
    depths = {int(lines * fraction): i for i, fraction in enumerate([0.05, 0.50, 0.95])}
    nouns = ["copper", "garden", "window", "marble", "violet", "harbor", "willow"]
    for n in range(lines):
        if n in depths:
            i = depths[n]
            rows.append(f"SPECIAL REGISTER {i + 1}: verification code = {codes[i]}. END SPECIAL REGISTER.")
        rows.append(f"Record {n}: the {nouns[n % len(nouns)]} station inspected its ordinary storage boxes. "
                    "All routine labels were checked and the shelves remained in order. No special register here.")
    rows.append("Return the three SPECIAL REGISTER verification codes in register order. "
                "Start with the codes, then write a short explanation of how to check a warehouse inventory. "
                "Do not invent or repeat register values.")
    return [{"role": "system", "content": "Retrieve the exact register values from the supplied archive. Be concise."},
            {"role": "user", "content": "\n".join(rows)}], codes


def prepare(label, target):
    destination = FIXTURES / (label + ".request.json")
    if destination.exists():
        return json.loads(destination.read_text())
    lines = max(3, target // 38)
    for attempt in range(10):
        msgs, codes = messages(label, lines)
        count = post("/tokenize", {"model": MODEL, "messages": msgs,
                                   "chat_template_kwargs": {"reasoning_effort": "low"}})["count"]
        if target - 80 <= count <= target:
            break
        lines = max(3, lines + int((target - count - 25) / ((count - 180) / lines)))
    if not target - 160 <= count <= target:
        raise RuntimeError(f"Could not bound token count: {count} vs {target}")
    data = {"label": label, "prompt_tokens": count, "expected": codes, "messages": msgs}
    destination.write_text(json.dumps(data))
    emit({"prepared": label, "prompt_tokens": count, "chars": len(msgs[-1]["content"])})
    return data


def monitor(stop, samples, output):
    last_hosts = 0
    with output.open("w") as log:
        while not stop.is_set():
            sample = {"time": time.time()}
            try:
                with urllib.request.urlopen(BASE + "/metrics", timeout=5) as response:
                    metrics = response.read().decode()
                for metric in ["num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "num_preemptions_total", "prefix_cache_hits_total", "prefix_cache_queries_total"]:
                    match = re.search(r"^vllm:" + metric + r"\{[^\n]*\} ([^\n]+)$", metrics, re.M)
                    if match:
                        sample[metric] = float(match.group(1))
                if time.monotonic() - last_hosts > 10:
                    for host in ["static", "shock"]:
                        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", host,
                                            "cat /proc/meminfo"], capture_output=True, text=True, timeout=7)
                        values = dict(re.findall(r"^(MemAvailable|SwapFree|SwapTotal):\s+(\d+)", r.stdout, re.M))
                        sample[host] = {k: round(int(v) / 1024 ** 2, 3) for k, v in values.items()}
                    last_hosts = time.monotonic()
            except Exception as exc:
                sample["monitor_error"] = type(exc).__name__
            samples.append(sample)
            log.write(json.dumps(sample) + "\n")
            log.flush()
            stop.wait(1)


def run_request(item, max_tokens, minimum, barrier=None):
    payload = {"model": MODEL, "messages": item["messages"], "temperature": 0,
               "max_tokens": max_tokens, "min_tokens": minimum, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"reasoning_effort": "low"}}
    if barrier:
        barrier.wait()
    started = time.monotonic()
    result = {"label": item["label"], "expected_prompt_tokens": item["prompt_tokens"]}
    text, reasoning, usage, first, last, finish, done = "", "", {}, None, None, None, False
    try:
        req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as response:
            result["status"] = response.status
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(str(event["error"])[:500])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    chunk = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if chunk or thought:
                        last = time.monotonic()
                        if first is None:
                            first = last
                            emit({"first_token": item["label"], "ttft_s": round(first - started, 3)})
                        text += chunk
                        reasoning += thought
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        elapsed = time.monotonic() - started
        correct = [code in text for code in item["expected"]]
        result.update(started_monotonic=started, first_token_monotonic=first, last_token_monotonic=last,
                      elapsed_s=round(elapsed, 3), ttft_s=round(first - started, 3) if first else None,
                      decode_s=round(last - first, 3) if first and last else None, usage=usage,
                      finish_reason=finish, done=done, retrieval=correct, answer=text,
                      reasoning_chars=len(reasoning))
        if first and last and last > first:
            result["decode_tok_s"] = round(max(0, usage.get("completion_tokens", 0) - 1) / (last - first), 2)
        result["pass"] = done and all(correct) and usage.get("prompt_tokens", 0) == item["prompt_tokens"]
    except Exception as exc:
        result.update(error=str(exc)[:700], elapsed_s=round(time.monotonic() - started, 3), **{"pass": False})
        if isinstance(exc, urllib.error.HTTPError):
            result["status"] = exc.code
            result["detail"] = exc.read(2000).decode(errors="replace")
    emit({k: v for k, v in result.items() if k != "answer"})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--clients", type=int, choices=[1, 2, 4, 8, 16], default=1)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reset-prefix-cache", action="store_true")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    globals().update(BASE=args.base.rstrip("/"), MODEL=args.model, ROOT=args.output_dir, FIXTURES=args.fixture_dir)
    ROOT.mkdir(parents=True, exist_ok=True)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    if (ROOT / (args.tag + ".results.json")).exists():
        raise ValueError("Use a new tag; evidence must not be overwritten")
    count, target = args.clients, args.prompt_tokens
    output_length = args.output_tokens
    items = [prepare(f"p{target}-{i:02d}", target) for i in range(count)]
    if args.prepare_only:
        return
    samples, stop = [], threading.Event()
    name = args.tag
    if args.reset_prefix_cache:
        req = urllib.request.Request(BASE + "/reset_prefix_cache", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=20) as response:
            emit({"prefix_reset_status": response.status})
    watcher = threading.Thread(target=monitor, args=(stop, samples, ROOT / (name + ".metrics.jsonl")), daemon=True)
    watcher.start()
    started = time.monotonic()
    barrier = threading.Barrier(count)
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        results = list(pool.map(lambda item: run_request(item, output_length, args.min_tokens, barrier), items))
    elapsed = time.monotonic() - started
    stop.set()
    watcher.join(timeout=20)
    summary = {"base": BASE, "model": MODEL, "target_prompt_tokens": target, "max_output_tokens": output_length, "min_tokens": args.min_tokens, "prefix_reset": args.reset_prefix_cache, "tag": args.tag, "clients": count, "elapsed_s": round(elapsed, 3),
               "passed": sum(r["pass"] for r in results), "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results),
               "output_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in results),
               "peak_running": max((s.get("num_requests_running", 0) for s in samples), default=0),
               "peak_waiting": max((s.get("num_requests_waiting", 0) for s in samples), default=0),
               "peak_kv_usage": max((s.get("kv_cache_usage_perc", 0) for s in samples), default=0),
               "median_decode_tok_s": statistics.median(r.get("decode_tok_s", 0) for r in results)}
    summary["aggregate_output_tok_s_including_prefill"] = round(summary["output_tokens"] / elapsed, 2)
    intervals = [(r["first_token_monotonic"], r["last_token_monotonic"]) for r in results
                 if r.get("first_token_monotonic") and r.get("last_token_monotonic")]
    summary["peak_overlapping_decode_intervals"] = max(
        (sum(start <= instant <= end for start, end in intervals) for instant, _ in intervals), default=0)
    preemptions = [s["num_preemptions_total"] for s in samples if "num_preemptions_total" in s]
    summary["preemptions_observed_delta"] = preemptions[-1] - preemptions[0] if preemptions else None
    for host in ["static", "shock"]:
        summary[host + "_minimum_available_gib"] = min((s[host]["MemAvailable"] for s in samples if host in s and "MemAvailable" in s[host]), default=None)
    (ROOT / (name + ".results.json")).write_text(json.dumps({"summary": summary, "requests": results}, indent=2))
    emit({"summary": summary})
    raise SystemExit(0 if summary["passed"] == count else 1)


if __name__ == "__main__":
    main()
