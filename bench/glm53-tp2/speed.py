#!/usr/bin/env python3
"""Measure isolated cold/warm prefill and sustained decode on an existing server."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

import capacity


def metrics(base):
    with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
        text = response.read().decode()
    values = {}
    wanted = {"num_requests_running", "num_requests_waiting", "num_preemptions_total",
              "prefix_cache_hits_total", "prefix_cache_queries_total",
              "spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
              "spec_decode_num_accepted_tokens_total"}
    wanted.update(f"request_{phase}_time_seconds_{field}"
                  for phase in ("prefill", "decode", "queue") for field in ("sum", "count"))
    for name, value in re.findall(r"^vllm:([a-z_]+)\{[^\n]*\} ([^\n]+)$", text, re.M):
        if name not in wanted:
            continue
        if name in values:
            raise ValueError("Expected one model/engine in metrics: " + name)
        values[name] = float(value)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--contexts", type=int, nargs="+", required=True)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "speed-summary.json"
    if summary_path.exists():
        raise ValueError("Use a new output directory; evidence must not be overwritten")
    namespace = hashlib.sha256(str(args.output_dir.resolve()).encode()).hexdigest()[:12]
    fixtures = args.fixture_dir / namespace
    fixtures.mkdir(parents=True, exist_ok=True)
    capacity.BASE, capacity.MODEL, capacity.FIXTURES = base, args.model, fixtures
    rows = []
    for context in args.contexts:
        item = capacity.prepare(f"speed-{namespace}-p{context}", context)
        (fixtures / f"p{context}-00.request.json").write_text(json.dumps(item))
        for state in ("cold", "warm"):
            tag = f"p{context}-{state}"
            before = metrics(base)
            if before["num_requests_running"] or before["num_requests_waiting"]:
                raise RuntimeError("Latency measurements require an idle endpoint")
            command = [sys.executable, str(Path(__file__).with_name("capacity.py")),
                       "--base", base, "--model", args.model, "--clients", "1",
                       "--prompt-tokens", str(context), "--output-tokens", str(args.output_tokens),
                       "--min-tokens", str(args.output_tokens), "--tag", tag,
                       "--output-dir", str(args.output_dir), "--fixture-dir", str(fixtures)]
            print(json.dumps({"starting": tag, "output_tokens": args.output_tokens}), flush=True)
            child = subprocess.run(command)
            result_path = args.output_dir / (tag + ".results.json")
            if child.returncode or not result_path.exists():
                raise RuntimeError(f"Capacity/stream/retrieval probe failed: {tag}")
            result = json.loads(result_path.read_text())
            request = result["requests"][0]
            after = metrics(base)
            for _ in range(20):
                if after["request_prefill_time_seconds_count"] > before["request_prefill_time_seconds_count"]:
                    break
                time.sleep(0.25)
                after = metrics(base)
            delta = {key: after[key] - value for key, value in before.items() if key in after}
            for phase in ("prefill", "decode", "queue"):
                if delta[f"request_{phase}_time_seconds_count"] != 1:
                    raise RuntimeError("Cannot attribute server metrics to exactly one request")
            cached = delta["prefix_cache_hits_total"]
            prompt = request["usage"]["prompt_tokens"]
            uncached = prompt - cached
            prefill = delta["request_prefill_time_seconds_sum"]
            if (state == "cold" and cached != 0) or not 0 <= cached <= prompt:
                raise RuntimeError("Unexpected cache-hit accounting")
            if state == "warm" and context > 16384 and cached == 0:
                raise RuntimeError("Warm long-context request did not reuse its prefix")
            row = {"tag": tag, "state": state, "prompt_tokens": prompt,
                   "output_tokens": request["usage"]["completion_tokens"],
                   "client_ttft_s": request["ttft_s"],
                   "server_prefill_s": prefill,
                   "uncached_prompt_tokens": uncached, "cached_prompt_tokens": cached,
                   "uncached_prefill_tok_s": uncached / prefill if prefill else None,
                   "client_decode_tok_s": request["decode_tok_s"],
                   "server_decode_s": delta["request_decode_time_seconds_sum"],
                   "server_queue_s": delta["request_queue_time_seconds_sum"],
                   "total_s": request["elapsed_s"],
                   "preemptions": delta["num_preemptions_total"],
                   "static_min_available_gib": result["summary"]["static_minimum_available_gib"],
                   "shock_min_available_gib": result["summary"]["shock_minimum_available_gib"],
                   "pass": request["pass"]}
            rows.append(row)
            (args.output_dir / (tag + ".server-metrics.json")).write_text(
                json.dumps({"before": before, "after": after, "delta": delta}, indent=2) + "\n")
            summary_path.write_text(json.dumps({"base": base, "model": args.model,
                "scope": "synthetic fixed-length decode; warm rate excludes cached tokens",
                "rows": rows}, indent=2) + "\n")
            print(json.dumps({"speed": row}), flush=True)


if __name__ == "__main__":
    main()
