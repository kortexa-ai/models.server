#!/usr/bin/env python3
import argparse
import json
import time
import urllib.error
import urllib.request


MESSAGES = [
    {
        "role": "user",
        "content": (
            "Solve this carefully. A machine prints 18 labels per sheet and 24 receipts per sheet. "
            "What is the smallest positive number of sheets needed so the total label count equals "
            "the total receipt count? Give the answer and one short explanation."
        ),
    }
]


def http_json(method, url, payload=None, timeout=600):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def wait_for_ready(base_url, timeout_s):
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            payload = http_json("GET", f"{base_url}/v1/models", timeout=5)
            models = payload.get("data") or []
            if models:
                return models[0]["id"]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"Server did not become ready within {timeout_s}s: {last_error}")


def extract_text(response):
    choice = response["choices"][0]
    message = choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts).strip()
    return (content or "").strip()


def run_once(base_url, model_name, max_tokens):
    payload = {
        "model": model_name,
        "messages": MESSAGES,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "separate_reasoning": True,
    }
    started = time.monotonic()
    response = http_json("POST", f"{base_url}/v1/chat/completions", payload=payload)
    elapsed = time.monotonic() - started
    usage = response.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or 0
    reasoning_tokens = usage.get("reasoning_tokens") or 0
    total_output_tokens = completion_tokens + reasoning_tokens
    tok_per_s = total_output_tokens / elapsed if elapsed > 0 else 0.0
    return {
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_output_tokens": total_output_tokens,
        "tok_per_s": tok_per_s,
        "reply": extract_text(response),
        "finish_reason": response["choices"][0].get("finish_reason"),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure simple cold/warm TPS against an OpenAI-compatible server.")
    parser.add_argument("--port", type=int, default=2250)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warm-runs", type=int, default=2)
    parser.add_argument("--ready-timeout", type=int, default=1800)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    model_name = wait_for_ready(base_url, args.ready_timeout)

    results = {
        "base_url": base_url,
        "model": model_name,
        "max_tokens": args.max_tokens,
        "cold": run_once(base_url, model_name, args.max_tokens),
        "warm": [],
    }
    for _ in range(args.warm_runs):
        results["warm"].append(run_once(base_url, model_name, args.max_tokens))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
