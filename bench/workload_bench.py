#!/usr/bin/env python3
"""Run focused mixed-chat or independent-image workloads against llama-server."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import struct
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

CHAT_PROFILES = (
    ("short", 512, 2),
    ("medium", 8192, 4),
    ("long", 32768, 8),
    ("very_long", 131072, 16),
)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def pipeline_png(size: int, request_id: int) -> bytes:
    """Create a deterministic, distinct RGB image with equal dimensions and cost."""
    rows = []
    block = max(1, size // 16)
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            region = (x // block) + 3 * (y // block) + request_id
            row.extend(
                (
                    (37 * region + 17 * request_id) % 256,
                    (73 * region + 29 * request_id) % 256,
                    (109 * region + 43 * request_id) % 256,
                )
            )
        rows.append(bytes(row))
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        header
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
        + png_chunk(b"IEND", b"")
    )


def filler_words(count: int, label: str) -> str:
    phrase = (
        f"{label} datum records a stable observation for this synthetic chat history "
        "while preserving deterministic order and reproducible workload shape. "
    )
    words = phrase.split()
    repetitions = math.ceil(count / len(words))
    return " ".join((words * repetitions)[:count])


def mixed_chat_messages(
    label: str, target_words: int, turns: int
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": "Answer the final request concisely. Treat earlier turns as chat history.",
        }
    ]
    words_per_message = max(1, target_words // (turns * 2))
    for turn in range(turns):
        messages.append(
            {
                "role": "user",
                "content": filler_words(words_per_message, f"{label}-user-{turn}"),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": filler_words(words_per_message, f"{label}-assistant-{turn}"),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": f"For session {label}, reply with only its label and the word ready.",
        }
    )
    return messages


def post_json(
    url: str, payload: dict[str, Any], timeout: float
) -> tuple[float, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    return time.monotonic() - started, body


def summarize_response(
    name: str,
    metadata: dict[str, Any],
    action: Any,
) -> dict[str, Any]:
    try:
        wall_seconds, response = action()
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        return {
            "name": name,
            "ok": True,
            **metadata,
            "wall_seconds": wall_seconds,
            "usage": response.get("usage") or {},
            "timings": response.get("timings") or {},
            "finish_reason": choice.get("finish_reason"),
            "response_chars": len(content),
            "assistant_content": content,
        }
    except Exception as error:  # noqa: BLE001 - failed requests are benchmark results.
        return {"name": name, "ok": False, **metadata, "error": str(error)}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def aggregate(results: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    successful = [item for item in results if item["ok"]]
    latencies = [float(item["wall_seconds"]) for item in successful]
    prompt_tokens = sum(
        int(item["usage"].get("prompt_tokens", 0)) for item in successful
    )
    completion_tokens = sum(
        int(item["usage"].get("completion_tokens", 0)) for item in successful
    )
    return {
        "wall_seconds": wall_seconds,
        "requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "requests_per_second": len(successful) / wall_seconds if wall_seconds else None,
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_per_second": prompt_tokens / wall_seconds
        if wall_seconds
        else None,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": (
            completion_tokens / wall_seconds if wall_seconds else None
        ),
        "latency_seconds": {
            "min": min(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
    }


def run_mixed_chat(
    base_url: str,
    model_id: str,
    concurrency: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    sessions = [
        {
            "label": label,
            "target_history_words": target_words,
            "messages": mixed_chat_messages(label, target_words, turns),
        }
        for label, target_words, turns in CHAT_PROFILES
    ]
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    for round_number in (1, 2):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for session in sessions:
                payload = {
                    "model": model_id,
                    "messages": session["messages"],
                    "temperature": 0,
                    "max_tokens": 128,
                    "cache_prompt": True,
                }
                metadata = {
                    "session": session["label"],
                    "round": round_number,
                    "target_history_words": session["target_history_words"],
                    "message_count": len(session["messages"]),
                    "cache_prompt": True,
                }
                futures.append(
                    (
                        session,
                        executor.submit(
                            summarize_response,
                            f"{session['label']}_round_{round_number}",
                            metadata,
                            lambda payload=payload: post_json(
                                f"{base_url}/v1/chat/completions", payload, timeout
                            ),
                        ),
                    )
                )
            for session, future in futures:
                result = future.result()
                results.append(result)
                if result["ok"]:
                    session["messages"].append(
                        {
                            "role": "assistant",
                            "content": result.pop("assistant_content") or "ready",
                        }
                    )
                    session["messages"].append(
                        {
                            "role": "user",
                            "content": (
                                f"Continue session {session['label']}. Reply with only "
                                "the label and the round number."
                            ),
                        }
                    )
                else:
                    result.pop("assistant_content", None)
    return results, time.monotonic() - started


def run_vision_pipeline(
    base_url: str,
    model_id: str,
    concurrency: int,
    request_count: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    payloads = []
    for request_id in range(request_count):
        encoded = base64.b64encode(pipeline_png(1024, request_id)).decode()
        payloads.append(
            {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe the dominant colors and geometry in one sentence.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded}"
                                },
                            },
                        ],
                    }
                ],
                "temperature": 0,
                "max_tokens": 64,
                "cache_prompt": False,
            }
        )

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                summarize_response,
                f"image_{request_id}",
                {
                    "request_id": request_id,
                    "image_size_px": 1024,
                    "cache_prompt": False,
                },
                lambda payload=payload: post_json(
                    f"{base_url}/v1/chat/completions", payload, timeout
                ),
            )
            for request_id, payload in enumerate(payloads)
        ]
        results = [future.result() for future in futures]
    for result in results:
        result.pop("assistant_content", None)
    return results, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument(
        "--scenario", choices=("mixed-chat", "vision-pipeline"), required=True
    )
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    if args.concurrency < 1 or args.requests < 1:
        parser.error("concurrency and requests must be positive")

    config = json.loads(args.model_file.read_text())
    if config.get("embedding"):
        parser.error("focused generation workloads do not support embedding models")
    if args.scenario == "vision-pipeline" and not config.get("multimodal"):
        parser.error("vision-pipeline requires a multimodal model")

    model_id = config["id"]
    base_url = f"http://127.0.0.1:{config['port']}"
    if args.scenario == "mixed-chat":
        results, wall_seconds = run_mixed_chat(
            base_url, model_id, args.concurrency, args.timeout
        )
    else:
        results, wall_seconds = run_vision_pipeline(
            base_url, model_id, args.concurrency, args.requests, args.timeout
        )

    document = {
        "schema_version": 1,
        "model_id": model_id,
        "base_url": base_url,
        "scenario": args.scenario,
        "concurrency": args.concurrency,
        "aggregate": aggregate(results, wall_seconds),
        "results": results,
    }
    args.output_file.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    aggregate_result = document["aggregate"]
    print(
        f"{args.scenario}: {aggregate_result['successful_requests']}/"
        f"{aggregate_result['requests']} requests succeeded, "
        f"{aggregate_result['requests_per_second']:.3f} requests/s"
    )
    return 0 if aggregate_result["failed_requests"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
