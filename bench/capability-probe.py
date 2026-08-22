#!/usr/bin/env python3
"""Run small compatibility and cache probes against an existing endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any


def png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def checkerboard_png(size: int) -> bytes:
    rows = []
    block = max(1, size // 8)
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            bright = ((x // block) + (y // block)) % 2 == 0
            row.extend((240, 240, 240) if bright else (20, 60, 140))
        rows.append(bytes(row))
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return header + png_chunk(b"IHDR", ihdr) + png_chunk(
        b"IDAT", zlib.compress(b"".join(rows), 9)
    ) + png_chunk(b"IEND", b"")


def post_json(url: str, payload: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
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
    return {"wall_seconds": time.monotonic() - started, "response": body}


def guarded_probe(name: str, action: Any) -> dict[str, Any]:
    try:
        result = action()
        return {"name": name, "ok": True, **result}
    except Exception as error:  # noqa: BLE001 - capability failures are results.
        return {"name": name, "ok": False, "error": str(error)}


def chat_payload(model_id: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
        "cache_prompt": True,
    }


def summarize_embedding(result: dict[str, Any]) -> dict[str, Any]:
    response = result.pop("response")
    vectors = [item.get("embedding", []) for item in response.get("data", [])]
    result["vectors"] = [
        {
            "dimension": len(vector),
            "l2_norm": math.sqrt(sum(float(value) ** 2 for value in vector)),
        }
        for vector in vectors
    ]
    result["usage"] = response.get("usage")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()

    config = json.loads(args.model_file.read_text())
    model_id = config["id"]
    base_url = f"http://127.0.0.1:{config['port']}"
    results: list[dict[str, Any]] = []

    if config.get("embedding"):
        payload = {
            "model": model_id,
            "input": [
                "The blue heron stands beside the lake.",
                "A bird waits at the edge of the water.",
                "Distributed databases need careful consistency rules.",
            ],
        }
        results.append(
            guarded_probe(
                "embedding_batch",
                lambda: summarize_embedding(
                    post_json(f"{base_url}/v1/embeddings", payload)
                ),
            )
        )
    else:
        prompt = (
            "Write a compact paragraph explaining why deterministic inputs "
            "matter in performance tests."
        )
        results.append(
            guarded_probe(
                "text_cold",
                lambda: post_json(
                    f"{base_url}/v1/chat/completions", chat_payload(model_id, prompt)
                ),
            )
        )
        results.append(
            guarded_probe(
                "text_warm",
                lambda: post_json(
                    f"{base_url}/v1/chat/completions", chat_payload(model_id, prompt)
                ),
            )
        )

        tool_payload = chat_payload(model_id, "What is the weather in Paris?")
        tool_payload.update(
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "description": "Look up current weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "lookup_weather"},
                },
            }
        )
        results.append(
            guarded_probe(
                "required_tool_call",
                lambda: post_json(f"{base_url}/v1/chat/completions", tool_payload),
            )
        )

        if config.get("multimodal"):
            for size in (512, 1536):
                encoded = base64.b64encode(checkerboard_png(size)).decode()
                vision_payload = {
                    "model": model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Briefly describe the geometric pattern.",
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
                    "cache_prompt": True,
                }
                results.append(
                    guarded_probe(
                        f"vision_{size}px",
                        lambda payload=vision_payload: post_json(
                            f"{base_url}/v1/chat/completions", payload
                        ),
                    )
                )

    document = {
        "schema_version": 1,
        "model_id": model_id,
        "base_url": base_url,
        "results": results,
    }
    args.output_file.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    failed = sum(not result["ok"] for result in results)
    print(f"Capability probes complete: {len(results) - failed} passed, {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
