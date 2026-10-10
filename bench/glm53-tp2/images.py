#!/usr/bin/env python3
"""Eight large images with distinct readable codes; stdlib-only PNG fixture."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct
import zlib

from quality import stream

DIGITS = [
    ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
]
CODES = ["8173", "2491", "5630", "9264", "1385", "7042", "3958", "6819"]


def png(code):
    width, height, scale = 2560, 960, 32
    pixels = bytearray(b"\xff" * (width * height * 3))
    x0 = (width - len(code) * 7 * scale) // 2
    y0 = (height - 7 * scale) // 2
    for index, digit in enumerate(code):
        for y, row in enumerate(DIGITS[int(digit)]):
            for x, bit in enumerate(row):
                if bit == "1":
                    for dy in range(scale):
                        start = ((y0 + y * scale + dy) * width + x0 + (index * 7 + x) * scale) * 3
                        pixels[start:start + scale * 3] = bytes(scale * 3)
    raw = b"".join(b"\0" + pixels[y * width * 3:(y + 1) * width * 3]
                   for y in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output file")
    images = [png(code) for code in CODES]
    content = [{"type": "text", "text": "Read the four-digit number in each image. Return the eight numbers in image order, with no explanation."}]
    content += [{"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(image).decode()}}
                for image in images]
    result = stream(args.base, {
        "model": args.model, "messages": [{"role": "user", "content": content}],
        "temperature": 0, "max_tokens": 512, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": "low"},
    })
    positions = [result["content"].find(code) for code in CODES]
    result.update(model=args.model, base=args.base, image_count=8,
                  dimensions=[2560, 960], expected=CODES,
                  image_sha256=[hashlib.sha256(image).hexdigest() for image in images])
    result["passed"] = (all(position >= 0 for position in positions)
                        and positions == sorted(positions)
                        and result["finish_reason"] == "stop")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("passed", "content", "elapsed_s", "usage")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
