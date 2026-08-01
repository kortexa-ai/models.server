#!/usr/bin/env python3
"""Small CPU HTTP server for Transformers tasks unsupported by llama.cpp."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class FillMaskModel:
    def __init__(self, model_id, max_length, threads, trust_remote_code):
        torch.set_num_threads(threads)
        self.max_length = max_length
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        ).to("cpu").eval()

        if self.tokenizer.mask_token_id is None:
            raise ValueError(f"{model_id} does not define a mask token")

    def predict(self, text, top_k):
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"]
        mask_positions = (input_ids[0] == self.tokenizer.mask_token_id).nonzero(
            as_tuple=False
        ).flatten()

        if len(mask_positions) != 1:
            raise ValueError(
                f"input must contain exactly one {self.tokenizer.mask_token!r} token"
            )

        mask_position = mask_positions[0].item()
        with self.lock, torch.inference_mode():
            logits = self.model(**encoded).logits[0, mask_position]
            probabilities = torch.softmax(logits, dim=-1)
            scores, token_ids = probabilities.topk(min(top_k, len(probabilities)))

        predictions = []
        for score, token_id in zip(scores.tolist(), token_ids.tolist()):
            completed = input_ids[0].clone()
            completed[mask_position] = token_id
            predictions.append(
                {
                    "score": score,
                    "token": token_id,
                    "token_str": self.tokenizer.decode([token_id]),
                    "sequence": self.tokenizer.decode(
                        completed,
                        skip_special_tokens=True,
                    ),
                }
            )
        return predictions


def make_handler(model, alias, default_top_k):
    class Handler(BaseHTTPRequestHandler):
        server_version = "models.server-transformers/1"

        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                self.send_json(200, {"status": "ok", "model": alias})
            elif path == "/v1/models":
                self.send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": alias,
                                "object": "model",
                                "owned_by": "LiquidAI",
                            }
                        ],
                    },
                )
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/v1/fill-mask":
                self.send_json(404, {"error": "not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 1_048_576:
                    raise ValueError("request body must be between 1 byte and 1 MiB")
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                text = payload.get("input")
                if not isinstance(text, str) or not text:
                    raise ValueError("input must be a non-empty string")
                top_k = payload.get("top_k", default_top_k)
                if not isinstance(top_k, int) or not 1 <= top_k <= 100:
                    raise ValueError("top_k must be an integer from 1 to 100")

                predictions = model.predict(text, top_k)
                self.send_json(
                    200,
                    {
                        "model": alias,
                        "object": "fill_mask",
                        "data": predictions,
                    },
                )
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                self.log_error("inference failed: %s", exc)
                self.send_json(500, {"error": "inference failed"})

    return Handler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--task", choices=("fill-mask",), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model = FillMaskModel(
        args.model,
        args.max_length,
        args.threads,
        args.trust_remote_code,
    )
    handler = make_handler(model, args.alias, args.top_k)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {args.alias} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
