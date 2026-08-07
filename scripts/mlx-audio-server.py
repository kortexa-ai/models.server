#!/usr/bin/env python3
"""Run one pinned mlx-audio model behind a stable local model alias."""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--max-batch-size", default=1, type=int)
    parser.add_argument("--sample-rate", default=0, type=int)
    parser.add_argument("--default-voice", default="")
    parser.add_argument("--voices", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["MLX_AUDIO_TTS_MAX_BATCH_SIZE"] = str(args.max_batch_size)

    import uvicorn
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from mlx_audio.server import app, model_provider

    model = model_provider.load_model(args.model)
    model_provider.models[args.alias] = model
    if args.alias != args.model:
        del model_provider.models[args.model]

    voices = [voice for voice in args.voices.split(",") if voice]

    @app.middleware("http")
    async def normalize_openai_speech_request(request: Request, call_next):
        if request.method == "POST" and request.url.path == "/v1/audio/speech":
            body = await request.body()
            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = None
            if isinstance(payload, dict):
                if "instructions" in payload and "instruct" not in payload:
                    payload["instruct"] = payload["instructions"]
                payload.pop("stream_format", None)
                normalized = json.dumps(payload).encode("utf-8")

                async def receive():
                    return {
                        "type": "http.request",
                        "body": normalized,
                        "more_body": False,
                    }

                request = Request(request.scope, receive)

        response = await call_next(request)
        if request.url.path == "/v1/audio/speech" and args.sample_rate:
            response.headers["X-Sample-Rate"] = str(args.sample_rate)
        return response

    @app.get("/health")
    async def health():
        return {
            "ready": True,
            "model": args.alias,
            "sample_rate": args.sample_rate,
        }

    @app.get("/v1/voices")
    async def compatibility_voices():
        return JSONResponse(
            {
                "object": "list",
                "default_voice": args.default_voice or None,
                "data": [{"id": voice} for voice in voices],
            }
        )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
