#!/usr/bin/env python3
"""Standalone file transcription with Photon; independent of asr.server."""

import argparse
import logging
import tempfile
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


def create_app(model_id, alias, device="cpu", threads=8, model_factory=None):
    if model_factory is None:
        import moondream

        model_factory = moondream.photon

    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        options = {"device": device}
        if device == "cpu":
            options["cpu_threads"] = threads
        with model_factory(model_id, **options) as speech:
            app.state.speech = speech
            yield

    app = FastAPI(title="Photon ASR", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": alias, "device": device}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{
            "id": alias, "object": "model", "created": 0,
            "owned_by": "moondream", "type": "transcription",
            "input_modalities": ["audio"], "output_modalities": ["text"],
            "endpoints": ["/v1/audio/transcriptions"],
            "capabilities": {"streaming": False, "word_timestamps": True},
        }]}

    def transcribe_file(upload, timestamps):
        # Own the file and lock inside the worker, including when a caller leaves.
        with tempfile.NamedTemporaryFile(suffix=".audio") as audio:
            size = 0
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Audio must be at most 25 MiB")
                audio.write(chunk)
            if size == 0:
                raise HTTPException(400, "Audio file is empty")
            audio.flush()
            with lock:
                return app.state.speech.transcribe(audio=audio.name, timestamps=timestamps)

    @app.post("/v1/audio/transcriptions")
    async def transcribe(request: Request):
        async with request.form(max_files=1, max_fields=8) as form:
            allowed = {"file", "model", "response_format", "timestamp_granularities[]"}
            if set(form) - allowed:
                raise HTTPException(400, "Supported fields: file, model, response_format, timestamp_granularities[]")
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(400, "A multipart audio file is required")
            if upload.size is not None and upload.size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Audio must be at most 25 MiB")
            if form.get("model", alias) not in (alias, model_id):
                raise HTTPException(400, f"This server only serves {alias}")
            response_format = form.get("response_format", "json")
            if response_format not in ("json", "text", "verbose_json"):
                raise HTTPException(400, "response_format must be json, text, or verbose_json")
            granularities = form.getlist("timestamp_granularities[]")
            if any(value not in ("word", "segment") for value in granularities):
                raise HTTPException(400, "Timestamp granularity must be word or segment")
            if granularities and response_format != "verbose_json":
                raise HTTPException(400, "Timestamps require response_format=verbose_json")
            timestamps = "word" if "word" in granularities else (
                "segment" if response_format == "verbose_json" else "none"
            )
            try:
                result = await run_in_threadpool(transcribe_file, upload, timestamps)
            except HTTPException:
                raise
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, "Invalid or unsupported audio") from exc
            except Exception as exc:
                LOGGER.exception("Photon transcription failed")
                raise HTTPException(500, "Transcription failed") from exc

        if response_format == "text":
            return PlainTextResponse(result["text"])
        if response_format == "json":
            return {"text": result["text"]}
        output = {"task": "transcribe", **result}
        if "word" in granularities:
            output["words"] = [
                word for segment in result.get("segments", [])
                for word in segment.get("words", [])
            ]
        return output

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    import uvicorn

    uvicorn.run(create_app(args.model, args.alias, args.device, args.threads),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
