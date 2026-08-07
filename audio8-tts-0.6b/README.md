# Audio8 TTS Preview 0.6B

Served on port `2034` through the standard OpenAI-compatible
`POST /v1/audio/speech` endpoint.

- macOS: MLX-Audio
- Linux/CUDA: Audio8's SGLang-Omni adapter
- Model alias: `audio8-tts-0.6b`
- Output: 44.1 kHz mono

## Generate a file

```bash
curl http://localhost:2034/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "audio8-tts-0.6b",
    "input": "Hello from Audio8 text to speech.",
    "response_format": "wav"
  }' \
  --output audio8-tts.wav
```

Audio8 supports zero-shot cloning. On MLX, pass server-local `ref_audio` and
matching `ref_text` fields. On CUDA, pass the same material through the
SGLang-Omni `references` array.

## Stream audio

Incremental output is currently supported by the CUDA SGLang-Omni backend.
It returns SSE events containing base64-encoded PCM chunks:

```python
import base64
import json
import requests
import wave

payload = {
    "model": "audio8-tts-0.6b",
    "input": "Audio8 streams this sentence as it is generated.",
    "response_format": "pcm",
    "stream": True,
}

chunks = []
sample_rate = 44100
with requests.post(
    "http://localhost:2034/v1/audio/speech",
    json=payload,
    stream=True,
    timeout=300,
) as response:
    response.raise_for_status()
    for line in response.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        value = line[6:]
        if value == b"[DONE]":
            break
        event = json.loads(value)
        audio = event.get("audio")
        if audio:
            sample_rate = audio["sample_rate"]
            chunks.append(base64.b64decode(audio["data"]))

with wave.open("audio8-tts-stream.wav", "wb") as output:
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(sample_rate)
    output.writeframes(b"".join(chunks))
```

The current MLX `arktts` implementation returns a completed clip rather than
incremental chunks.
