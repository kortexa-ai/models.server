# Qwen3 TTS 0.6B CustomVoice

Served on port `2033` through the standard OpenAI-compatible
`POST /v1/audio/speech` endpoint.

- macOS: MLX-Audio
- Linux/CUDA: vLLM-Omni
- Model alias: `qwen3-tts-0.6b-customvoice`
- Output: 24 kHz mono

## Generate a file

```bash
curl http://localhost:2033/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts-0.6b-customvoice",
    "input": "Hello from Qwen text to speech.",
    "voice": "ryan",
    "response_format": "wav"
  }' \
  --output qwen3-tts.wav
```

The checkpoint includes preset speakers such as `ryan`, `aiden`, `vivian`,
and `serena`.

## Stream audio

Request raw PCM chunks, then wrap them in a WAV container if desired:

```bash
curl --no-buffer http://localhost:2033/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts-0.6b-customvoice",
    "input": "This audio starts arriving before the sentence is finished.",
    "voice": "ryan",
    "response_format": "pcm",
    "stream": true,
    "stream_format": "audio"
  }' \
  --output qwen3-tts.pcm

ffmpeg -f s16le -ar 24000 -ac 1 \
  -i qwen3-tts.pcm qwen3-tts-stream.wav
```

On CUDA, vLLM-Omni also supports OpenAI speech SSE streaming and incremental
text input through `/v1/audio/speech/stream`.
