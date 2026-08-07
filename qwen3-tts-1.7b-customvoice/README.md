# Qwen3 TTS 1.7B CustomVoice

Served on port `2035` through the standard OpenAI-compatible
`POST /v1/audio/speech` endpoint.

- macOS: MLX-Audio
- Linux/CUDA: vLLM-Omni
- Model alias: `qwen3-tts-1.7b-customvoice`
- Output: 24 kHz mono

This is the larger CustomVoice checkpoint with the same nine preset speakers
as the 0.6B model. It adds natural-language instruction control for style,
emotion, and delivery. It does not provide arbitrary reference-audio cloning;
use a Base checkpoint for that workflow.

## Generate a file

```bash
curl http://localhost:2035/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts-1.7b-customvoice",
    "input": "Hello from the larger Qwen text to speech model.",
    "voice": "ryan",
    "instructions": "Speak warmly and conversationally.",
    "response_format": "wav"
  }' \
  --output qwen3-tts-1.7b.wav
```

The checkpoint includes preset speakers such as `ryan`, `aiden`, `vivian`,
and `serena`.

## Stream audio

Request raw PCM chunks, then wrap them in a WAV container if desired:

```bash
curl --no-buffer http://localhost:2035/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts-1.7b-customvoice",
    "input": "This audio starts arriving before the sentence is finished.",
    "voice": "ryan",
    "instructions": "Sound energetic and optimistic.",
    "response_format": "pcm",
    "stream": true,
    "stream_format": "audio"
  }' \
  --output qwen3-tts-1.7b.pcm

ffmpeg -f s16le -ar 24000 -ac 1 \
  -i qwen3-tts-1.7b.pcm qwen3-tts-1.7b-stream.wav
```

On CUDA, vLLM-Omni also supports OpenAI speech SSE streaming and incremental
text input through `/v1/audio/speech/stream`.
