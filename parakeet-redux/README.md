# Parakeet Redux

[Moondream's Parakeet Redux](https://huggingface.co/moondream/parakeet-redux)
is a 178 MB ternary speech recognition model for 25 European languages. It
runs locally through Photon (`moondream==2.4.0`) with no API key. Model weights
use CC-BY-4.0; retain the publisher's attribution when redistributing them.

## Run on demand

From `models.server` on macOS or Linux:

```bash
./scripts/setup-photon.sh       # once; requires uv and ffmpeg
./run.sh parakeet-redux         # CPU, 8 threads, port 2063
```

Setup installs an isolated `parakeet-redux/.venv/`. First startup downloads
the model into the Hugging Face cache. Linux installs CPU-only PyTorch; the
default launcher hides CUDA. Ctrl-C stops a foreground server and releases
the model. `HOST` and `PORT` work as with other model launchers.

For managed operation, install through `ktxsvc`, then disable its automatic
startup and crash restart. `ktxsvc install` starts the service immediately:

```bash
ktxsvc install models/parakeet-redux
ktxsvc disable models/parakeet-redux
ktxsvc keep-alive models/parakeet-redux off
ktxsvc stop models/parakeet-redux

ktxsvc start models/parakeet-redux
ktxsvc stop models/parakeet-redux
```

## Transcribe

```bash
curl http://localhost:2063/health
curl http://localhost:2063/v1/models
curl http://localhost:2063/v1/audio/transcriptions \
  -F model=parakeet-redux \
  -F file=@speech.wav

curl http://localhost:2063/v1/audio/transcriptions \
  -F file=@speech.wav \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=word'
```

The API accepts multipart audio files up to 25 MiB. FFmpeg decodes common
audio formats. `response_format` supports `json` (text only), `text`, and
`verbose_json` (segment timestamps, plus word timestamps when requested).
Language detection is automatic; language hints, prompts, translation, and
live streaming are not exposed by this server. Concurrent requests queue for
one inference worker. Photon segments long recordings at speech pauses.

Redux is independent of the Qwen/Nemotron modes in `asr.server`. Its declared
type, modalities, endpoint, and capabilities appear in the operations model
inventory and its own `/v1/models`. Use its direct port: `api.server` continues
to route public speech requests to `asr.server`, and excludes standalone
audio models from its public catalog just as it does the standalone TTS models.

## Validation

```bash
parakeet-redux/.venv/bin/python -m unittest discover -s tests
```
