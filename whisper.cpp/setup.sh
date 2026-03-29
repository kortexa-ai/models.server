#!/usr/bin/env bash
set -euo pipefail

# Fetch Whisper models into the local whisper.cpp checkout. No Python env here.

WHISPER_SRC="${WHISPER_SRC:-"$HOME/src/whisper.cpp"}"
GGML_DL_SH="$WHISPER_SRC/models/download-ggml-model.sh"
VAD_DL_SH="$WHISPER_SRC/models/download-vad-model.sh"

MODELS=(
  "large-v3-turbo"
  "base.en"
)
VAD_MODEL="silero-v5.1.2"

echo "[setup] whisper.cpp path: $WHISPER_SRC"

if [[ ! -d "$WHISPER_SRC" ]]; then
  echo "[setup] Error: whisper.cpp not found at: $WHISPER_SRC" >&2
  echo "[setup] Set WHISPER_SRC to your local clone (e.g. export WHISPER_SRC=~/src/whisper.cpp)." >&2
  exit 1
fi

if [[ ! -x "$GGML_DL_SH" ]]; then
  echo "[setup] Error: missing script: $GGML_DL_SH" >&2
  exit 1
fi

if [[ ! -x "$VAD_DL_SH" ]]; then
  echo "[setup] Error: missing script: $VAD_DL_SH" >&2
  exit 1
fi

echo "[setup] Downloading Whisper models to: $WHISPER_SRC/models"
for model in "${MODELS[@]}"; do
  echo "[setup] -> $model"
  bash "$GGML_DL_SH" "$model"
done

echo "[setup] Downloading VAD model to: $WHISPER_SRC/models"
echo "[setup] -> $VAD_MODEL"
bash "$VAD_DL_SH" "$VAD_MODEL"

echo "[setup] Done. Models are in: $WHISPER_SRC/models"
echo "[setup] - ggml-large-v3-turbo.bin"
echo "[setup] - ggml-base.en.bin"
echo "[setup] - ggml-silero-v5.1.2.bin"
