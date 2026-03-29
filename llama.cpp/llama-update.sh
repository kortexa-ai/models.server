#!/bin/bash

# Update and rebuild llama.cpp (and optionally whisper.cpp if present)

set -e

LLAMA_DIR="$HOME/src/llama.cpp"
WHISPER_DIR="$HOME/src/whisper.cpp"
LLAMA_REPO="https://github.com/ggerganov/llama.cpp.git"

echo "=== Updating llama.cpp ==="
if [ -d "$LLAMA_DIR" ]; then
    echo "Pulling latest changes..."
    cd "$LLAMA_DIR"
    git pull
else
    echo "Cloning llama.cpp..."
    git clone "$LLAMA_REPO" "$LLAMA_DIR"
fi

echo ""
echo "=== Building llama.cpp ==="
build-llama.sh

# whisper.cpp is optional — update and rebuild if present, warn if not
echo ""
if [ -d "$WHISPER_DIR" ]; then
    echo "=== Updating whisper.cpp ==="
    cd "$WHISPER_DIR"
    git pull
    echo ""
    echo "=== Building whisper.cpp ==="
    build-whisper.sh
else
    echo "=== whisper.cpp not found at $WHISPER_DIR (skipping — clone it manually if needed) ==="
fi

echo ""
echo "=== All done! ==="
