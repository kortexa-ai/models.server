#!/bin/bash

# Update and rebuild llama.cpp and whisper.cpp

set -e

LLAMA_DIR="$HOME/src/llama.cpp"
WHISPER_DIR="$HOME/src/whisper.cpp"
LLAMA_REPO="https://github.com/ggerganov/llama.cpp.git"
WHISPER_REPO="https://github.com/ggerganov/whisper.cpp.git"

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
echo "=== Updating whisper.cpp ==="
if [ -d "$WHISPER_DIR" ]; then
    echo "Pulling latest changes..."
    cd "$WHISPER_DIR"
    git pull
else
    echo "Cloning whisper.cpp..."
    git clone "$WHISPER_REPO" "$WHISPER_DIR"
fi

echo ""
echo "=== Building llama.cpp ==="
build-llama.sh

echo ""
echo "=== Building whisper.cpp ==="
build-whisper.sh

echo ""
echo "=== All done! ==="
