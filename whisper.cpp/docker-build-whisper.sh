#!/usr/bin/env bash
set -euo pipefail

# Build whisper.cpp with CUDA inside a Docker container (mirrors ../llama.cpp approach).

SCRIPT_REAL="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_REAL}")" && pwd)"
WHISPER_SRC="$HOME/src/whisper.cpp"
BIN_DIR="$HOME/bin"

if [[ ! -d "$WHISPER_SRC" ]]; then
  echo "[docker-build] whisper.cpp not found at $WHISPER_SRC" >&2
  echo "[docker-build] Clone it there or adjust this script." >&2
  exit 1
fi

IMAGE_NAME="ggml-cuda-builder"
if ! docker images | awk '{print $1":"$2}' | grep -q "^${IMAGE_NAME}:"; then
  echo "[docker-build] Building shared GGML CUDA image: $IMAGE_NAME"
  # Prefer llama.cpp Dockerfile if present (it includes libcurl dev)
  if [[ -f "$SCRIPT_DIR/../llama.cpp/Dockerfile.cuda-build" ]]; then
    docker build -f "$SCRIPT_DIR/../llama.cpp/Dockerfile.cuda-build" -t "$IMAGE_NAME" "$SCRIPT_DIR/../llama.cpp"
  else
    docker build -f "$SCRIPT_DIR/Dockerfile.cuda-build" -t "$IMAGE_NAME" "$SCRIPT_DIR"
  fi
fi

echo "[docker-build] Running build in Docker ($IMAGE_NAME) ..."
docker run --rm \
  --gpus all \
  --user "$(id -u)":"$(id -g)" \
  -e HOME=/tmp \
  -v "$WHISPER_SRC:/workspace/whisper.cpp" \
  -v "$BIN_DIR:/workspace/bin" \
  -w /workspace/whisper.cpp \
  "$IMAGE_NAME" \
  bash -lc '
    set -euo pipefail
    echo "[container] Building whisper.cpp with CUDA support ..."
    git config --global --add safe.directory /workspace/whisper.cpp
    # Auto-detect GPU architecture inside container
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ".")
    if [ -z "$GPU_ARCH" ]; then
        echo "[container] Could not detect GPU architecture, using native"
        GPU_ARCH=native
    fi
    echo "[container] Building for CUDA architecture: $GPU_ARCH"
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
    cmake --build build --config Release -j "$(nproc)"
    echo "[container] Build complete"
  '

echo "[docker-build] Linking binaries into $BIN_DIR ..."
mkdir -p "$BIN_DIR"
if [[ -x "$WHISPER_SRC/build/bin/whisper-cli" ]]; then
  ln -sfn "$WHISPER_SRC/build/bin/whisper-cli" "$BIN_DIR/whisper-cli"
  echo "[docker-build] Linked: whisper-cli"
fi
if [[ -x "$WHISPER_SRC/build/bin/whisper-server" ]]; then
  ln -sfn "$WHISPER_SRC/build/bin/whisper-server" "$BIN_DIR/whisper-server"
  echo "[docker-build] Linked: whisper-server"
fi

echo "[docker-build] Done. Ensure ~/bin is on PATH (export PATH=\"$HOME/bin:$PATH\")."
