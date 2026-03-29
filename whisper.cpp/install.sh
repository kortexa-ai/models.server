#!/usr/bin/env bash
set -euo pipefail

# Install shortcuts to whisper binaries/scripts into ~/bin (like ../llama.cpp/install.sh)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_SRC="${WHISPER_SRC:-$HOME/src/whisper.cpp}"
BUILD_DIR="${BUILD_DIR:-$WHISPER_SRC/build}"
INSTALL_BIN_DIR="${INSTALL_BIN_DIR:-$HOME/bin}"

mkdir -p "$INSTALL_BIN_DIR"

echo "[install] Installing shortcuts into: $INSTALL_BIN_DIR"

# Always link build script, regardless of binaries being present
ln -sfn "$SCRIPT_DIR/build-whisper.sh" "$INSTALL_BIN_DIR/build-whisper.sh"
echo "[install] Linked: build-whisper.sh -> $SCRIPT_DIR/build-whisper.sh"

# Optionally link binaries if they exist
if [[ -x "$BUILD_DIR/bin/whisper-cli" ]]; then
  ln -sfn "$BUILD_DIR/bin/whisper-cli" "$INSTALL_BIN_DIR/whisper-cli"
  echo "[install] Linked: whisper-cli -> $BUILD_DIR/bin/whisper-cli"
else
  echo "[install] Note: whisper-cli not found under $BUILD_DIR/bin (run ./build-whisper.sh to build)"
fi

if [[ -x "$BUILD_DIR/bin/whisper-server" ]]; then
  ln -sfn "$BUILD_DIR/bin/whisper-server" "$INSTALL_BIN_DIR/whisper-server"
  echo "[install] Linked: whisper-server -> $BUILD_DIR/bin/whisper-server"
fi

echo "[install] Done. Ensure ~/bin is in your PATH. Example:"
echo '  export PATH="$HOME/bin:$PATH"'
