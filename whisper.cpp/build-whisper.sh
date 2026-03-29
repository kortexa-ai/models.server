#!/usr/bin/env bash
set -euo pipefail

# Build whisper.cpp with OS-specific accelerators and fetch the large-v3-turbo model.
# - Linux (Ubuntu): builds with CUDA via -DGGML_CUDA=1
# - macOS: build with BLAS for now; skip CoreML conversion (can be enabled later)

SCRIPT_REAL="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_REAL}")" && pwd)"
WHISPER_SRC="$HOME/src/whisper.cpp"
BUILD_DIR="${BUILD_DIR:-$WHISPER_SRC/build}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"

# CUDA controls (Linux):
# - Set CUDA=0 to force CPU build
# - Override archs via CUDA_ARCHS (default 89 for RTX 4090 Ada)
# - Optionally set CUDA_HOST_COMPILER to a specific g++ if nvcc dislikes the default
CUDA="${CUDA:-1}"
CUDA_ARCHS="${CUDA_ARCHS:-}"
CUDA_HOST_COMPILER="${CUDA_HOST_COMPILER:-}"
FALLBACK_CPU="${FALLBACK_CPU:-1}"

CPU_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu-only) CPU_ONLY=1; CUDA=0; shift ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --arch) CUDA_ARCHS="$2"; shift 2 ;;
    --host-compiler) CUDA_HOST_COMPILER="$2"; shift 2 ;;
    --no-fallback) FALLBACK_CPU=0; shift ;;
    *) echo "[build] Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$WHISPER_SRC" ]]; then
  echo "[build] whisper.cpp source not found at: $WHISPER_SRC" >&2
  echo "[build] Set WHISPER_SRC to your local clone of whisper.cpp" >&2
  exit 1
fi

uname_s="$(uname -s)"
case "$uname_s" in
  Linux)  OS="linux" ;;
  Darwin) OS="mac" ;;
  *)      echo "[build] Unsupported OS: $uname_s" >&2 ; exit 1 ;;
esac

echo "[build] Source: $WHISPER_SRC"
echo "[build] Build dir: $BUILD_DIR"
echo "[build] OS: $OS"

# On Ubuntu 25+ with CUDA enabled, prefer Docker build automatically
if [[ "$OS" == "linux" && "${NO_DOCKER:-0}" != "1" && "$CUDA" != "0" && "$CPU_ONLY" != "1" ]]; then
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    if [[ "${ID:-}" == "ubuntu" ]]; then
      ver_major="${VERSION_ID%%.*}"
      if [[ -n "$ver_major" && "$ver_major" -ge 25 ]]; then
        if command -v docker >/dev/null 2>&1; then
          echo "[build] Detected Ubuntu ${VERSION_ID}. Using Docker CUDA build for compatibility."
          exec "$SCRIPT_DIR/docker-build-whisper.sh"
        else
          echo "[build] Ubuntu ${VERSION_ID} detected, but 'docker' not found."
          echo "[build] Install Docker (with NVIDIA runtime) or set NO_DOCKER=1 to force native build."
          echo "[build] Falling back to native CPU build for now."
          CUDA=0
          CPU_ONLY=1
        fi
      fi
    fi
  fi
fi

mkdir -p "$BUILD_DIR"

echo "[build] Configuring CMake"
cmake_args=(
  -B "$BUILD_DIR"
  -S "$WHISPER_SRC"
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF
)

if [[ "$OS" == "linux" ]]; then
  if [[ "$CPU_ONLY" == "1" || "$CUDA" == "0" ]]; then
    echo "[build] CUDA disabled; building CPU-only"
  else
    # Ensure nvcc is on PATH; if not, but /usr/local/cuda/bin exists, prepend it
    if ! command -v nvcc >/dev/null 2>&1 && [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
      export PATH="/usr/local/cuda/bin:$PATH"
      echo "[build] Added /usr/local/cuda/bin to PATH"
    fi
    if ! command -v nvcc >/dev/null 2>&1; then
      echo "[build] CUDA requested but 'nvcc' not found. Set CUDA=0 for CPU-only or install CUDA." >&2
      if [[ "$FALLBACK_CPU" == "1" ]]; then
        echo "[build] Falling back to CPU-only configure"
      else
        exit 1
      fi
    else
      echo "[build] Enabling CUDA backend (-DGGML_CUDA=1)"
      cmake_args+=( -DGGML_CUDA=1 )
      # Detect or set CUDA architectures
      # Use 'native' to let cmake auto-detect (handles Blackwell 121a suffix etc.)
      if [[ -z "$CUDA_ARCHS" ]]; then
        CUDA_ARCHS="native"
        if command -v nvidia-smi >/dev/null 2>&1; then
          cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)"
          echo "[build] Detected GPU compute capability: $cc (using native cmake detection)"
        fi
      fi
      cmake_args+=( -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" )

      # Work around GCC/glibc C++ noexcept prototypes conflicting with CUDA math headers on Ubuntu 25
      # See errors like: bits/mathcalls.h: error: exception specification is incompatible with previous function "cospi"
      NVCC_FIX_FLAGS=(
        "--allow-unsupported-compiler"
        "-Wno-deprecated-gpu-targets"
        "-D__THROW="
      )
      if [[ -n "${NVCC_FLAGS_EXTRA:-}" ]]; then
        # Allow user to extend/override flags
        # shellcheck disable=SC2206
        NVCC_FIX_FLAGS=( ${NVCC_FIX_FLAGS[*]} ${NVCC_FLAGS_EXTRA} )
      fi
      echo "[build] Adding NVCC flags: ${NVCC_FIX_FLAGS[*]}"
      cmake_args+=( -DCMAKE_CUDA_FLAGS="${NVCC_FIX_FLAGS[*]}" )
      # Choose a CUDA host compiler if system GCC is too new for NVCC
      if [[ -z "$CUDA_HOST_COMPILER" ]]; then
        sys_gpp="$(command -v g++ || true)"
        if [[ -n "$sys_gpp" ]]; then
          sys_ver_major="$($sys_gpp -dumpfullversion -dumpversion 2>/dev/null | cut -d. -f1)"
        else
          sys_ver_major=0
        fi
        if [[ "$sys_ver_major" -ge 14 ]]; then
          for cand in /usr/bin/g++-13 /usr/bin/g++-12 /usr/bin/g++-11; do
            if [[ -x "$cand" ]]; then CUDA_HOST_COMPILER="$cand"; break; fi
          done
        fi
      fi
      if [[ -n "$CUDA_HOST_COMPILER" ]]; then
        echo "[build] Using CUDA host compiler: $CUDA_HOST_COMPILER"
        cmake_args+=( -DCMAKE_CUDA_HOST_COMPILER="$CUDA_HOST_COMPILER" )
      fi
    fi
  fi
elif [[ "$OS" == "mac" ]]; then
  # Build with Metal (no CoreML) for Apple GPU acceleration
  cmake_args+=( -DCMAKE_OSX_ARCHITECTURES=arm64 -DGGML_METAL=1 )
fi

# Run configure, with optional CPU fallback if CUDA configure fails
if [[ "$OS" == "linux" && "$CUDA" == "1" && "$CPU_ONLY" == "0" ]]; then
  set +e
  cmake "${cmake_args[@]}"
  st=$?
  set -e
  if [[ $st -ne 0 ]]; then
    if [[ "$FALLBACK_CPU" == "1" ]]; then
      echo "[build] CUDA configure failed; retrying CPU-only"
      cmake_args=( -B "$BUILD_DIR" -S "$WHISPER_SRC" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=0 )
      cmake "${cmake_args[@]}"
      echo "[build] TIP: For a reliable CUDA build on Ubuntu 25, try: $SCRIPT_DIR/docker-build-whisper.sh"
    else
      echo "[build] CUDA configure failed and fallback disabled." >&2
      exit $st
    fi
  fi
else
  cmake "${cmake_args[@]}"
fi

echo "[build] Building"
cmake --build "$BUILD_DIR" --config Release -j

# NOTE: CoreML conversion skipped for now. To enable later, reintroduce the
# conversion step and compile the CoreML encoder with coremltools.

echo "[build] Done. Binaries in: $BUILD_DIR"
