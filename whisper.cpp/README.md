**Overview**
- **Goal:** Build and run ggml-org/whisper.cpp with large-v3-turbo on macOS (CoreML) and Ubuntu 24 (CUDA), without Docker.
- **Location assumptions:** whisper.cpp sources live at `~/src/whisper.cpp`. These scripts live here and call into that repo.

**What’s Included**
- `build-whisper.sh`: OS-aware build. Downloads `large-v3-turbo`; builds with CUDA on Ubuntu, CoreML on macOS; converts model to CoreML on macOS.
- `setup.sh` (macOS): Sets up `.venv` with `uv` and installs `ane_transformers`, `openai-whisper`, `coremltools` for CoreML conversion. No-op on Ubuntu.
- `install.sh`: Installs shortcuts into `~/bin` (symlinks to the built binary and `run-large.sh`).
- `run-large.sh`: Runs the built whisper binary against an audio file. On macOS, prefers CoreML model if present.

**Prerequisites**
- Common: `cmake`, C/C++ toolchain, Python 3.10+.
- Ubuntu 24 + NVIDIA 4090: CUDA drivers/toolkit installed and working. The build uses `-DGGML_CUDA=1`.
- macOS (Apple Silicon e.g. Mac Mini M4 Pro): Xcode Command Line Tools; CoreML used via `-DWHISPER_COREML=1`.
- `uv` (Python tool) in `PATH`. The installer in `install.sh` attempts to install it if missing.

**Environment Variables**
- `BUILD_DIR`: CMake build directory. Default: `~/src/whisper.cpp/build`.
- `VENV_DIR`: Python virtualenv for conversion. Default: `.venv` under this folder.
- `CUDA`: Set `1` to enable CUDA on Linux (default), `0` for CPU-only.
- `CUDA_ARCHS`: CUDA SM architectures to target (auto-detected; default `89` for RTX 4090/Ada).
- `CUDA_HOST_COMPILER`: Path to a specific `g++` for NVCC host compilation (e.g. `/usr/bin/g++-13`).

**Setup**
- macOS only: Install Python deps (creates `.venv`) for CoreML conversion
  - `./setup.sh`
  - Ubuntu: no action needed.

**Build**
- Build and fetch `large-v3-turbo`:
  - `./build-whisper.sh`
  - On Ubuntu 25+, the script automatically uses Docker for CUDA builds.
    - Set `NO_DOCKER=1` to force a native build attempt.
- This performs:
  - Model download via `whisper.cpp/models/download-ggml-model.sh large-v3-turbo`.
  - Ubuntu: `cmake -DGGML_CUDA=1` + build.
  - macOS: `cmake -DWHISPER_COREML=1` + build, then CoreML conversion via `whisper.cpp/models/generate-coreml-model.sh large-v3-turbo` using the `.venv`.

**Run**
- Basic usage:
  - `./run-large.sh path/to/audio.wav [extra whisper args]`
- Behavior:
  - macOS: If a CoreML `.mlpackage` for `large-v3-turbo` exists under `~/src/whisper.cpp/models`, the script uses it automatically; otherwise it falls back to the GGML `.bin`.
  - Ubuntu: Uses the GGML `.bin`.
- The script auto-locates a typical `main`/`whisper-cli` binary inside the CMake build output and passes `-m` with the chosen model path.

**Server**
- Start the HTTP server (normalized to llama-style flags):
  - `./run-server.sh [--host 0.0.0.0] [--port 8080] [--model /path/to/model] [extra args...]`
- Behavior:
  - macOS: Prefers a CoreML `.mlpackage` named `ggml-large-v3-turbo.mlpackage` if present; otherwise uses `ggml-large-v3-turbo.bin`.
  - Linux: Uses `ggml-large-v3-turbo.bin`.
- The script locates the `whisper-server` binary under CMake `build/` and passes `-m/--host/--port` for you. Use `--convert` to enable ffmpeg input conversion.

**Install Shortcuts**
- Create symlinks in `~/bin`:
  - `./install.sh`
- After this, you can run:
  - `whisper -h` (direct binary)
  - `run-whisper-large path/to/audio.wav` (uses ggml/coreml automatically on macOS)
  - `run-whisper-server --host 0.0.0.0 --port 8080` (serves HTTP API)

**Notes**
- CoreML: The conversion runs on macOS to produce a CoreML package from `large-v3-turbo`. The default `run-large.sh` uses the GGML model; you can adapt it to point at the generated CoreML package if desired once present.
- CUDA: Ensure your NVIDIA drivers and CUDA toolkit match your kernel and userland. The build only toggles `-DGGML_CUDA=1` and relies on your environment.
- Paths: This script assumes your whisper.cpp checkout is at `~/src/whisper.cpp`.

**Troubleshooting (Ubuntu 24/25 + CUDA)**
- NVCC vs GCC mismatch (e.g. GCC 14 on Ubuntu 25): install an older host compiler and point NVCC to it:
  - `sudo apt install g++-13` (or `g++-12`)
  - `CUDA_HOST_COMPILER=/usr/bin/g++-13 ./build-whisper.sh`
- Add explicit Ada arch for RTX 4090 to avoid fatbin issues:
  - `CUDA_ARCHS=89 ./build-whisper.sh`
- Ensure CUDA toolkit in `PATH` and `LD_LIBRARY_PATH` (adjust version as needed):
  - `export PATH=/usr/local/cuda/bin:$PATH`
  - `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`
- Verify NVCC works: `nvcc --version`
- Fall back to CPU-only to validate the rest of the toolchain:
  - `CUDA=0 ./build-whisper.sh`

**Docker Build (recommended for Ubuntu 25)**
- Build inside the shared GGML CUDA image `ggml-cuda-builder` (reused by llama.cpp):
  - `./docker-build-whisper.sh`
- This mounts `~/src/whisper.cpp` and builds with `-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_ARCHITECTURES=89` inside `nvidia/cuda:12.9-devel-ubuntu24.04`.
- Container runs as your host UID/GID to avoid root-owned files; it then links `whisper-cli`/`whisper-server` into `~/bin`.

If your `whisper.cpp` repo is not in `~/src/whisper.cpp`, move or symlink it there.

**Examples**
- macOS:
  - `./setup.sh && ./build-whisper.sh`
  - `./install.sh`
  - `run-whisper-large samples/jfk.wav`
- Ubuntu 24 + 4090:
  - `./build-whisper.sh`
  - `./install.sh`
  - `run-whisper-large ~/audio/test.wav -t 8`
