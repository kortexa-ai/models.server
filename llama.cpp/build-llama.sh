#!/bin/bash

# Avoid CDPATH surprises
CDPATH=

# Check if ~/src/llama.cpp exists
if [ ! -d "$HOME/src/llama.cpp" ]; then
    echo "$HOME/src/llama.cpp doesn't exist"
    exit 1
fi

# Detect the operating system
OS=$(uname -s)
ARCH=$(uname -m)

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"

# Initialize variables
CMAKE_FLAGS=""

# Determine platform-specific settings
if [ "$OS" = "Linux" ]; then
    # Check if running on Raspberry Pi (32-bit ARM only)
    if [ "$ARCH" = "armv7l" ]; then
        echo "Detected Raspberry Pi (ARM 32-bit)"
        # Pi CPU - no GPU support
    else
        echo "Detected Linux (Ubuntu-like)"

        # Install dependencies if not present
        if ! dpkg -s build-essential cmake libcurl4-openssl-dev libssl-dev ccache >/dev/null 2>&1; then
            echo "Installing dependencies..."
            sudo apt update
            sudo apt install -y build-essential cmake libcurl4-openssl-dev libssl-dev ccache
        fi

        # Detect glibc / Ubuntu version to decide on Docker path early
        GLIBC_VERSION=$(ldd --version | head -1 | awk '{print $NF}')
        UBUNTU_VERSION=$(grep -oP 'VERSION_ID="?([0-9]{2}\.[0-9]{2})"?' /etc/os-release 2>/dev/null | head -1 | sed 's/[^0-9.]//g')
        should_use_docker=false
        if [ -n "$UBUNTU_VERSION" ] && echo "$UBUNTU_VERSION" | grep -q '^25\.'; then
            should_use_docker=true
        elif command_exists dpkg && dpkg --compare-versions "$GLIBC_VERSION" ge 2.41; then
            should_use_docker=true
        elif [ "$GLIBC_VERSION" = "2.41" ]; then
            should_use_docker=true
        fi

        # Check for NVIDIA CUDA
        if command_exists nvidia-smi; then
            echo "Detected NVIDIA GPU, enabling CUDA"

            # Install CUDA toolkit if nvcc is not present
            if ! command_exists nvcc; then
                echo "Installing CUDA toolkit..."
                sudo apt install -y nvidia-cuda-toolkit
            fi
            
            # Build for both GPUs used on smarty. Keep the explicit list so an
            # eGPU disconnect during configuration does not drop Ada support.
            GPU_ARCH="89;120"
            echo "Building for CUDA architectures: $GPU_ARCH (Ada and Blackwell)"

            if [ "$should_use_docker" = true ]; then
                echo "Detected Ubuntu 25.x or glibc >= 2.41 - using Docker build to avoid CUDA compatibility issues"
                exec "$SCRIPT_DIR/docker-build-llama.sh"
                exit 0
            fi

            # Standard CUDA build for compatible glibc versions
            # NCCL is only for multi-GPU collective comms; our machines are all single-GPU,
            # so disable it (also silences the "NCCL not found" cmake configure warning).
            CMAKE_FLAGS="-DGGML_CUDA=ON -DLLAMA_OPENSSL=ON -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA_NCCL=OFF"
        else
            # CPU-only Linux (e.g. Raspberry Pi aarch64)
            CMAKE_FLAGS="-DLLAMA_OPENSSL=ON -DBUILD_SHARED_LIBS=OFF"
        fi
    fi
elif [ "$OS" = "Darwin" ]; then
    echo "Detected macOS"
    # macOS with Metal
    CMAKE_FLAGS="-DGGML_METAL=ON -DLLAMA_OPENSSL=ON"
else
    echo "Unsupported operating system: $OS"
    exit 1
fi

# Determine parallel jobs based on available RAM
# Each llama.cpp compile job can use ~500MB+, so cap at RAM_GB - 1 (leave 1GB for OS)
if [ "$OS" = "Linux" ]; then
    TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))
elif [ "$OS" = "Darwin" ]; then
    TOTAL_RAM_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
else
    TOTAL_RAM_GB=8
fi
# 1 job per 1GB of RAM, minimum 1, maximum 8
JOBS=$((TOTAL_RAM_GB > 1 ? TOTAL_RAM_GB - 1 : 1))
JOBS=$((JOBS > 8 ? 8 : JOBS))
echo "Building with -j $JOBS (${TOTAL_RAM_GB}GB RAM detected)"

# Build the project
cd "$HOME/src/llama.cpp" || exit 1

# Clear stale build dir if it was generated from a different source path (e.g., Docker)
if [ -f build/CMakeCache.txt ]; then
    CACHE_SRC=$(grep -m1 '^CMAKE_HOME_DIRECTORY:' build/CMakeCache.txt | cut -d= -f2-)
    if [ "$CACHE_SRC" != "$(pwd)" ]; then
        echo "Clearing stale build directory (was configured for $CACHE_SRC)"
        rm -rf build
    fi
fi

echo "Running cmake with flags: $CMAKE_FLAGS"
if ! cmake -B build $CMAKE_FLAGS; then
    echo "CMake configure failed on host toolchain. Falling back to Docker build..."
    rm -rf build
    exec "$SCRIPT_DIR/docker-build-llama.sh"
fi

if ! cmake --build build --config Release -j "$JOBS"; then
    echo "Build failed on host toolchain. Falling back to Docker build..."
    rm -rf build
    exec "$SCRIPT_DIR/docker-build-llama.sh"
fi

# Create and verify symbolic links
LINK_FILES="llama-cli llama-mtmd-cli llama-server"
LINK_PREFIX="$HOME/src/llama.cpp/build/bin"

echo "Creating symbolic links..."
for link in $LINK_FILES; do
    ln -sf "$LINK_PREFIX/$link" "$HOME/bin/$link"
done

echo "Verifying symbolic links..."
for link in $LINK_FILES; do
    if [ -L "$HOME/bin/$link" ] && [ -e "$HOME/bin/$link" ]; then
        target=$(readlink -f "$HOME/bin/$link")
        expected="$LINK_PREFIX/$link"
        if [ "$target" = "$expected" ]; then
            echo "Symbolic link $link created successfully, pointing to $target"
        else
            echo "Error: Symbolic link $link points to $target, expected $expected"
            exit 1
        fi
    else
        echo "Error: Symbolic link $link was not created or is broken"
        exit 1
    fi
done

echo "Build and setup completed successfully!"
