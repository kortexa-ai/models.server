#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="ggml-cuda-builder"
# Build the Docker image if it doesn't exist
if ! docker images | awk '{print $1":"$2}' | grep -q "^${IMAGE_NAME}:"; then
    echo "Building Docker image ($IMAGE_NAME)..."
    docker build -f "$SCRIPT_DIR/Dockerfile.cuda-build" -t "$IMAGE_NAME" "$SCRIPT_DIR"
fi

# Run the build inside Docker container
echo "Running build in Docker container..."
if ! docker run --rm \
    --gpus all \
    --user "$(id -u)":"$(id -g)" \
    -e HOME=/tmp \
    -v "$HOME/src/llama.cpp:/workspace/llama.cpp" \
    -v "$HOME/bin:/workspace/bin" \
    -w /workspace/llama.cpp \
    "$IMAGE_NAME" \
    bash -c "
        echo 'Building llama.cpp with CUDA support inside Docker...'
        git config --global --add safe.directory /workspace/llama.cpp
        if [ -f build/CMakeCache.txt ]; then
            cache_src=\$(grep -m1 '^CMAKE_HOME_DIRECTORY:' build/CMakeCache.txt | cut -d= -f2-)
            if [ \"\$cache_src\" != \"/workspace/llama.cpp\" ]; then
                echo \"Clearing stale build directory (was configured for \\$cache_src)\"
                rm -rf build
            fi
        fi
        # Auto-detect GPU architecture inside container
        GPU_ARCH=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
        if [ -z \"\$GPU_ARCH\" ]; then
            echo 'Could not detect GPU architecture, using native'
            GPU_ARCH=native
        fi
        echo \"Building for CUDA architecture: \$GPU_ARCH\"
        cmake -B build -DGGML_CUDA=ON -DLLAMA_OPENSSL=ON -DCMAKE_CUDA_ARCHITECTURES=\$GPU_ARCH -DBUILD_SHARED_LIBS=OFF
        # Scale jobs to container memory (1 per GB, min 1, max 8)
        TOTAL_RAM_GB=\$(((\$(grep MemTotal /proc/meminfo | awk '{print \$2}') / 1024 / 1024)))
        JOBS=\$((\$TOTAL_RAM_GB > 1 ? \$TOTAL_RAM_GB - 1 : 1))
        JOBS=\$((\$JOBS > 8 ? 8 : \$JOBS))
        echo \"Building with -j \$JOBS (\${TOTAL_RAM_GB}GB RAM)\"
        if ! cmake --build build --config Release -j \$JOBS; then
            echo 'Build failed!'
            exit 1
        fi
        
        echo 'Build complete!'
    "; then
    echo "Docker build failed!"
    exit 1
fi

# Create and verify symbolic links (same as in build-llama.sh)
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
