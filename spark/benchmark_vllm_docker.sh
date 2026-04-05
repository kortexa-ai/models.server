#!/bin/bash
# Benchmark script for vLLM Docker on DGX Spark
# Usage: ./benchmark_vllm_docker.sh <model_name> [port]

MODEL="${1:-Qwen/Qwen3.5-4B}"
PORT="${2:-2242}"
CONTAINER_NAME="vllm-bench-$(echo $MODEL | tr '/-' '--')"

echo "=== Benchmarking $MODEL on port $PORT ==="
echo "Container: $CONTAINER_NAME"
echo ""

# Kill any existing container with this name
docker rm -f $CONTAINER_NAME 2>/dev/null

# Start the container
echo "Starting vLLM container..."
docker run -d --name $CONTAINER_NAME \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e VLLM_BASE_DIR=/root/.cache/huggingface \
  vllm-node:latest \
  $MODEL \
  --port $PORT \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.7 \
  --load-format fastsafetensors \
  --kv-cache-dtype fp8

# Wait for server to be ready
echo "Waiting for server to start..."
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  Waited ${WAITED}s..."
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Server did not start within ${MAX_WAIT}s"
    docker logs $CONTAINER_NAME 2>&1 | tail -30
    exit 1
fi

# Run benchmarks
echo ""
echo "=== Running benchmarks ==="

for i in 1 2 3; do
    echo ""
    echo "Test $i:"
    START=$(date +%s.%N)
    RESULT=$(curl -s http://localhost:$PORT/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "'$MODEL'",
        "messages": [{"role": "user", "content": "Write a short story about a robot learning to paint."}],
        "max_tokens": 256,
        "temperature": 0.7
      }')
    END=$(date +%s.%N)
    
    DURATION=$(echo "$END - $START" | bc)
    TOKENS=$(echo "$RESULT" | jq -r '.usage.completion_tokens // 0')
    if [ "$TOKENS" -gt 0 ]; then
        TPS=$(echo "scale=2; $TOKENS / $DURATION" | bc)
        echo "  Tokens: $TOKENS, Duration: ${DURATION}s, TPS: $TPS"
    else
        echo "  ERROR: No tokens generated"
        echo "$RESULT" | jq .
    fi
done

echo ""
echo "=== Container logs (last 20 lines) ==="
docker logs $CONTAINER_NAME 2>&1 | tail -20

echo ""
echo "=== Memory usage ==="
docker logs $CONTAINER_NAME 2>&1 | grep -E "(memory|GiB)" | tail -5

# Leave container running for further testing
echo ""
echo "Container $CONTAINER_NAME is still running on port $PORT"
echo "To stop: docker stop $CONTAINER_NAME"
