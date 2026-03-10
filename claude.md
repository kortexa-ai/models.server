# Claude Notes

## Current Work

- We are building latest-source TensorRT-LLM from `main` in Docker on the DGX Spark/GB10 machine to get `Qwen 3.5` working through autodeploy from safetensors.
- The active long-running build is:
  `FULL_SOURCE_BUILD=1 CUDA_ARCHITECTURES=120-real JOB_COUNT=4 ./trtllm.spark/build-main-image.sh`
- Goal after the image finishes: try `Qwen/Qwen3.5-0.8B`, record TPS if it serves successfully, then repeat for `4B` and `9B`.
- Extended context and benchmark history live in `/home/francip/src/models.server/sglang_spark.md`.
- TensorRT-LLM-specific build and run context live in `/home/francip/src/models.server/trtllm.spark/README.md`, `/home/francip/src/models.server/trtllm.spark/build-main-image.sh`, and `/home/francip/src/models.server/trtllm.spark/Dockerfile.main-source`.

### Monitoring

- The live build output stream is attached to another terminal session, so monitor it indirectly from the host with:
  `ps -eo pid,ppid,pcpu,pmem,etime,cmd | rg 'docker build|build-main-image|cmake --build|nvcc|sourcebuild-120-real'`
- Check host memory headroom with:
  `free -h`
- Poll both together with:
  `watch -n 10 "ps -eo pid,ppid,pcpu,pmem,etime,cmd | rg 'docker build|build-main-image|cmake --build|nvcc|sourcebuild-120-real'; echo; free -h"`
- When the build is done, the image should appear here:
  `docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedSince}}' | rg 'local/trtllm-main|tensorrt-llm'`
- If the current build needs to be restarted, use:
  `cd /home/francip/src/models.server && FULL_SOURCE_BUILD=1 CUDA_ARCHITECTURES=120-real JOB_COUNT=4 ./trtllm.spark/build-main-image.sh`
