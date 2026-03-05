# LFM 2.5 1.2B Thinking

Liquid AI's reasoning model with state-space architecture.

## Features
- 1.2B parameters
- Chain-of-thought reasoning with `<think>` tags
- 32K context length
- State-space architecture (fast inference)
- ~450 tok/s on Blackwell

## Backends

Setup auto-detects platform and installs the appropriate backend:
- **macOS**: MLX (Apple Silicon GPU acceleration)
- **Linux + NVIDIA**: vLLM (CUDA)
- **Linux (CPU)**: vLLM (CPU fallback)

## Setup

```bash
./setup.sh
```

## Run

```bash
./run.sh
```

Server runs on port 2022 with OpenAI-compatible API.

## Systemd

```bash
sudo cp systemd/kortexa-ai-llm-lfm-2.5-1.2b-thinking.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable kortexa-ai-llm-lfm-2.5-1.2b-thinking
sudo systemctl start kortexa-ai-llm-lfm-2.5-1.2b-thinking
```
