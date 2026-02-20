# anime-worker

Distributed GPU worker for Anime Studio video generation. Connects outbound to the central server via WebSocket, receives job assignments, runs GPU inference via Python subprocess, and uploads results.

## Quick Start

```bash
# 1. Build the binary
cargo build --release

# 2. Register a worker on the server (via the Workers dashboard)
#    Copy the worker_id and api_key

# 3. Initialize config
./target/release/anime-worker init \
  --server-url https://llm.pescheck.dev/api/anime \
  --worker-id <worker-id> \
  --api-key <api-key> \
  --name "My Gaming PC"

# 4. Setup Python environment (one-time)
./target/release/anime-worker setup-python

# 5. Start the worker
./target/release/anime-worker run
```

## Commands

- `anime-worker run` - Connect to server and start processing tasks
- `anime-worker init` - Create config file (~/.anime-worker/config.toml)
- `anime-worker hardware` - Show detected GPUs, RAM, and platform
- `anime-worker models` - List locally cached models
- `anime-worker setup-python` - Setup Python venv with ML dependencies

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 anime-worker (Rust)              │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ WebSocket │  │ Hardware │  │ Model Cache   │  │
│  │ Client    │  │ Detect   │  │ Management    │  │
│  └────┬─────┘  └──────────┘  └───────────────┘  │
│       │                                          │
│  ┌────▼──────────────────────────────────────┐   │
│  │         Python Subprocess Runner          │   │
│  │  stdin: JSON job config                   │   │
│  │  stdout: JSON progress/completion         │   │
│  │  stderr: logging                          │   │
│  └────┬──────────────────────────────────────┘   │
│       │                                          │
│  ┌────▼─────┐                                    │
│  │ HTTP     │                                    │
│  │ Uploader │                                    │
│  └──────────┘                                    │
└─────────────────────────────────────────────────┘
```

## Configuration

Config file: `~/.anime-worker/config.toml`

**Basic Configuration:**
```toml
server_url = "https://llm.pescheck.dev/api/anime"
worker_id = "uuid-from-server"
api_key = "aw_..."
worker_name = "My Worker"
models_dir = "/home/user/.anime-worker/models"
python_path = "python3"
heartbeat_interval_secs = 30
```

**Worker Constraints (Optional):**

Control which models this worker supports:

```toml
[constraints]
# Maximum model size to download (GB)
max_model_size_gb = 30

# Maximum total cache size (GB)
max_total_cache_gb = 100

# Only accept tasks for these models (allowlist)
# supported_models = ["wan22_ti2v_5b", "mochi_preview_bf16"]

# Never accept tasks for these models (blocklist)
# excluded_models = ["huge_model_100gb"]
```

**Examples:**

```toml
# Small worker (laptop with limited storage)
[constraints]
max_model_size_gb = 15
supported_models = ["wan22_ti2v_5b"]

# Large worker (server with lots of storage)
[constraints]
max_model_size_gb = 100  # Accept large models

# Specialized worker (only runs one model)
[constraints]
supported_models = ["wan22_ti2v_14b"]
```

See `config.toml.example` for full documentation.

## Requirements

- NVIDIA GPU with CUDA support
- `nvidia-smi` in PATH
- Python 3.10+ with CUDA-compatible PyTorch
- `hf` CLI tool (for model downloads): `pip install huggingface_hub[cli]`
