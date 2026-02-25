# anime-worker

Distributed GPU worker for Anime Studio video generation. Connects outbound to the central server via WebSocket, receives job assignments, runs GPU inference via Python subprocess, and uploads results.

## Quick Start (New Interactive Setup)

```bash
# 1. Build the binary
cargo build --release

# 2. Run interactive setup (auto-configures everything)
./target/release/anime-worker setup
# → Prompts for API key (get from Workers dashboard)
# → Optionally configure resource limits (CPU/RAM/Disk)
# → Auto-runs Python setup
# → Creates config.toml

# 3. Start the worker
./target/release/anime-worker run
```

## Alternative: Manual Setup (Advanced)

```bash
# 1. Build the binary
cargo build --release

# 2. Register worker via web UI → copy worker_id and api_key

# 3. Initialize config manually
./target/release/anime-worker init \
  --server-url https://your-api.example.com \
  --worker-id <worker-id> \
  --api-key <api-key> \
  --name "My Worker"

# 4. Setup Python environment
./target/release/anime-worker setup-python

# 5. Start the worker
./target/release/anime-worker run
```

## Commands

- **`anime-worker setup`** - 🆕 Interactive setup wizard (recommended!)
  - Auto-detects existing config
  - Prompts for API key and settings
  - Configure resource limits (CPU/RAM/Disk)
  - Auto-runs Python setup

- `anime-worker run` - Connect to server and start processing tasks
- `anime-worker init` - Create config file (manual alternative to setup)
- `anime-worker hardware` - Show detected GPUs, RAM, CPU, and disk
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

Control resources and model filtering:

```toml
[constraints]
# 🆕 Resource Limits (configured via interactive setup)
cpu_limit = 8              # Allocate 8 cores (0 = use all)
ram_limit_gb = 32.0        # Limit to 32GB RAM (0 = use all)
disk_limit_gb = 500.0      # Report 500GB disk space (0 = use all available)

# Model filtering
max_model_size_gb = 30     # Don't download models larger than 30GB
max_total_cache_gb = 100   # Keep cache under 100GB total

# Allowlist/blocklist
# supported_models = ["wan22_ti2v_5b"]  # Only these models
# excluded_models = ["huge_model"]       # Never these models
```

**Why set resource limits?**
- Share machine resources with other services
- Prevent worker from using all available resources
- Set predictable capacity for multi-tenant setups
- Disk limits enforce safe buffer (10% extra space kept free)

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
