# anime-worker

Distributed GPU worker for Anime Studio video generation. Connects outbound to the central server via WebSocket, receives job assignments (batches of scenes), runs GPU inference via Python subprocess, and uploads results. Cross-platform: Linux, Windows, and macOS.

📖 **[Privacy & Data Usage](PRIVACY.md)** - See exactly what data is sent/received and why it's needed

## Quick Start

```bash
# 1. Build the binary
cargo build --release

# 2. Run interactive setup (auto-configures everything)
./target/release/anime-worker setup
# -> Prompts for API key (get from Workers dashboard)
# -> Optionally configure resource limits (CPU/RAM/Disk)
# -> Auto-runs Python setup

# 3. Start the worker
./target/release/anime-worker run
```

On first `run`, the worker will auto-create a Python virtual environment and install all dependencies if one doesn't already exist.

## Alternative: Manual Setup

```bash
# 1. Build the binary
cargo build --release

# 2. Register worker via web UI -> copy worker_id and api_key

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

| Command | Description |
|---------|-------------|
| `anime-worker setup` | Interactive setup wizard (recommended) |
| `anime-worker run` | Connect to server and start processing jobs |
| `anime-worker init` | Create config file manually |
| `anime-worker hardware` | Show detected GPUs, RAM, CPU, disk, and Python |
| `anime-worker models` | List locally cached models |
| `anime-worker setup-python` | Setup Python venv with ML dependencies |

Global option: `--config <path>` to specify a custom config file path.

## Supported Task Types

- **preview** - Low quality video generation (fast iteration)
- **render** - High quality video generation (final output)
- **upscale** - Real-ESRGAN upscaling (720p -> 1080p using AnimeVideo-v3)

## Supported Models

Configured via `python/model_configs.yaml`:

| Model ID | Pipeline | Notes |
|----------|----------|-------|
| `wan22_ti2v_5b` | WanPipeline | Wan 2.2 Text/Image-to-Video 5B |
| `wan22_t2v_14b` | WanPipeline | Wan 2.2 Text-to-Video 14B |
| `mochi_1_i2v` | MochiPipeline | Mochi 1 |
| `ltx_video` | LTXPipeline | LTX Video |
| `hunyuanvideo` | HunyuanVideoPipeline | HunyuanVideo |

Models are automatically downloaded from HuggingFace on first use with progress reporting and disk space checks.

## Architecture

```
                          Central Server
                               |
                          WebSocket (wss://)
                               |
┌──────────────────────────────┴──────────────────────────────┐
│                     anime-worker (Rust)                      │
│                                                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │WebSocket │  │ Hardware   │  │  Model   │  │  Config   │  │
│  │ Client   │  │ Detection  │  │  Cache   │  │ Manager   │  │
│  │(client)  │  │(hardware)  │  │(models)  │  │(config)   │  │
│  └────┬─────┘  └───────────┘  └──────────┘  └───────────┘  │
│       │                                                      │
│  ┌────▼────────────────────────────────────────────────┐    │
│  │          Python Subprocess Runner (runner)           │    │
│  │  stdin:  JSON job config (scenes, model, pipeline)   │    │
│  │  stdout: JSON progress / completion messages         │    │
│  │  stderr: logging                                     │    │
│  └────┬────────────────────────────────────────────────┘    │
│       │                                                      │
│  ┌────▼──────┐                                               │
│  │  HTTP     │  multipart upload per scene                   │
│  │  Uploader │  + lastframe PNG for continuity               │
│  └───────────┘                                               │
└──────────────────────────────────────────────────────────────┘
```

### Job Batch Processing

The server assigns a **batch of scenes** per job. The worker:

1. Downloads the model (if not cached) with progress reporting
2. Loads the model into VRAM **once**
3. Processes all scenes sequentially, reusing the loaded model
4. Passes the last frame of each scene to the next for visual continuity
5. Uploads each scene's video + lastframe PNG back to the server

### WebSocket Protocol

The worker communicates via **two channels**:

1. **WebSocket** (`wss://`) - Real-time control protocol for job assignment and progress
2. **HTTP** - File uploads/downloads (multipart upload of videos + lastframe PNGs, downloading input files)

**WebSocket: Worker -> Server:**
- `hello` - Authentication + hardware stats + cached models + constraints
- `heartbeat` - Periodic hardware stats update
- `request_task` - Ask for work
- `task_progress` - Generation progress (phase + percentage)
- `model_progress` - Download progress (GB downloaded / total)
- `task_complete` - Scene done with metadata

**WebSocket: Server -> Worker:**
- `hello_ack` - Authentication accepted
- `job_batch_assign` - Batch of scenes to process
- `task_cancel` - Cancel current work
- `tasks_available` - Nudge to request work
- `heartbeat_ack` - Heartbeat acknowledged

**HTTP:**
- `POST {server_url}{upload_path}` - Multipart file upload (video + lastframe)
- `GET {server_url}{path}` - Download input files (e.g., lastframes from previous scenes)

All HTTP requests use `X-API-Key` header for authentication.

## Configuration

The worker looks for config in this order:
1. `--config <path>` flag
2. `./config.toml` (current directory)
3. `~/.anime-worker/config.toml`

### Basic Config

```toml
server_url = "https://animestudiocontaineryaw5gajb-anime-api.functions.fnc.nl-ams.scw.cloud"
worker_id = "uuid-from-server"
api_key = "aw_..."
worker_name = "My Worker"
```

### Optional Settings

```toml
models_dir = "/path/to/models"        # Default: ~/.anime-worker/models
python_path = "python3"                # Auto-detected if not set
python_scripts_dir = "./python"        # Default: auto-detected near binary
heartbeat_interval_secs = 30           # Default: 30
cleanup_interval_secs = 3600           # Cleanup interval (default: 1 hour)
retention_hours = 24                   # File retention period (default: 24 hours)
```

### Worker Constraints

```toml
[constraints]
# Resource limits
cpu_limit = 8              # Allocate 8 cores (omit = use all)
ram_limit_gb = 32.0        # Limit to 32GB RAM (omit = use all)
disk_limit_gb = 500.0      # Report 500GB disk space (omit = use all)

# Model filtering
max_model_size_gb = 30     # Don't download models larger than 30GB
max_total_cache_gb = 100   # Keep cache under 100GB total

# Allowlist/blocklist
# supported_models = ["wan22_ti2v_5b"]  # Only these models
# excluded_models = ["huge_model"]       # Never these models
```

### Example Configurations

```toml
# Small worker (laptop with limited storage)
[constraints]
max_model_size_gb = 15
supported_models = ["wan22_ti2v_5b"]

# Large worker (server with lots of storage)
[constraints]
max_model_size_gb = 100

# Specialized worker (only runs one model)
[constraints]
supported_models = ["wan22_ti2v_14b"]
```

See `config.toml.example` for full documentation.

## Python Environment

The `python/` directory contains the inference scripts:

| File | Purpose |
|------|---------|
| `inference_runner.py` | Entry point spawned by Rust worker |
| `setup_env.py` | Creates venv and installs dependencies |
| `model_configs.yaml` | Model ID -> pipeline class mapping |
| `lib/pipeline.py` | Pipeline loading and setup |
| `lib/inference.py` | Frame generation logic |
| `lib/upscale.py` | Real-ESRGAN video upscaling |
| `lib/assembly.py` | Video assembly utilities |
| `lib/config.py` | Parameter validation per model |
| `lib/hardware.py` | GPU/hardware detection helpers |

The Python venv is auto-created on first `run` if missing. It installs PyTorch (CUDA), diffusers, transformers, and other ML dependencies.

## CI/CD

GitHub Actions runs on every push/PR:

- **Lint** - `cargo fmt --check` + `cargo clippy -D warnings`
- **Build + Test** - Linux, Windows, macOS (x86_64)
- **Config loading test** - Validates binary can load config on each platform

Releases are triggered by version tags (`v*.*.*`). Artifacts are packaged with the `python/` directory included:

| Platform | Artifact |
|----------|----------|
| Linux x86_64 | `anime-worker-linux-x86_64.tar.gz` |
| Windows x86_64 | `anime-worker-windows-x86_64.zip` |
| macOS x86_64 | `anime-worker-macos-x86_64.tar.gz` |
| macOS ARM64 | `anime-worker-macos-arm64.tar.gz` |

## Requirements

- NVIDIA GPU with CUDA support (for inference)
- `nvidia-smi` in PATH
- Python 3.10+
- Rust toolchain (for building from source)

## Maintenance

### Automatic Cleanup

The worker automatically cleans up old temporary files and outputs while running. No cron setup needed!

**What gets cleaned:**
- Output directories older than 24 hours from `~/.anime-worker/output`
- PNG files older than 24 hours from `~/.anime-worker/tmp`

**Configuration** (in `config.toml`):

```toml
cleanup_interval_secs = 3600  # Run cleanup every hour (default)
retention_hours = 24           # Keep files for 24 hours (default)
```

The cleanup runs automatically in the background while the worker daemon is active. You can adjust the interval and retention period to suit your needs.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANIME_WORKER_PYTHON_DIR` | Override Python scripts directory |
| `RUST_LOG` | Control log level (e.g., `info`, `debug`, `warn`) |

## License

MIT
