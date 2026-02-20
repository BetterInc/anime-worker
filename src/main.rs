//! anime-worker: Distributed GPU worker for Anime Studio video generation.
//!
//! Connects outbound to the central server via WebSocket, receives job
//! assignments, runs GPU inference via Python subprocess, and uploads results.

mod client;
mod config;
mod hardware;
mod models;
mod protocol;
mod runner;
mod upload;

use std::sync::Arc;

use clap::{Parser, Subcommand};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "anime-worker", version, about = "Distributed GPU worker for Anime Studio")]
struct Cli {
    /// Path to config file (default: ~/.anime-worker/config.toml)
    #[arg(short, long)]
    config: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the worker (connect to server and process tasks)
    Run,

    /// Initialize a new config file interactively
    Init {
        /// Server URL (e.g., https://llm.pescheck.dev/api/anime)
        #[arg(long)]
        server_url: String,

        /// Worker ID (from server registration)
        #[arg(long)]
        worker_id: String,

        /// API key (from server registration)
        #[arg(long)]
        api_key: String,

        /// Worker name
        #[arg(long)]
        name: String,
    },

    /// Show detected hardware
    Hardware,

    /// List cached models
    Models,

    /// Setup Python environment (create venv, install requirements)
    SetupPython,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run => {
            let config_path = cli
                .config
                .map(std::path::PathBuf::from)
                .unwrap_or_else(config::config_file_path);

            let cfg = config::WorkerConfig::load(&config_path)?;
            info!("Loaded config from {}", config_path.display());
            info!("Worker: {} ({})", cfg.worker_name, &cfg.worker_id[..8]);
            info!("Server: {}", cfg.server_url);
            info!("Models dir: {}", cfg.models_dir.display());

            // Ensure models dir exists
            std::fs::create_dir_all(&cfg.models_dir)?;

            client::run(Arc::new(cfg)).await;
        }

        Commands::Init {
            server_url,
            worker_id,
            api_key,
            name,
        } => {
            let config_path = cli
                .config
                .map(std::path::PathBuf::from)
                .unwrap_or_else(config::config_file_path);

            let cfg = config::WorkerConfig {
                server_url,
                worker_id,
                api_key,
                worker_name: name,
                models_dir: config::config_dir().join("models"),
                python_path: "python3".to_string(),
                python_scripts_dir: std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|p| p.join("python")))
                    .unwrap_or_else(|| std::path::PathBuf::from("python")),
                heartbeat_interval_secs: 30,
            };

            cfg.save(&config_path)?;
            info!("Config saved to {}", config_path.display());
            info!("Run `anime-worker run` to start the worker.");
        }

        Commands::Hardware => {
            println!("Platform: {}", hardware::platform());
            println!("RAM: {:.1} GB total, {:.1} GB free",
                hardware::total_ram_gb(), hardware::free_ram_gb());
            let gpus = hardware::detect_gpus();
            if gpus.is_empty() {
                println!("No GPUs detected (nvidia-smi not found or no NVIDIA GPU)");
            } else {
                for (i, gpu) in gpus.iter().enumerate() {
                    println!(
                        "GPU {}: {} — {} MB total, {} MB free",
                        i, gpu.name, gpu.vram_total_mb, gpu.vram_free_mb
                    );
                }
            }
        }

        Commands::Models => {
            let config_path = cli
                .config
                .map(std::path::PathBuf::from)
                .unwrap_or_else(config::config_file_path);

            let cfg = config::WorkerConfig::load(&config_path)?;
            let cached = models::list_cached_models(&cfg.models_dir);
            if cached.is_empty() {
                println!("No models cached in {}", cfg.models_dir.display());
            } else {
                println!("Cached models in {}:", cfg.models_dir.display());
                for model in &cached {
                    println!("  - {}", model);
                }
            }
        }

        Commands::SetupPython => {
            let config_path = cli
                .config
                .map(std::path::PathBuf::from)
                .unwrap_or_else(config::config_file_path);

            let cfg = config::WorkerConfig::load(&config_path)?;
            let setup_script = cfg.python_scripts_dir.join("setup_env.py");
            if !setup_script.exists() {
                anyhow::bail!(
                    "setup_env.py not found at {}. Make sure python/ directory is next to the binary.",
                    setup_script.display()
                );
            }

            info!("Running Python environment setup...");
            let status = std::process::Command::new("python3")
                .arg(&setup_script)
                .status()?;

            if status.success() {
                info!("Python environment setup complete");
            } else {
                anyhow::bail!("Python setup failed with exit code {:?}", status.code());
            }
        }
    }

    Ok(())
}
