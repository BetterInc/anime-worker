//! anime-worker: Distributed GPU worker for Anime Studio video generation.
//!
//! Connects outbound to the central server via WebSocket, receives job
//! assignments, runs GPU inference via Python subprocess, and uploads results.

mod cleanup;
mod client;
mod config;
mod download_manager;
mod hardware;
mod log_layer;
mod metrics;
mod models;
mod protocol;
mod runner;
mod setup;
mod upload;

use std::sync::Arc;

use clap::{Parser, Subcommand};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

fn version() -> &'static str {
    concat!(
        env!("GIT_VERSION"),
        " (",
        env!("GIT_COMMIT"),
        " ",
        env!("BUILD_DATE"),
        ")"
    )
}

#[derive(Parser)]
#[command(
    name = "anime-worker",
    version = version(),
    about = "Distributed GPU worker for Anime Studio"
)]
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

    /// Interactive configuration wizard — creates config.toml in the current directory
    Setup,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run => {
            let config_path = cli
                .config
                .map(std::path::PathBuf::from)
                .unwrap_or_else(config::config_file_path);

            let mut cfg = config::WorkerConfig::load(&config_path)?;

            // Initialize tracing with optional API log forwarding
            let log_tx = if cfg.enable_log_streaming {
                let (tx, _rx) = tokio::sync::mpsc::channel(1000);

                // Setup tracing with API layer
                use tracing_subscriber::layer::SubscriberExt;
                let api_layer = log_layer::ApiLogLayer::new(tx.clone());
                let subscriber = tracing_subscriber::fmt()
                    .with_env_filter(
                        EnvFilter::try_from_default_env()
                            .unwrap_or_else(|_| EnvFilter::new("info")),
                    )
                    .finish()
                    .with(api_layer);

                tracing::subscriber::set_global_default(subscriber)?;

                // Return both tx and rx, we'll use rx in client
                Some(tx)
            } else {
                // No log streaming, just basic console logging
                tracing_subscriber::fmt()
                    .with_env_filter(
                        EnvFilter::try_from_default_env()
                            .unwrap_or_else(|_| EnvFilter::new("info")),
                    )
                    .init();
                None
            };
            info!("Loaded config from {}", config_path.display());
            info!("Worker: {} ({})", cfg.worker_name, &cfg.worker_id[..8]);
            info!("Server: {}", cfg.server_url);
            info!("Models dir: {}", cfg.models_dir.display());

            // Auto-setup Python environment if needed
            let venv_path = cfg.python_scripts_dir.join("venv");
            let venv_python = if cfg!(windows) {
                venv_path.join("Scripts").join("python.exe")
            } else {
                venv_path.join("bin").join("python")
            };
            if !venv_path.exists() {
                info!(
                    "Python virtual environment not found at {}",
                    venv_path.display()
                );
                info!("Running automatic Python setup...");

                let bootstrap_python = config::discover_python()
                    .ok_or_else(|| anyhow::anyhow!("No Python found on PATH for setup"))?;

                let setup_script = cfg.python_scripts_dir.join("setup_env.py");
                if !setup_script.exists() {
                    anyhow::bail!(
                        "setup_env.py not found at {}. Ensure python/ directory is properly deployed.",
                        setup_script.display()
                    );
                }

                info!("This will install PyTorch, diffusers, and other dependencies (may take a few minutes)...");
                let status = std::process::Command::new(&bootstrap_python)
                    .arg(&setup_script)
                    .status()?;

                if !status.success() {
                    anyhow::bail!("Python setup failed. Run 'anime-worker setup-python' manually for details.");
                }

                info!("Python environment setup complete!");
            }

            // Auto-configure to use venv Python if it exists (unless user has custom path)
            if venv_python.exists() {
                let venv_python_str = venv_python.to_string_lossy().to_string();

                // Only auto-configure if:
                // 1. Config points to default "python3" or doesn't exist
                // 2. Config points to non-existent Python
                let should_auto_configure = cfg.python_path == "python3"
                    || cfg.python_path == "python"
                    || !std::path::Path::new(&cfg.python_path).exists();

                if should_auto_configure && cfg.python_path != venv_python_str {
                    info!("Auto-configuring to use venv Python: {}", venv_python_str);
                    cfg.python_path = venv_python_str.clone();
                    cfg.save(&config_path)?;
                } else if !should_auto_configure {
                    info!("Using custom Python from config: {}", cfg.python_path);
                }
            }

            // Validate Python before connecting
            match config::validate_python(&cfg.python_path) {
                Ok(version) => info!("Python: {} at '{}'", version, cfg.python_path),
                Err(msg) => {
                    error!("{}", msg);
                    anyhow::bail!("Cannot start without a working Python installation");
                }
            }

            // Ensure models dir exists
            std::fs::create_dir_all(&cfg.models_dir)?;

            client::run(Arc::new(cfg), log_tx).await;
        }

        _ => {
            // For non-Run commands, initialize simple console logging
            tracing_subscriber::fmt()
                .with_env_filter(
                    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
                )
                .init();

            handle_other_commands(cli).await?;
        }
    }

    Ok(())
}

async fn handle_other_commands(cli: Cli) -> anyhow::Result<()> {
    let config_path = cli
        .config
        .as_ref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(config::config_file_path);

    match cli.command {
        Commands::Run => unreachable!(),

        Commands::Init {
            server_url,
            worker_id,
            api_key,
            name,
        } => {
            // Auto-discover Python
            let python_path = match config::discover_python() {
                Some(path) => {
                    info!("Auto-discovered Python at '{}'", path);
                    path
                }
                None => {
                    info!("No Python found on PATH. Set python_path in config after installing Python 3.10+");
                    "python3".to_string()
                }
            };

            let cfg = config::WorkerConfig {
                server_url,
                worker_id,
                api_key,
                worker_name: name,
                models_dir: config::config_dir().join("models"),
                python_path,
                python_scripts_dir: std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|p| p.join("python")))
                    .unwrap_or_else(|| std::path::PathBuf::from("python")),
                heartbeat_interval_secs: 30,
                constraints: config::WorkerConstraints::default(),
                enable_log_streaming: false,
                enable_metrics_collection: false,
                cleanup_interval_secs: 3600,
                retention_hours: 24,
            };

            cfg.save(&config_path)?;
            info!("Config saved to {}", config_path.display());
            info!("Run `anime-worker run` to start the worker.");
        }

        Commands::Hardware => {
            println!("Platform: {}", hardware::platform());
            println!(
                "RAM: {:.1} GB total, {:.1} GB free",
                hardware::total_ram_gb(),
                hardware::free_ram_gb()
            );

            // Python check
            match config::discover_python() {
                Some(path) => {
                    let version = config::validate_python(&path).unwrap_or_else(|e| e);
                    println!("Python: {} at '{}'", version, path);
                }
                None => {
                    println!("Python: NOT FOUND — install Python 3.10+ and add to PATH");
                }
            }

            let gpus = hardware::detect_gpus();
            if gpus.is_empty() {
                println!("GPUs: NOT FOUND (nvidia-smi not found or no NVIDIA GPU)");
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

        Commands::Setup => {
            // Use config.toml in current dir if --config not specified
            let setup_config_path = if cli.config.is_some() {
                config_path.clone()
            } else {
                std::path::PathBuf::from("config.toml")
            };

            // Locate the python/ directory: try exe-relative first, then cwd-relative.
            let scripts_dir = {
                // First try next to the executable (deployed scenario)
                let exe_relative = std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|p| p.join("python")));

                if let Some(ref path) = exe_relative {
                    if path.join("setup_env.py").exists() {
                        path.clone()
                    } else {
                        // Fallback: check current working directory (development scenario)
                        let cwd_relative = std::path::PathBuf::from("python");
                        if cwd_relative.join("setup_env.py").exists() {
                            cwd_relative
                        } else {
                            exe_relative.unwrap_or_else(|| std::path::PathBuf::from("python"))
                        }
                    }
                } else {
                    std::path::PathBuf::from("python")
                }
            };

            setup::run(&setup_config_path, &scripts_dir)?;
        }

        Commands::SetupPython => {
            // Find any Python to bootstrap with
            let bootstrap_python = config::discover_python().ok_or_else(|| {
                anyhow::anyhow!(
                    "No Python 3.10+ found on PATH.\n\
                     Install Python first:\n  \
                     - Linux: sudo apt install python3 python3-venv\n  \
                     - Windows: https://www.python.org/downloads/"
                )
            })?;

            let scripts_dir = cli
                .config
                .map(std::path::PathBuf::from)
                .and_then(|p| config::WorkerConfig::load(&p).ok())
                .map(|c| c.python_scripts_dir)
                .unwrap_or_else(|| {
                    // First try next to the executable (deployed scenario)
                    let exe_relative = std::env::current_exe()
                        .ok()
                        .and_then(|p| p.parent().map(|p| p.join("python")));

                    if let Some(ref path) = exe_relative {
                        if path.join("setup_env.py").exists() {
                            return path.clone();
                        }
                    }

                    // Fallback: check current working directory (development scenario)
                    let cwd_relative = std::path::PathBuf::from("python");
                    if cwd_relative.join("setup_env.py").exists() {
                        return cwd_relative;
                    }

                    // Default to exe-relative path for error messaging
                    exe_relative.unwrap_or_else(|| std::path::PathBuf::from("python"))
                });

            let setup_script = scripts_dir.join("setup_env.py");
            if !setup_script.exists() {
                anyhow::bail!(
                    "setup_env.py not found at {}.\n\
                     Make sure the python/ directory is next to the binary.",
                    setup_script.display()
                );
            }

            info!(
                "Setting up Python environment using '{}'...",
                bootstrap_python
            );
            let status = std::process::Command::new(&bootstrap_python)
                .arg(&setup_script)
                .status()?;

            if status.success() {
                info!("Python environment setup complete");
                info!("Update python_path in your config.toml if needed.");
            } else {
                anyhow::bail!("Python setup failed with exit code {:?}", status.code());
            }
        }
    }

    Ok(())
}
