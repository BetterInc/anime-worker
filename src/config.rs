//! Local configuration management (~/.anime-worker/config.toml).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    pub server_url: String,
    pub worker_id: String,
    pub api_key: String,
    pub worker_name: String,

    /// Directory for cached models (default: ~/.anime-worker/models)
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,

    /// Path to Python executable (default: python3 or python in venv)
    #[serde(default = "default_python_path")]
    pub python_path: String,

    /// Directory containing the bundled Python inference scripts
    #[serde(default = "default_python_scripts_dir")]
    pub python_scripts_dir: PathBuf,

    /// Heartbeat interval in seconds
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval_secs: u64,
}

fn default_models_dir() -> PathBuf {
    config_dir().join("models")
}

fn default_python_path() -> String {
    "python3".to_string()
}

fn default_python_scripts_dir() -> PathBuf {
    // Default: python/ directory next to the binary
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.join("python")))
        .unwrap_or_else(|| PathBuf::from("python"))
}

fn default_heartbeat_interval() -> u64 {
    30
}

/// Get the config directory (~/.anime-worker)
pub fn config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".anime-worker")
}

/// Get the default config file path
pub fn config_file_path() -> PathBuf {
    config_dir().join("config.toml")
}

impl WorkerConfig {
    /// Load config from file
    pub fn load(path: &PathBuf) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: WorkerConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save config to file
    pub fn save(&self, path: &PathBuf) -> anyhow::Result<()> {
        let dir = path.parent().unwrap_or(std::path::Path::new("."));
        std::fs::create_dir_all(dir)?;
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}
