//! Local configuration management (~/.anime-worker/config.toml).

use std::path::PathBuf;
use std::process::Command;

use serde::{Deserialize, Serialize};
use tracing::warn;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkerConstraints {
    /// Maximum model size in GB this worker will download
    pub max_model_size_gb: Option<f64>,

    /// Maximum total cache size in GB
    pub max_total_cache_gb: Option<f64>,

    /// Specific models this worker supports (allowlist)
    /// If set, worker only accepts tasks for these models
    pub supported_models: Option<Vec<String>>,

    /// Models this worker explicitly excludes (blocklist)
    pub excluded_models: Option<Vec<String>>,

    /// Minimum VRAM required (GB) - used for filtering tasks
    pub min_vram_gb: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    pub server_url: String,
    pub worker_id: String,
    pub api_key: String,
    pub worker_name: String,

    /// Directory for cached models (default: ~/.anime-worker/models)
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,

    /// Path to Python executable (auto-discovered if not set)
    #[serde(default = "default_python_path")]
    pub python_path: String,

    /// Directory containing the bundled Python inference scripts
    #[serde(default = "default_python_scripts_dir")]
    pub python_scripts_dir: PathBuf,

    /// Heartbeat interval in seconds
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval_secs: u64,

    /// Worker constraints for model filtering
    #[serde(default)]
    pub constraints: WorkerConstraints,
}

fn default_models_dir() -> PathBuf {
    config_dir().join("models")
}

fn default_python_path() -> String {
    discover_python().unwrap_or_else(|| "python3".to_string())
}

fn default_python_scripts_dir() -> PathBuf {
    use std::env;

    // Check multiple locations for python/ directory:

    // 1. Environment variable override
    if let Ok(env_path) = env::var("ANIME_WORKER_PYTHON_DIR") {
        let path = PathBuf::from(env_path);
        if path.exists() {
            return path;
        }
    }

    // 2. Next to the binary (for installed/release builds)
    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            let path = parent.join("python");
            if path.exists() {
                return path;
            }
        }
    }

    // 3. In the source directory (for development from cargo run)
    // Go up from target/release/ or target/debug/ to find python/
    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            // Try ../../python (from target/release or target/debug)
            if let Some(grandparent) = parent.parent() {
                if let Some(root) = grandparent.parent() {
                    let path = root.join("python");
                    if path.exists() {
                        return path;
                    }
                }
            }
        }
    }

    // 4. Current working directory
    let path = PathBuf::from("python");
    if path.exists() {
        return path;
    }

    // 5. Fallback
    warn!("Python scripts directory not found in any expected location");
    PathBuf::from("python")
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

/// Auto-discover a working Python 3.10+ interpreter.
///
/// Checks (in order):
/// 1. Bundled venv (python/venv/bin/python or python/venv/Scripts/python.exe)
/// 2. python3 on PATH
/// 3. python on PATH (Windows often only has `python`)
///
/// Returns None if no suitable Python is found.
pub fn discover_python() -> Option<String> {
    // 1. Check bundled venv next to the binary
    if let Ok(exe) = std::env::current_exe() {
        if let Some(base) = exe.parent() {
            let venv_candidates = if cfg!(windows) {
                vec![base.join("python/venv/Scripts/python.exe")]
            } else {
                vec![base.join("python/venv/bin/python")]
            };
            for candidate in venv_candidates {
                if candidate.exists() {
                    if let Some(path) = candidate.to_str() {
                        if check_python_version(path) {
                            return Some(path.to_string());
                        }
                    }
                }
            }
        }
    }

    // 2. Check python3 on PATH
    if check_python_version("python3") {
        return Some("python3".to_string());
    }

    // 3. Check python on PATH (Windows)
    if check_python_version("python") {
        return Some("python".to_string());
    }

    None
}

/// Check if a Python executable exists and is version 3.10+.
fn check_python_version(python: &str) -> bool {
    match Command::new(python).args(["--version"]).output() {
        Ok(output) if output.status.success() => {
            let version_str = String::from_utf8_lossy(&output.stdout);
            // Parse "Python 3.12.1" -> (3, 12)
            if let Some(ver) = version_str.strip_prefix("Python ") {
                let parts: Vec<&str> = ver.trim().split('.').collect();
                if parts.len() >= 2 {
                    let major: u32 = parts[0].parse().unwrap_or(0);
                    let minor: u32 = parts[1].parse().unwrap_or(0);
                    return major == 3 && minor >= 10;
                }
            }
            false
        }
        _ => false,
    }
}

/// Validate that the configured Python is usable. Call this at startup.
/// Returns an error message if Python is missing or too old.
pub fn validate_python(python_path: &str) -> Result<String, String> {
    match Command::new(python_path).args(["--version"]).output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            // Check minimum version
            if let Some(ver) = version.strip_prefix("Python ") {
                let parts: Vec<&str> = ver.split('.').collect();
                if parts.len() >= 2 {
                    let major: u32 = parts[0].parse().unwrap_or(0);
                    let minor: u32 = parts[1].parse().unwrap_or(0);
                    if major < 3 || (major == 3 && minor < 10) {
                        return Err(format!(
                            "Python 3.10+ required, found {} at '{}'",
                            version, python_path
                        ));
                    }
                }
            }
            Ok(version)
        }
        Ok(output) => Err(format!(
            "'{}' exited with error: {}",
            python_path,
            String::from_utf8_lossy(&output.stderr).trim()
        )),
        Err(e) => Err(format!(
            "Python not found at '{}': {}\n\n\
             Install Python 3.10+ and either:\n  \
             - Add it to your PATH, or\n  \
             - Set python_path in ~/.anime-worker/config.toml, or\n  \
             - Run `anime-worker setup-python` to create a bundled venv",
            python_path, e
        )),
    }
}

impl WorkerConfig {
    /// Load config from file and apply auto-detection
    pub fn load(path: &PathBuf) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut config: WorkerConfig = toml::from_str(&content)?;

        // Auto-detect max_model_size_gb if not set
        if config.constraints.max_model_size_gb.is_none() {
            let models_dir = &config.models_dir;
            std::fs::create_dir_all(models_dir).ok();

            let available = crate::hardware::available_disk_space_gb(models_dir);
            if available > 10.0 {
                // Use 80% of available space (leave 20% buffer)
                let recommended = (available * 0.8).floor();
                config.constraints.max_model_size_gb = Some(recommended);
                warn!(
                    "Auto-detected max_model_size_gb: {:.1} GB (80% of {:.1} GB available)",
                    recommended, available
                );
            }
        }

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
