//! HuggingFace model downloading and cache management.

use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::{info, warn};

/// Check if a model is cached locally (has model_index.json).
pub fn is_model_cached(models_dir: &Path, local_dir: &str) -> bool {
    let model_path = models_dir.join(local_dir);
    model_path.join("model_index.json").exists()
}

/// List all cached model directories (those with model_index.json).
/// Returns model IDs (not local directory names) by mapping via model_configs.yaml
pub fn list_cached_models(models_dir: &Path) -> Vec<String> {
    let mut local_dirs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            if entry.path().join("model_index.json").exists() {
                if let Some(name) = entry.file_name().to_str() {
                    local_dirs.push(name.to_string());
                }
            }
        }
    }

    // Map local directories to model IDs using model_configs.yaml
    map_local_dirs_to_model_ids(&local_dirs)
}

/// Map local directory names to model IDs using model_configs.yaml
fn map_local_dirs_to_model_ids(local_dirs: &[String]) -> Vec<String> {
    use std::collections::HashMap;

    // Try to load model_configs.yaml from python dir
    let config_paths = vec![
        PathBuf::from("/app/python/model_configs.yaml"),
        PathBuf::from("./python/model_configs.yaml"),
        PathBuf::from("../python/model_configs.yaml"),
    ];

    for config_path in config_paths {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_yaml::from_str::<serde_yaml::Value>(&content) {
                if let Some(models) = config.get("models").and_then(|m| m.as_mapping()) {
                    // Build reverse map: local_dir -> model_id
                    let mut dir_to_id: HashMap<String, String> = HashMap::new();
                    for (model_id, model_data) in models {
                        if let (Some(id_str), Some(local_dir)) = (
                            model_id.as_str(),
                            model_data.get("local_dir").and_then(|v| v.as_str()),
                        ) {
                            dir_to_id.insert(local_dir.to_string(), id_str.to_string());
                        }
                    }

                    // Map the local dirs to model IDs
                    return local_dirs
                        .iter()
                        .filter_map(|dir| dir_to_id.get(dir).cloned())
                        .collect();
                }
            }
        }
    }

    // Fallback: return local dirs if mapping fails
    warn!("Failed to load model_configs.yaml, reporting local dirs instead of model IDs");
    local_dirs.to_vec()
}

/// Download a model from HuggingFace using the `hf` CLI tool.
///
/// Returns the local path on success.
/// Progress is reported via the callback (downloaded_bytes, total_bytes_estimate).
pub async fn download_model(
    hf_repo: &str,
    models_dir: &Path,
    local_dir: &str,
    python_path: &str,
    progress_callback: impl Fn(f64, f64) + Send + 'static,
) -> anyhow::Result<PathBuf> {
    let model_path = models_dir.join(local_dir);
    std::fs::create_dir_all(&model_path)?;

    info!("Downloading model {} to {}", hf_repo, model_path.display());

    // Use hf CLI for downloading (handles resume, auth, etc.)
    let model_path_str = model_path.to_string_lossy().to_string();
    let hf_repo = hf_repo.to_string();

    // Determine hf CLI path from python_path
    // If python_path is /path/to/venv/bin/python, then hf is /path/to/venv/bin/hf
    let python_pathbuf = PathBuf::from(python_path);
    let hf_cli_path = if let Some(bin_dir) = python_pathbuf.parent() {
        let hf = bin_dir.join("hf");
        if hf.exists() {
            hf.to_string_lossy().to_string()
        } else {
            "hf".to_string() // Fallback to PATH
        }
    } else {
        "hf".to_string()
    };

    tokio::task::spawn_blocking(move || {
        // Use hf CLI from venv
        let mut cmd = Command::new(&hf_cli_path);
        cmd.args(["download", &hf_repo, "--local-dir", &model_path_str]);

        // Add HF token if available
        if let Ok(token) = std::env::var("HF_TOKEN") {
            if !token.is_empty() {
                cmd.args(["--token", &token]);
            }
        }

        // Spawn a monitor thread that checks disk size periodically
        let monitor_path = PathBuf::from(&model_path_str);
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_clone = stop.clone();

        let monitor = std::thread::spawn(move || {
            while !stop_clone.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_secs(5));
                if let Ok(size) = dir_size_bytes(&monitor_path) {
                    let gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
                    progress_callback(gb, 0.0); // total unknown from disk monitoring
                }
            }
        });

        let output = cmd.output();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = monitor.join();

        match output {
            Ok(out) if out.status.success() => Ok(()),
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                anyhow::bail!("hf download failed: {}", stderr);
            }
            Err(e) => anyhow::bail!("Failed to run hf CLI: {}", e),
        }
    })
    .await??;

    info!("Model download complete: {}", model_path.display());
    Ok(model_path)
}

/// Calculate directory size in bytes.
fn dir_size_bytes(path: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                total += dir_size_bytes(&entry.path())?;
            } else {
                total += metadata.len();
            }
        }
    }
    Ok(total)
}
