//! HuggingFace model downloading and cache management.
//! Pure Rust implementation - no Python dependencies required.

use std::path::{Path, PathBuf};

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

/// Download a model from HuggingFace using Python CLI (more reliable for large files).
///
/// Returns the local path on success.
/// Progress is reported via the callback (downloaded_gb, total_gb_estimate).
pub async fn download_model(
    hf_repo: &str,
    models_dir: &Path,
    local_dir: &str,
    progress_callback: impl Fn(f64, f64) + Send + 'static,
) -> anyhow::Result<PathBuf> {
    let model_path = models_dir.join(local_dir);
    std::fs::create_dir_all(&model_path)?;

    info!("Downloading model {} to {}", hf_repo, model_path.display());

    let hf_repo = hf_repo.to_string();
    let model_path_str = model_path.to_string_lossy().to_string();

    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        // Use huggingface-cli download for reliability with large files
        let python_path = std::env::current_dir()?.join("python/venv/bin/python");

        if !python_path.exists() {
            return Err(anyhow::anyhow!(
                "Python venv not found at {}",
                python_path.display()
            ));
        }

        info!("Using Python HuggingFace Hub to download (more reliable for large files)");

        let output = std::process::Command::new(&python_path)
            .args([
                "-c",
                &format!(
                    "from huggingface_hub import snapshot_download; \
                     snapshot_download(\
                         repo_id='{}', \
                         local_dir='{}', \
                         local_dir_use_symlinks=False, \
                         resume_download=True\
                     )",
                    hf_repo, model_path_str
                ),
            ])
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to spawn Python process: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Model download failed: {}", stderr));
        }

        info!("Model downloaded successfully");

        // Report final size
        if let Ok(size) = dir_size_bytes(&PathBuf::from(&model_path_str)) {
            let gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
            progress_callback(gb, 0.0);
        }

        Ok(())
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
            if metadata.is_file() {
                total += metadata.len();
            } else if metadata.is_dir() {
                total += dir_size_bytes(&entry.path())?;
            }
        }
    }
    Ok(total)
}
