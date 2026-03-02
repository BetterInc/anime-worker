//! HuggingFace model downloading and cache management.
//! Pure Rust implementation - no Python dependencies required.

use std::path::{Path, PathBuf};

use tracing::info;

/// Check if a model is fully cached locally.
/// Requires both model_index.json AND .download_complete marker.
pub fn is_model_cached(models_dir: &Path, local_dir: &str) -> bool {
    let model_path = models_dir.join(local_dir);
    model_path.join("model_index.json").exists() && model_path.join(".download_complete").exists()
}

/// List all fully cached model directories.
/// Requires both model_index.json AND .download_complete marker.
/// Returns model IDs by reading .model_id marker file, or falls back to local_dir name
pub fn list_cached_models(models_dir: &Path) -> Vec<String> {
    let mut model_ids = Vec::new();
    if let Ok(entries) = std::fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            // Only report as cached if download completed successfully
            if path.join("model_index.json").exists() && path.join(".download_complete").exists() {
                // Try to read .model_id marker file first
                if let Ok(model_id) = std::fs::read_to_string(path.join(".model_id")) {
                    model_ids.push(model_id.trim().to_string());
                } else if let Some(name) = entry.file_name().to_str() {
                    // Fallback: use directory name (for old downloads without .model_id)
                    model_ids.push(name.to_string());
                }
            }
        }
    }

    model_ids
}

// Removed map_local_dirs_to_model_ids - now using .model_id marker files

/// Download a model from HuggingFace using Python CLI (more reliable for large files).
///
/// Returns the local path on success.
/// Progress is reported via the callback (downloaded_gb, total_gb).
pub async fn download_model(
    hf_repo: &str,
    models_dir: &Path,
    local_dir: &str,
    model_size_gb: f64,
    progress_callback: impl Fn(f64, f64) + Send + 'static,
) -> anyhow::Result<PathBuf> {
    download_model_with_id(
        hf_repo,
        models_dir,
        local_dir,
        None,
        model_size_gb,
        progress_callback,
    )
    .await
}

/// Download a model with optional model_id tracking and cancellation support
pub async fn download_model_with_cancellation(
    hf_repo: &str,
    models_dir: &Path,
    local_dir: &str,
    model_id: Option<&str>,
    model_size_gb: f64,
    cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    progress_callback: impl Fn(f64, f64) + Send + 'static,
) -> anyhow::Result<PathBuf> {
    download_model_internal(
        hf_repo,
        models_dir,
        local_dir,
        model_id,
        model_size_gb,
        Some(cancel_flag),
        progress_callback,
    )
    .await
}

/// Download a model with optional model_id tracking
pub async fn download_model_with_id(
    hf_repo: &str,
    models_dir: &Path,
    local_dir: &str,
    model_id: Option<&str>,
    model_size_gb: f64,
    progress_callback: impl Fn(f64, f64) + Send + 'static,
) -> anyhow::Result<PathBuf> {
    download_model_internal(
        hf_repo,
        models_dir,
        local_dir,
        model_id,
        model_size_gb,
        None,
        progress_callback,
    )
    .await
}

/// Internal download implementation with optional cancellation
async fn download_model_internal(
    hf_repo: &str,
    models_dir: &Path,
    local_dir: &str,
    model_id: Option<&str>,
    model_size_gb: f64,
    cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    progress_callback: impl Fn(f64, f64) + Send + 'static,
) -> anyhow::Result<PathBuf> {
    let model_path = models_dir.join(local_dir);
    std::fs::create_dir_all(&model_path)?;

    info!("Downloading model {} to {}", hf_repo, model_path.display());

    let hf_repo = hf_repo.to_string();
    let model_path_str = model_path.to_string_lossy().to_string();
    let model_id_owned = model_id.map(|s| s.to_string());

    // Check available disk space before download
    let model_path_for_check = model_path.clone();
    let available_gb = tokio::task::spawn_blocking(move || -> f64 {
        crate::hardware::available_disk_space_gb(&model_path_for_check)
    })
    .await?;

    info!("Available disk space: {:.1} GB", available_gb);
    info!("Model size: {:.1} GB", model_size_gb);

    // Fail if insufficient disk space (require 10% buffer)
    let required_space = model_size_gb / 0.9; // Account for 10% buffer
    if available_gb > 0.0 && available_gb < required_space {
        anyhow::bail!(
            "Insufficient disk space: {:.1} GB available, but model requires {:.1} GB ({:.1} GB + 10% buffer). \
             Free up at least {:.1} GB before downloading this model.",
            available_gb,
            required_space,
            model_size_gb,
            required_space - available_gb
        );
    }

    // Spawn download in background thread with progress monitoring
    let download_handle = tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        use std::process::{Command, Stdio};

        let python_path = if cfg!(windows) {
            std::env::current_dir()?
                .join("python")
                .join("venv")
                .join("Scripts")
                .join("python.exe")
        } else {
            std::env::current_dir()?
                .join("python")
                .join("venv")
                .join("bin")
                .join("python")
        };

        if !python_path.exists() {
            return Err(anyhow::anyhow!(
                "Python venv not found at {}",
                python_path.display()
            ));
        }

        info!("Using Python HuggingFace Hub to download (more reliable for large files)");

        // Use forward slashes for Python compatibility on Windows
        let model_path_python = model_path_str.replace('\\', "/");

        let mut child = Command::new(&python_path)
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
                    hf_repo, model_path_python
                ),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn Python process: {}", e))?;

        // Poll directory size while download is running
        let model_path_buf = PathBuf::from(&model_path_str);
        let mut last_size_gb = 0.0;

        loop {
            // Check for cancellation
            if let Some(ref flag) = cancel_flag {
                if flag.load(std::sync::atomic::Ordering::SeqCst) {
                    info!("Download cancelled, killing process");
                    #[cfg(unix)]
                    {
                        unsafe {
                            libc::kill(child.id() as i32, libc::SIGTERM);
                        }
                    }
                    #[cfg(windows)]
                    {
                        let _ = child.kill();
                    }
                    let _ = child.wait();
                    return Err(anyhow::anyhow!("Download cancelled by user"));
                }
            }

            // Check if process is still running
            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        // Capture stderr for error details
                        let stderr = child
                            .stderr
                            .take()
                            .map(|mut s| {
                                use std::io::Read;
                                let mut buf = String::new();
                                s.read_to_string(&mut buf).ok();
                                buf
                            })
                            .unwrap_or_default();
                        return Err(anyhow::anyhow!(
                            "Model download failed with exit code: {}\nError: {}",
                            status,
                            stderr.trim()
                        ));
                    }
                    info!("Model download process completed");
                    break;
                }
                Ok(None) => {
                    // Still running - report progress
                    if let Ok(size) = dir_size_bytes(&model_path_buf) {
                        let gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
                        if (gb - last_size_gb).abs() > 0.1 {
                            // Only report if changed by >100MB
                            progress_callback(gb, model_size_gb);
                            last_size_gb = gb;
                            info!(
                                "Download progress: {:.2} / {:.1} GB ({:.0}%)",
                                gb,
                                model_size_gb,
                                (gb / model_size_gb * 100.0).min(100.0)
                            );
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
                Err(e) => return Err(anyhow::anyhow!("Failed to check process status: {}", e)),
            }
        }

        info!("Model downloaded successfully");

        // Report final size
        if let Ok(size) = dir_size_bytes(&model_path_buf) {
            let gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
            progress_callback(gb, model_size_gb);
            info!(
                "Final model size: {:.2} GB (expected {:.1} GB)",
                gb, model_size_gb
            );
        }

        // Create marker file to indicate download is complete
        let marker_path = model_path_buf.join(".download_complete");
        std::fs::write(
            &marker_path,
            format!("Downloaded: {}\nSize: {:.2} GB\n", hf_repo, model_size_gb),
        )
        .map_err(|e| anyhow::anyhow!("Failed to create download marker: {}", e))?;

        // Create model_id marker if provided
        if let Some(mid) = model_id_owned {
            let id_marker_path = model_path_buf.join(".model_id");
            std::fs::write(&id_marker_path, mid)
                .map_err(|e| anyhow::anyhow!("Failed to create model_id marker: {}", e))?;
        }

        info!("Created download complete marker");

        Ok(())
    });

    download_handle.await??;

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
