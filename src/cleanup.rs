//! Automatic cleanup of temporary files and old output directories.

use std::path::Path;
use std::time::{Duration, SystemTime};

use tokio::fs;
use tracing::{debug, info, warn};

/// Cleanup configuration
#[derive(Debug, Clone)]
pub struct CleanupConfig {
    /// Base directory (~/.anime-worker or wherever models_dir is)
    pub base_dir: std::path::PathBuf,
    /// How old files must be before deletion (in hours)
    pub retention_hours: u64,
}

impl CleanupConfig {
    pub fn from_models_dir(models_dir: &Path, retention_hours: u64) -> Self {
        let base_dir = models_dir.parent().unwrap_or(models_dir).to_path_buf();

        Self {
            base_dir,
            retention_hours,
        }
    }
}

/// Run cleanup - removes old files from output/ and tmp/ directories
pub async fn run_cleanup(config: &CleanupConfig) -> anyhow::Result<()> {
    let retention_duration = Duration::from_secs(config.retention_hours * 3600);
    let now = SystemTime::now();

    let output_dir = config.base_dir.join("output");
    let tmp_dir = config.base_dir.join("tmp");

    let mut total_removed = 0;

    // Cleanup output directories
    if output_dir.exists() {
        match cleanup_output_dirs(&output_dir, now, retention_duration).await {
            Ok(count) => {
                if count > 0 {
                    info!("Removed {} old output directories", count);
                }
                total_removed += count;
            }
            Err(e) => warn!("Failed to cleanup output directories: {}", e),
        }
    }

    // Cleanup tmp PNG files
    if tmp_dir.exists() {
        match cleanup_tmp_files(&tmp_dir, now, retention_duration).await {
            Ok(count) => {
                if count > 0 {
                    info!("Removed {} old temporary files", count);
                }
                total_removed += count;
            }
            Err(e) => warn!("Failed to cleanup tmp files: {}", e),
        }
    }

    if total_removed > 0 {
        info!("Cleanup complete: {} items removed", total_removed);
    } else {
        debug!("Cleanup complete: no old files found");
    }

    Ok(())
}

/// Remove output directories older than retention period
async fn cleanup_output_dirs(
    output_dir: &Path,
    now: SystemTime,
    retention: Duration,
) -> anyhow::Result<usize> {
    let mut count = 0;
    let mut entries = fs::read_dir(output_dir).await?;

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();

        // Only process directories
        if !path.is_dir() {
            continue;
        }

        // Check modification time
        if let Ok(metadata) = entry.metadata().await {
            if let Ok(modified) = metadata.modified() {
                if let Ok(age) = now.duration_since(modified) {
                    if age > retention {
                        debug!("Removing old directory: {}", path.display());
                        if let Err(e) = fs::remove_dir_all(&path).await {
                            warn!("Failed to remove directory {}: {}", path.display(), e);
                        } else {
                            count += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(count)
}

/// Remove PNG files from tmp directory older than retention period
async fn cleanup_tmp_files(
    tmp_dir: &Path,
    now: SystemTime,
    retention: Duration,
) -> anyhow::Result<usize> {
    let mut count = 0;
    let mut entries = fs::read_dir(tmp_dir).await?;

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();

        // Only process PNG files
        if !path.is_file() || path.extension().and_then(|s| s.to_str()) != Some("png") {
            continue;
        }

        // Check modification time
        if let Ok(metadata) = entry.metadata().await {
            if let Ok(modified) = metadata.modified() {
                if let Ok(age) = now.duration_since(modified) {
                    if age > retention {
                        debug!("Removing old file: {}", path.display());
                        if let Err(e) = fs::remove_file(&path).await {
                            warn!("Failed to remove file {}: {}", path.display(), e);
                        } else {
                            count += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cleanup_respects_retention() {
        let temp_dir = std::env::temp_dir().join(format!("cleanup-test-{}", std::process::id()));
        let output_dir = temp_dir.join("output");
        let tmp_dir = temp_dir.join("tmp");

        // Create test directories
        tokio::fs::create_dir_all(&output_dir).await.unwrap();
        tokio::fs::create_dir_all(&tmp_dir).await.unwrap();

        // Create a recent directory (should NOT be deleted)
        let recent_dir = output_dir.join("recent-job");
        tokio::fs::create_dir_all(&recent_dir).await.unwrap();

        // Create a recent file (should NOT be deleted)
        let recent_file = tmp_dir.join("recent.png");
        tokio::fs::write(&recent_file, b"test").await.unwrap();

        // Run cleanup with 24 hour retention
        let config = CleanupConfig {
            base_dir: temp_dir.clone(),
            retention_hours: 24,
        };

        run_cleanup(&config).await.unwrap();

        // Recent items should still exist
        assert!(
            recent_dir.exists(),
            "Recent directory should not be deleted"
        );
        assert!(recent_file.exists(), "Recent file should not be deleted");

        // Cleanup test directory
        tokio::fs::remove_dir_all(&temp_dir).await.ok();
    }
}
