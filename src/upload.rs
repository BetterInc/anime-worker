//! HTTP multipart upload of generated files to the server.

use std::path::Path;

use reqwest::multipart;
use tracing::info;

/// Upload a generated file to the server with retry logic.
pub async fn upload_file(
    server_url: &str,
    upload_path: &str,
    api_key: &str,
    file_path: &Path,
) -> anyhow::Result<()> {
    const MAX_RETRIES: u32 = 3;
    const INITIAL_BACKOFF_MS: u64 = 1000;
    const TIMEOUT_SECS: u64 = 300; // 5 minutes for large files

    let url = format!("{}{}", server_url.trim_end_matches('/'), upload_path);
    let file_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("output.mp4")
        .to_string();

    info!(
        "Uploading {} to {} (max {} retries)",
        file_path.display(),
        url,
        MAX_RETRIES
    );

    let mut last_error = None;

    for attempt in 1..=MAX_RETRIES {
        if attempt > 1 {
            let backoff_ms = INITIAL_BACKOFF_MS * 2_u64.pow(attempt - 2);
            info!(
                "Retry attempt {}/{} after {}ms",
                attempt, MAX_RETRIES, backoff_ms
            );
            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
        }

        match upload_attempt(&url, api_key, file_path, &file_name, TIMEOUT_SECS).await {
            Ok(()) => {
                info!("Upload successful (attempt {}/{})", attempt, MAX_RETRIES);
                return Ok(());
            }
            Err(e) => {
                info!("Upload attempt {}/{} failed: {}", attempt, MAX_RETRIES, e);
                last_error = Some(e);
            }
        }
    }

    Err(last_error
        .unwrap_or_else(|| anyhow::anyhow!("Upload failed after {} retries", MAX_RETRIES)))
}

/// Single upload attempt with timeout.
async fn upload_attempt(
    url: &str,
    api_key: &str,
    file_path: &Path,
    file_name: &str,
    timeout_secs: u64,
) -> anyhow::Result<()> {
    let file_bytes = tokio::fs::read(file_path).await?;
    let part = multipart::Part::bytes(file_bytes)
        .file_name(file_name.to_string())
        .mime_str("application/octet-stream")?;

    let form = multipart::Form::new().part("file", part);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()?;

    let resp = client
        .post(url)
        .header("X-API-Key", api_key)
        .multipart(form)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Upload failed ({}): {}", status, body);
    }

    Ok(())
}

/// Download a file from the server (e.g., lastframe from predecessor task) with retry logic.
pub async fn download_file(
    server_url: &str,
    path: &str,
    api_key: &str,
    dest: &Path,
) -> anyhow::Result<()> {
    const MAX_RETRIES: u32 = 3;
    const INITIAL_BACKOFF_MS: u64 = 1000;
    const TIMEOUT_SECS: u64 = 60;

    let url = format!("{}{}", server_url.trim_end_matches('/'), path);
    info!(
        "Downloading {} to {} (max {} retries)",
        url,
        dest.display(),
        MAX_RETRIES
    );

    let mut last_error = None;

    for attempt in 1..=MAX_RETRIES {
        if attempt > 1 {
            let backoff_ms = INITIAL_BACKOFF_MS * 2_u64.pow(attempt - 2);
            info!(
                "Retry attempt {}/{} after {}ms",
                attempt, MAX_RETRIES, backoff_ms
            );
            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
        }

        match download_attempt(&url, api_key, dest, TIMEOUT_SECS).await {
            Ok(()) => {
                info!("Download successful (attempt {}/{})", attempt, MAX_RETRIES);
                return Ok(());
            }
            Err(e) => {
                info!("Download attempt {}/{} failed: {}", attempt, MAX_RETRIES, e);
                last_error = Some(e);
            }
        }
    }

    Err(last_error
        .unwrap_or_else(|| anyhow::anyhow!("Download failed after {} retries", MAX_RETRIES)))
}

/// Single download attempt with timeout.
async fn download_attempt(
    url: &str,
    api_key: &str,
    dest: &Path,
    timeout_secs: u64,
) -> anyhow::Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()?;

    let resp = client.get(url).header("X-API-Key", api_key).send().await?;

    if !resp.status().is_success() {
        let status = resp.status();
        anyhow::bail!("Download failed ({})", status);
    }

    let bytes = resp.bytes().await?;
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(dest, &bytes).await?;

    info!("Downloaded {} bytes", bytes.len());
    Ok(())
}
