//! HTTP multipart upload of generated files to the server.

use std::path::Path;

use reqwest::multipart;
use tracing::info;

/// Upload a generated file to the server.
pub async fn upload_file(
    server_url: &str,
    upload_path: &str,
    api_key: &str,
    file_path: &Path,
) -> anyhow::Result<()> {
    let url = format!("{}{}", server_url.trim_end_matches('/'), upload_path);
    info!("Uploading {} to {}", file_path.display(), url);

    let file_name = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("output.mp4")
        .to_string();

    let file_bytes = tokio::fs::read(file_path).await?;
    let part = multipart::Part::bytes(file_bytes)
        .file_name(file_name)
        .mime_str("application/octet-stream")?;

    let form = multipart::Form::new().part("file", part);

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header("X-API-Key", api_key)
        .multipart(form)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Upload failed ({}): {}", status, body);
    }

    info!("Upload successful");
    Ok(())
}

/// Upload a lastframe PNG (for continuity chaining).
pub async fn upload_lastframe(
    server_url: &str,
    task_id: &str,
    api_key: &str,
    file_path: &Path,
) -> anyhow::Result<()> {
    let upload_path = format!("/worker/upload/{}/lastframe", task_id);
    upload_file(server_url, &upload_path, api_key, file_path).await
}

/// Download a file from the server (e.g., lastframe from predecessor task).
pub async fn download_file(
    server_url: &str,
    path: &str,
    api_key: &str,
    dest: &Path,
) -> anyhow::Result<()> {
    let url = format!("{}{}", server_url.trim_end_matches('/'), path);
    info!("Downloading {} to {}", url, dest.display());

    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .header("X-API-Key", api_key)
        .send()
        .await?;

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
