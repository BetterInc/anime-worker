//! Hardware metrics collection (GPU, RAM) for monitoring.

use serde_json::json;
use std::process::Command;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{interval, Duration};

use crate::protocol::LogEntry;

#[derive(Debug, Clone)]
pub struct GpuMetrics {
    pub id: usize,
    pub sm: f32,  // Shader/compute utilization %
    pub mem: f32, // Memory utilization %
    pub enc: f32, // Encoder utilization %
    pub dec: f32, // Decoder utilization %
}

#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    pub used_gb: f64,
    pub total_gb: f64,
    pub usage_pct: f64,
}

/// Spawn background task to collect hardware metrics periodically
pub fn spawn_metrics_collector(
    log_tx: tokio::sync::mpsc::Sender<LogEntry>,
    current_job_id: Arc<Mutex<Option<String>>>,
) {
    tokio::spawn(async move {
        let mut tick = interval(Duration::from_secs(10));

        loop {
            tick.tick().await;

            // Only collect metrics during active job
            let job_id = current_job_id.lock().await.clone();
            if job_id.is_none() {
                continue;
            }

            // Collect GPU metrics
            if let Ok(gpus) = collect_gpu_metrics().await {
                for gpu in gpus {
                    let log = LogEntry {
                        job_id: job_id.clone(),
                        task_id: None,
                        level: "DEBUG".to_string(),
                        message: format!(
                            "GPU {}: sm={:.0}%, mem={:.0}%, enc={:.0}%, dec={:.0}%",
                            gpu.id, gpu.sm, gpu.mem, gpu.enc, gpu.dec
                        ),
                        source: "metrics".to_string(),
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        metadata: Some(json!({
                            "gpu_id": gpu.id,
                            "sm_util": gpu.sm,
                            "mem_util": gpu.mem,
                            "enc_util": gpu.enc,
                            "dec_util": gpu.dec,
                        })),
                    };
                    let _ = log_tx.send(log).await;
                }
            }

            // Collect memory metrics
            if let Ok(mem) = collect_memory_metrics().await {
                let log = LogEntry {
                    job_id: job_id.clone(),
                    task_id: None,
                    level: "DEBUG".to_string(),
                    message: format!(
                        "Memory: {:.1}/{:.1} GB ({:.0}%)",
                        mem.used_gb, mem.total_gb, mem.usage_pct
                    ),
                    source: "metrics".to_string(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    metadata: Some(json!({
                        "ram_used_gb": mem.used_gb,
                        "ram_total_gb": mem.total_gb,
                        "ram_usage_pct": mem.usage_pct,
                    })),
                };
                let _ = log_tx.send(log).await;
            }
        }
    });
}

/// Collect GPU metrics using nvidia-smi dmon
async fn collect_gpu_metrics() -> anyhow::Result<Vec<GpuMetrics>> {
    let output = tokio::task::spawn_blocking(|| {
        Command::new("nvidia-smi")
            .args(["dmon", "-c", "1", "-s", "u"])
            .output()
    })
    .await??;

    if !output.status.success() {
        anyhow::bail!("nvidia-smi dmon failed");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    // Parse output (skip header lines starting with #)
    for line in stdout.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 5 {
            // Format: gpu sm mem enc dec jpg ofa
            if let (Ok(id), Ok(sm), Ok(mem), Ok(enc), Ok(dec)) = (
                parts[0].parse::<usize>(),
                parts[1].parse::<f32>(),
                parts[2].parse::<f32>(),
                parts[3].parse::<f32>(),
                parts[4].parse::<f32>(),
            ) {
                gpus.push(GpuMetrics {
                    id,
                    sm,
                    mem,
                    enc,
                    dec,
                });
            }
        }
    }

    if gpus.is_empty() {
        anyhow::bail!("No GPU metrics parsed");
    }

    Ok(gpus)
}

/// Collect system memory metrics
async fn collect_memory_metrics() -> anyhow::Result<MemoryMetrics> {
    use sysinfo::System;

    let sys = tokio::task::spawn_blocking(|| {
        let mut sys = System::new_all();
        sys.refresh_memory();
        sys
    })
    .await?;

    let total_kb = sys.total_memory();
    let used_kb = sys.used_memory();

    let total_gb = total_kb as f64 / 1_048_576.0; // KB to GB
    let used_gb = used_kb as f64 / 1_048_576.0;
    let usage_pct = (used_gb / total_gb * 100.0).min(100.0);

    Ok(MemoryMetrics {
        used_gb,
        total_gb,
        usage_pct,
    })
}
