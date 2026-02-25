//! Cross-platform hardware detection (GPU via nvidia-smi, RAM via sysinfo).

use std::process::Command;

use sysinfo::System;
use tracing::{info, warn};

use crate::protocol::GpuInfo;

/// Detect GPUs by parsing nvidia-smi output. Works on Linux and Windows.
pub fn detect_gpus() -> Vec<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let mut gpus = Vec::new();
            for line in stdout.lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    let name = parts[0].to_string();
                    let vram_total_mb = parts[1].parse::<u64>().unwrap_or(0);
                    let vram_free_mb = parts[2].parse::<u64>().unwrap_or(0);
                    gpus.push(GpuInfo {
                        name,
                        vram_total_mb,
                        vram_free_mb,
                    });
                }
            }
            if !gpus.is_empty() {
                info!("Detected {} GPU(s):", gpus.len());
                for (i, gpu) in gpus.iter().enumerate() {
                    info!(
                        "  GPU {}: {} ({} MB total, {} MB free)",
                        i, gpu.name, gpu.vram_total_mb, gpu.vram_free_mb
                    );
                }
            }
            gpus
        }
        Ok(out) => {
            warn!(
                "nvidia-smi returned non-zero: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            Vec::new()
        }
        Err(e) => {
            warn!("nvidia-smi not found or failed: {}", e);
            Vec::new()
        }
    }
}

/// Get total system RAM in GB.
pub fn total_ram_gb() -> f64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Get free system RAM in GB.
pub fn free_ram_gb() -> f64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Get the platform string (linux, windows, darwin).
pub fn platform() -> &'static str {
    if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "darwin"
    } else {
        "unknown"
    }
}

/// Get number of CPU cores (logical processors).
pub fn cpu_cores() -> usize {
    num_cpus::get()
}

/// Get available disk space in GB for a given path (cross-platform).
/// Returns 0.0 if unable to determine.
pub fn available_disk_space_gb(path: &std::path::Path) -> f64 {
    use sysinfo::Disks;

    let disks = Disks::new_with_refreshed_list();

    // Find the disk that contains this path
    let mut best_match: Option<&sysinfo::Disk> = None;
    let mut best_match_len = 0;

    for disk in &disks {
        let mount_point = disk.mount_point();
        if let Ok(_rel) = path.strip_prefix(mount_point) {
            let mount_len = mount_point.as_os_str().len();
            if mount_len > best_match_len {
                best_match = Some(disk);
                best_match_len = mount_len;
            }
        } else if path.starts_with(mount_point) {
            let mount_len = mount_point.as_os_str().len();
            if mount_len > best_match_len {
                best_match = Some(disk);
                best_match_len = mount_len;
            }
        }
    }

    if let Some(disk) = best_match {
        let available_bytes = disk.available_space();
        return available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    }

    // Fallback: use first disk or 0
    if let Some(disk) = disks.first() {
        let available_bytes = disk.available_space();
        available_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    } else {
        0.0
    }
}
