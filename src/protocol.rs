//! WebSocket message types for worker <-> server communication.

use serde::{Deserialize, Serialize};

use crate::config::WorkerConstraints;

// ---------------------------------------------------------------------------
// Worker -> Server messages
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareStats {
    pub ram: RamStats,
    pub cpu: CpuStats,
    pub disk: DiskStats,
    pub gpus: Vec<GpuInfo>,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RamStats {
    pub total_gb: f64,
    pub used_gb: f64,
    pub free_gb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuStats {
    pub cores: usize,
    pub usage_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskStats {
    pub total_gb: f64,
    pub used_gb: f64,
    pub free_gb: f64,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerMessage {
    Hello {
        worker_id: String,
        api_key: String,
        name: String,
        hardware_stats: HardwareStats,
        platform: String,
        models_cached: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        constraints: Box<Option<WorkerConstraints>>,
    },
    Heartbeat {
        worker_id: String,
        hardware_stats: HardwareStats,
        models_cached: Vec<String>,
    },
    RequestTask {
        worker_id: String,
    },
    TaskProgress {
        task_id: String,
        progress: f64,
        message: String,
        phase: String,
    },
    ModelProgress {
        task_id: String,
        model_id: String,
        downloaded_gb: f64,
        total_gb: f64,
    },
    TaskComplete {
        task_id: String,
        result_filename: String,
        metadata: serde_json::Value,
    },
    Log {
        #[serde(skip_serializing_if = "Option::is_none")]
        job_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        task_id: Option<String>,
        level: String,
        message: String,
        source: String,
        timestamp: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
    },
    LogBatch {
        logs: Vec<LogEntry>,
    },
}

/// A single log entry for batching
#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<String>,
    pub level: String,
    pub message: String,
    pub source: String,
    pub timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Server -> Worker messages
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    HelloAck {
        server_version: String,
    },
    JobBatchAssign {
        job_id: String,
        task_type: String,
        tasks: Vec<serde_json::Value>, // Array of {task_id, scene_id, scene, upload_url, ...}
        project: Box<serde_json::Value>,
        model_id: String,
        model_config: Box<ModelConfig>,
        pipeline_config: Box<serde_json::Value>,
    },
    TaskCancel {
        task_id: String,
        reason: String,
    },
    DeleteModel {
        model_id: String,
        local_dir: String,
    },
    DownloadModel {
        model_id: String,
        model_name: String,
        hf_repo: String,
    },
    TasksAvailable {},
    HeartbeatAck {},
    Error {
        message: String,
    },
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hf_repo: String,
    pub local_dir: String,
    pub model_class: String,
    pub model_size_gb: f64,
    #[serde(flatten)]
    pub extra: serde_json::Value,
}
