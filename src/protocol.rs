//! WebSocket message types for worker <-> server communication.

use serde::{Deserialize, Serialize};

use crate::config::WorkerConstraints;

// ---------------------------------------------------------------------------
// Worker -> Server messages
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerMessage {
    Hello {
        worker_id: String,
        api_key: String,
        name: String,
        gpus: Vec<GpuInfo>,
        ram_total_gb: f64,
        cpu_cores: usize,
        disk_space_gb: f64,
        platform: String,
        models_cached: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        constraints: Option<WorkerConstraints>,
    },
    Heartbeat {
        worker_id: String,
        gpus: Vec<GpuInfo>,
        ram_free_gb: f64,
        disk_space_gb: f64,
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
    TaskFailed {
        task_id: String,
        error_message: String,
        phase: String,
    },
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
    TaskAssign {
        task_id: String,
        job_id: String,
        task_type: String,
        scene: serde_json::Value,
        project: serde_json::Value,
        model_id: String,
        model_config: Box<ModelConfig>,
        pipeline_config: Box<serde_json::Value>,
        upload_url: String,
        last_frame_url: Option<String>,
        input_video_url: Option<String>,
    },
    TaskCancel {
        task_id: String,
        reason: String,
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
