//! WebSocket client with auto-reconnect and message dispatch.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::Mutex;
use tokio::time;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

use crate::config::WorkerConfig;
use crate::hardware;
use crate::models;
use crate::protocol::{CpuStats, DiskStats, HardwareStats, RamStats, ServerMessage, WorkerMessage};
use crate::runner::{self, InferenceJob, PythonOutput, SceneTask};
use crate::upload;

/// Build hardware stats snapshot from current system state.
fn build_hardware_stats(config: &WorkerConfig) -> HardwareStats {
    let gpus = hardware::detect_gpus();

    // Apply resource limits from constraints (if configured)
    // If no limits set, use 90% of actual hardware (10% buffer for system)
    let cpu_cores = config
        .constraints
        .cpu_limit
        .unwrap_or_else(|| (hardware::cpu_cores() as f64 * 0.9).max(1.0) as usize);

    // Get actual RAM stats
    let ram_actual_total = hardware::total_ram_gb();
    let ram_actual_free = hardware::free_ram_gb();
    let ram_actual_used = ram_actual_total - ram_actual_free;

    // Apply constraint to total if configured
    let ram_total_gb = config
        .constraints
        .ram_limit_gb
        .unwrap_or(ram_actual_total);
    let ram_used_gb = ram_actual_used;
    let ram_free_gb = ram_actual_free;

    let disk_total_gb = hardware::total_disk_space_gb(&config.models_dir);
    let disk_free_gb = config
        .constraints
        .disk_limit_gb
        .unwrap_or_else(|| hardware::available_disk_space_gb(&config.models_dir) * 0.9);
    let disk_used_gb = disk_total_gb - disk_free_gb;

    let cpu_usage_percent = hardware::cpu_usage_percent();

    HardwareStats {
        ram: RamStats {
            total_gb: ram_total_gb,
            used_gb: ram_used_gb,
            free_gb: ram_free_gb,
        },
        cpu: CpuStats {
            cores: cpu_cores,
            usage_percent: cpu_usage_percent,
        },
        disk: DiskStats {
            total_gb: disk_total_gb,
            used_gb: disk_used_gb,
            free_gb: disk_free_gb,
        },
        gpus,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_string(),
    }
}

/// Run the worker client loop with auto-reconnect.
pub async fn run(config: Arc<WorkerConfig>) {
    loop {
        info!("Connecting to {}...", config.server_url);
        match connect_and_run(&config).await {
            Ok(()) => info!("Connection closed cleanly"),
            Err(e) => error!("Connection error: {}", e),
        }
        info!("Reconnecting in 5 seconds...");
        time::sleep(Duration::from_secs(5)).await;
    }
}

async fn connect_and_run(config: &WorkerConfig) -> anyhow::Result<()> {
    let ws_url = format!(
        "{}/ws/worker",
        config
            .server_url
            .replace("http://", "ws://")
            .replace("https://", "wss://")
    );

    let (ws_stream, _) = connect_async(&ws_url).await?;
    let (mut write, mut read) = ws_stream.split();
    info!("WebSocket connected");

    // Send hello
    let cached = models::list_cached_models(&config.models_dir);

    // Build hardware stats
    let hardware_stats = build_hardware_stats(config);

    let hello = WorkerMessage::Hello {
        worker_id: config.worker_id.clone(),
        api_key: config.api_key.clone(),
        name: config.worker_name.clone(),
        hardware_stats,
        platform: hardware::platform().to_string(),
        models_cached: cached.clone(),
        constraints: Box::new(Some(config.constraints.clone())),
    };
    let hello_json = serde_json::to_string(&hello)?;
    write.send(Message::Text(hello_json)).await?;

    // Wait for hello_ack
    let ack_msg = tokio::time::timeout(Duration::from_secs(10), read.next())
        .await?
        .ok_or_else(|| anyhow::anyhow!("Connection closed during hello"))??;

    if let Message::Text(text) = ack_msg {
        let msg: ServerMessage = serde_json::from_str(&text)?;
        match msg {
            ServerMessage::HelloAck { server_version } => {
                info!("Authenticated. Server version: {}", server_version);
            }
            ServerMessage::Error { message } => {
                anyhow::bail!("Server rejected hello: {}", message);
            }
            _ => {
                anyhow::bail!("Unexpected message during hello: {:?}", msg);
            }
        }
    }

    // Shared write half for sending from multiple tasks
    let write = Arc::new(Mutex::new(write));

    // Request initial task
    {
        let msg = serde_json::to_string(&WorkerMessage::RequestTask {
            worker_id: config.worker_id.clone(),
        })?;
        write.lock().await.send(Message::Text(msg)).await?;
    }

    // Heartbeat task
    let hb_write = write.clone();
    let hb_config = config.clone();
    let heartbeat_handle = tokio::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(hb_config.heartbeat_interval_secs));
        loop {
            interval.tick().await;
            let cached = models::list_cached_models(&hb_config.models_dir);
            let hardware_stats = build_hardware_stats(&hb_config);

            let msg = WorkerMessage::Heartbeat {
                worker_id: hb_config.worker_id.clone(),
                hardware_stats,
                models_cached: cached,
            };
            let json = match serde_json::to_string(&msg) {
                Ok(j) => j,
                Err(_) => continue,
            };
            if hb_write
                .lock()
                .await
                .send(Message::Text(json))
                .await
                .is_err()
            {
                break;
            }
        }
    });

    // Active inference handle (for cancellation)
    let active_inference: Arc<Mutex<Option<runner::RunningInference>>> = Arc::new(Mutex::new(None));

    // Message processing loop
    while let Some(msg_result) = read.next().await {
        let msg = msg_result?;
        match msg {
            Message::Text(text) => {
                let server_msg: ServerMessage = match serde_json::from_str(&text) {
                    Ok(m) => m,
                    Err(e) => {
                        warn!("Failed to parse server message: {}", e);
                        continue;
                    }
                };

                match server_msg {
                    ServerMessage::HeartbeatAck {} => {}

                    ServerMessage::TasksAvailable { .. } => {
                        // Server says tasks are available, request one
                        let msg = serde_json::to_string(&WorkerMessage::RequestTask {
                            worker_id: config.worker_id.clone(),
                        })?;
                        write.lock().await.send(Message::Text(msg)).await?;
                    }

                    ServerMessage::JobBatchAssign {
                        job_id,
                        task_type,
                        tasks,
                        project,
                        model_id,
                        model_config,
                        pipeline_config,
                    } => {
                        info!(
                            "Received job batch {} ({} scene(s), type: {})",
                            &job_id[..8],
                            tasks.len(),
                            task_type
                        );

                        let config = config.clone();
                        let write = write.clone();
                        let active = active_inference.clone();

                        tokio::spawn(async move {
                            let result = process_job_batch(
                                &config,
                                &write,
                                &active,
                                &job_id,
                                &task_type,
                                tasks,
                                project,
                                &model_id,
                                model_config,
                                pipeline_config,
                            )
                            .await;

                            if let Err(e) = result {
                                error!("Job {} failed: {}", &job_id[..8], e);
                            }

                            // Request next job
                            let req = serde_json::to_string(&WorkerMessage::RequestTask {
                                worker_id: config.worker_id.clone(),
                            })
                            .unwrap_or_default();
                            let _ = write.lock().await.send(Message::Text(req)).await;
                        });
                    }

                    ServerMessage::TaskCancel { task_id, reason } => {
                        info!("Task {} cancelled: {}", &task_id[..8], reason);
                        if let Some(handle) = active_inference.lock().await.as_ref() {
                            handle.cancel().await;
                        }
                    }

                    ServerMessage::Error { message } => {
                        error!("Server error: {}", message);
                    }

                    _ => {}
                }
            }
            Message::Close(_) => {
                info!("Server closed connection");
                break;
            }
            Message::Ping(data) => {
                write.lock().await.send(Message::Pong(data)).await?;
            }
            _ => {}
        }
    }

    heartbeat_handle.abort();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Process a job batch (all scenes from one job), keeping the model loaded
async fn process_job_batch<S>(
    config: &WorkerConfig,
    write: &Arc<Mutex<S>>,
    active_inference: &Arc<Mutex<Option<runner::RunningInference>>>,
    job_id: &str,
    task_type: &str,
    tasks: Vec<serde_json::Value>, // From API: [{task_id, scene_id, scene, upload_url, ...}]
    project: Box<serde_json::Value>,
    model_id: &str,
    model_config: Box<crate::protocol::ModelConfig>,
    pipeline_config: Box<serde_json::Value>,
) -> anyhow::Result<()>
where
    S: SinkExt<Message> + Unpin + Send + 'static,
    <S as futures_util::Sink<Message>>::Error: std::error::Error + Send + Sync + 'static,
{
    if tasks.is_empty() {
        return Ok(());
    }

    info!(
        "Processing {} scene(s) for job {} (model: {}, type: {})",
        tasks.len(),
        &job_id[..8],
        model_id,
        task_type
    );

    let first_task_id = tasks[0]
        .get("task_id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    // 1. Ensure model is cached
    if !models::is_model_cached(&config.models_dir, &model_config.local_dir) {
        info!("Model {} not cached, downloading...", model_id);
        let task_id_owned = first_task_id.to_string();
        let model_id_owned = model_id.to_string();
        let write_clone = write.clone();

        let progress_cb = move |downloaded_gb: f64, total_gb: f64| {
            let msg = WorkerMessage::ModelProgress {
                task_id: task_id_owned.clone(),
                model_id: model_id_owned.clone(),
                downloaded_gb,
                total_gb,
            };
            if let Ok(json) = serde_json::to_string(&msg) {
                let write = write_clone.clone();
                if let Ok(handle) = tokio::runtime::Handle::try_current() {
                    handle.spawn(async move {
                        let _ = write.lock().await.send(Message::Text(json)).await;
                    });
                }
            }
        };

        models::download_model(
            &model_config.hf_repo,
            &config.models_dir,
            &model_config.local_dir,
            model_config.model_size_gb,
            progress_cb,
        )
        .await?;
    }

    // 2. Download last frame if first task needs it
    let last_frame_path = if let Some(url) = tasks[0].get("last_frame_url").and_then(|v| v.as_str())
    {
        let dest = config
            .models_dir
            .parent()
            .unwrap_or(&config.models_dir)
            .join("tmp")
            .join(format!("{}_lastframe.png", job_id));
        upload::download_file(&config.server_url, url, &config.api_key, &dest).await?;
        Some(dest)
    } else {
        None
    };

    // 3. Create output directory
    let output_dir = config
        .models_dir
        .parent()
        .unwrap_or(&config.models_dir)
        .join("output")
        .join(job_id);
    tokio::fs::create_dir_all(&output_dir).await?;

    // 4. Build scene list for Python
    let scene_tasks: Vec<SceneTask> = tasks
        .iter()
        .map(|t| {
            let task_id = t
                .get("task_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let mut scene = t.get("scene").cloned().unwrap_or(serde_json::json!({}));
            if let Some(obj) = scene.as_object_mut() {
                obj.insert(
                    "task_id".to_string(),
                    serde_json::Value::String(task_id.clone()),
                );
            }
            SceneTask { task_id, scene }
        })
        .collect();

    // 5. Spawn Python with ALL scenes
    let model_path = config
        .models_dir
        .join(&model_config.local_dir)
        .to_string_lossy()
        .to_string();

    let job = InferenceJob {
        task_id: first_task_id.to_string(),
        task_type: task_type.to_string(),
        scene: None,
        scenes: Some(scene_tasks),
        project: *project,
        model_path,
        model_config: serde_json::to_value(&*model_config)?,
        pipeline_config: *pipeline_config,
        output_dir: output_dir.to_string_lossy().to_string(),
        last_frame_path: last_frame_path.map(|p| p.to_string_lossy().to_string()),
        input_video_path: None,
    };

    let (mut output_rx, handle) =
        runner::spawn_inference(&config.python_path, &config.python_scripts_dir, job).await?;

    *active_inference.lock().await = Some(handle);

    // 6. Process Python output and report progress
    let mut completed_scenes: HashMap<String, (String, serde_json::Value)> = HashMap::new();

    while let Some(output) = output_rx.recv().await {
        match output {
            PythonOutput::Progress { pct, message } => {
                let scaled_pct = 10.0 + pct * 0.85;
                let msg = WorkerMessage::TaskProgress {
                    task_id: first_task_id.to_string(),
                    progress: scaled_pct,
                    message,
                    phase: "generating".to_string(),
                };
                if let Ok(json) = serde_json::to_string(&msg) {
                    let _ = write.lock().await.send(Message::Text(json)).await;
                }
            }
            PythonOutput::Complete { metadata } => {
                if let Some(scenes_meta) = metadata.get("scenes").and_then(|v| v.as_array()) {
                    for scene_meta in scenes_meta.iter() {
                        if let Some(task_id) = scene_meta.get("task_id").and_then(|v| v.as_str()) {
                            if let Some(filename) =
                                scene_meta.get("filename").and_then(|v| v.as_str())
                            {
                                completed_scenes.insert(
                                    task_id.to_string(),
                                    (filename.to_string(), scene_meta.clone()),
                                );
                            }
                        }
                    }
                }
            }
            PythonOutput::Error { message } => {
                anyhow::bail!("Python inference error: {}", message);
            }
        }
    }

    // 7. Upload results for each task
    for task in &tasks {
        let task_id = task.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
        let upload_url = task
            .get("upload_url")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if let Some((filename, metadata)) = completed_scenes.get(task_id) {
            let file_path = output_dir.join(filename);

            upload::upload_file(&config.server_url, upload_url, &config.api_key, &file_path)
                .await?;

            let complete_msg = WorkerMessage::TaskComplete {
                task_id: task_id.to_string(),
                result_filename: filename.clone(),
                metadata: metadata.clone(),
            };
            if let Ok(json) = serde_json::to_string(&complete_msg) {
                let _ = write.lock().await.send(Message::Text(json)).await;
            }

            info!("✓ Task {} complete: {}", &task_id[..8], filename);
        }
    }

    *active_inference.lock().await = None;

    info!("✓ Job {} complete ({} scenes)", &job_id[..8], tasks.len());

    Ok(())
}
