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

/// Batches tasks by job_id for efficient model reuse
struct TaskBatcher {
    batches: HashMap<String, Vec<TaskAssignment>>,
}

#[derive(Clone)]
struct TaskAssignment {
    task_id: String,
    job_id: String,
    task_type: String,
    scene: serde_json::Value,
    project: serde_json::Value,
    model_id: String,
    model_config: crate::protocol::ModelConfig,
    pipeline_config: serde_json::Value,
    upload_url: String,
    last_frame_url: Option<String>,
    input_video_url: Option<String>,
}

impl TaskBatcher {
    fn new() -> Self {
        Self {
            batches: HashMap::new(),
        }
    }

    fn add_task(&mut self, assignment: TaskAssignment) {
        self.batches
            .entry(assignment.job_id.clone())
            .or_insert_with(Vec::new)
            .push(assignment);
    }

    fn take_batch(&mut self, job_id: &str) -> Option<Vec<TaskAssignment>> {
        self.batches.remove(job_id)
    }

    fn has_job(&self, job_id: &str) -> bool {
        self.batches.contains_key(job_id)
    }
}

/// Build hardware stats snapshot from current system state.
fn build_hardware_stats(config: &WorkerConfig) -> HardwareStats {
    let gpus = hardware::detect_gpus();

    // Apply resource limits from constraints (if configured)
    // If no limits set, use 90% of actual hardware (10% buffer for system)
    let cpu_cores = config
        .constraints
        .cpu_limit
        .unwrap_or_else(|| (hardware::cpu_cores() as f64 * 0.9).max(1.0) as usize);
    let ram_total_gb = config
        .constraints
        .ram_limit_gb
        .unwrap_or_else(|| hardware::total_ram_gb() * 0.9);
    let ram_free_gb = hardware::free_ram_gb();
    let ram_used_gb = ram_total_gb - ram_free_gb;

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

    // Task batcher for collecting scenes from the same job
    let batcher = Arc::new(Mutex::new(TaskBatcher::new()));
    let batch_timers: Arc<Mutex<HashMap<String, tokio::task::JoinHandle<()>>>> = Arc::new(Mutex::new(HashMap::new()));

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

                    ServerMessage::TaskAssign {
                        task_id,
                        job_id,
                        task_type,
                        scene,
                        project,
                        model_id,
                        model_config,
                        pipeline_config,
                        upload_url,
                        last_frame_url,
                        input_video_url,
                    } => {
                        info!(
                            "Received task {} (job {}, scene {}, type {})",
                            &task_id[..8],
                            &job_id[..8],
                            scene.get("id").and_then(|v| v.as_i64()).unwrap_or(-1),
                            task_type
                        );

                        // Add task to batcher
                        let assignment = TaskAssignment {
                            task_id: task_id.clone(),
                            job_id: job_id.clone(),
                            task_type: task_type.clone(),
                            scene,
                            project,
                            model_id: model_id.clone(),
                            model_config: *model_config,
                            pipeline_config: *pipeline_config,
                            upload_url: upload_url.clone(),
                            last_frame_url,
                            input_video_url,
                        };

                        let is_first_task = {
                            let mut batcher = batcher.lock().await;
                            let is_first = !batcher.has_job(&job_id);
                            batcher.add_task(assignment);
                            is_first
                        };

                        // If this is the first task for this job, start a batch timer
                        if is_first_task {
                            info!("Starting batch collection for job {} (300ms window)", &job_id[..8]);

                            let config = config.clone();
                            let write = write.clone();
                            let active = active_inference.clone();
                            let batcher = batcher.clone();
                            let batch_timers_for_task = batch_timers.clone();
                            let job_id_clone = job_id.clone();

                            let timer_handle = tokio::spawn(async move {
                                // Wait for more tasks to arrive
                                time::sleep(Duration::from_millis(300)).await;

                                // Extract all tasks for this job
                                let tasks = {
                                    let mut b = batcher.lock().await;
                                    b.take_batch(&job_id_clone)
                                };

                                if let Some(tasks) = tasks {
                                    info!("Processing batch for job {} with {} scene(s)", &job_id_clone[..8], tasks.len());

                                    let result = process_task_batch(
                                        &config,
                                        &write,
                                        &active,
                                        &job_id_clone,
                                        tasks,
                                    )
                                    .await;

                                    if let Err(e) = result {
                                        error!("Job {} failed: {}", &job_id_clone[..8], e);
                                        // Report first task as failed (for now)
                                    }
                                }

                                // Remove timer handle
                                batch_timers_for_task.lock().await.remove(&job_id_clone);
                            });

                            batch_timers.lock().await.insert(job_id.clone(), timer_handle);
                        } else {
                            info!("Added to existing batch for job {}", &job_id[..8]);
                        }

                        // Continue requesting tasks
                        let msg = serde_json::to_string(&WorkerMessage::RequestTask {
                            worker_id: config.worker_id.clone(),
                        })?;
                        write.lock().await.send(Message::Text(msg)).await?;
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
/// Process a batch of tasks (scenes) from the same job, keeping the model loaded
async fn process_task_batch<S>(
    config: &WorkerConfig,
    write: &Arc<Mutex<S>>,
    active_inference: &Arc<Mutex<Option<runner::RunningInference>>>,
    job_id: &str,
    tasks: Vec<TaskAssignment>,
) -> anyhow::Result<()>
where
    S: SinkExt<Message> + Unpin + Send + 'static,
    <S as futures_util::Sink<Message>>::Error: std::error::Error + Send + Sync + 'static,
{
    if tasks.is_empty() {
        return Ok(());
    }

    // Use first task's config (all tasks in a job should have same model/pipeline config)
    let first_task = &tasks[0];
    let task_type = &first_task.task_type;
    let model_id = &first_task.model_id;
    let model_config = &first_task.model_config;
    let pipeline_config = &first_task.pipeline_config;
    let project = &first_task.project;

    info!(
        "Processing {} scene(s) for job {} (model: {}, type: {})",
        tasks.len(),
        &job_id[..8],
        model_id,
        task_type
    );

    // 1. Ensure model is cached
    if !models::is_model_cached(&config.models_dir, &model_config.local_dir) {
        info!("Model {} not cached, downloading...", model_id);
        let task_id_owned = first_task.task_id.clone();
        let model_id_owned = model_id.clone();
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
    let last_frame_path = if let Some(url) = &first_task.last_frame_url {
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

    // 3. Create output directory for the job
    let output_dir = config
        .models_dir
        .parent()
        .unwrap_or(&config.models_dir)
        .join("output")
        .join(job_id);
    tokio::fs::create_dir_all(&output_dir).await?;

    // 4. Build scene list for Python
    let mut scene_tasks = Vec::new();
    for task in &tasks {
        let mut scene_with_task = task.scene.clone();
        if let Some(obj) = scene_with_task.as_object_mut() {
            obj.insert("task_id".to_string(), serde_json::Value::String(task.task_id.clone()));
        }
        scene_tasks.push(SceneTask {
            task_id: task.task_id.clone(),
            scene: scene_with_task,
        });
    }

    // 5. Spawn Python with ALL scenes
    let model_path = config
        .models_dir
        .join(&model_config.local_dir)
        .to_string_lossy()
        .to_string();

    let job = InferenceJob {
        task_id: first_task.task_id.clone(),  // Use first task's ID for backward compat
        task_type: task_type.clone(),
        scene: None,  // Use scenes instead
        scenes: Some(scene_tasks),
        project: project.clone(),
        model_path,
        model_config: serde_json::to_value(&model_config)?,
        pipeline_config: pipeline_config.clone(),
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
                // Report progress for first task (representative of whole batch)
                let scaled_pct = 10.0 + pct * 0.85;
                let msg = WorkerMessage::TaskProgress {
                    task_id: first_task.task_id.clone(),
                    progress: scaled_pct,
                    message,
                    phase: "generating".to_string(),
                };
                if let Ok(json) = serde_json::to_string(&msg) {
                    let _ = write.lock().await.send(Message::Text(json)).await;
                }
            }
            PythonOutput::Complete { files, metadata } => {
                // Parse metadata to map files to task_ids
                if let Some(scenes_meta) = metadata.get("scenes").and_then(|v| v.as_array()) {
                    for (_idx, scene_meta) in scenes_meta.iter().enumerate() {
                        if let Some(task_id) = scene_meta.get("task_id").and_then(|v| v.as_str()) {
                            if let Some(filename) = scene_meta.get("filename").and_then(|v| v.as_str()) {
                                completed_scenes.insert(
                                    task_id.to_string(),
                                    (filename.to_string(), scene_meta.clone()),
                                );
                            }
                        }
                    }
                } else {
                    // Fallback: single scene (backward compat)
                    if !files.is_empty() {
                        completed_scenes.insert(
                            first_task.task_id.clone(),
                            (files[0].clone(), metadata.clone()),
                        );
                    }
                }
            }
            PythonOutput::Error { message } => {
                anyhow::bail!("Python inference error: {}", message);
            }
        }
    }

    // 7. Wait for process and upload results for each task
    // (Upload individually so each scene is tracked separately)
    for task in &tasks {
        if let Some((filename, metadata)) = completed_scenes.get(&task.task_id) {
            let file_path = output_dir.join(filename);

            // Upload
            let upload_pct_msg = WorkerMessage::TaskProgress {
                task_id: task.task_id.clone(),
                progress: 95.0,
                message: format!("Uploading {}", filename),
                phase: "uploading".to_string(),
            };
            if let Ok(json) = serde_json::to_string(&upload_pct_msg) {
                let _ = write.lock().await.send(Message::Text(json)).await;
            }

            upload::upload_file(
                &config.server_url,
                &task.upload_url,
                &config.api_key,
                &file_path,
            )
            .await?;

            // Report complete
            let complete_msg = WorkerMessage::TaskComplete {
                task_id: task.task_id.clone(),
                result_filename: filename.clone(),
                metadata: metadata.clone(),
            };
            if let Ok(json) = serde_json::to_string(&complete_msg) {
                let _ = write.lock().await.send(Message::Text(json)).await;
            }

            info!("✓ Task {} complete: {}", &task.task_id[..8], filename);
        } else {
            warn!("No output found for task {}", &task.task_id[..8]);
        }
    }

    *active_inference.lock().await = None;

    info!("✓ Batch complete for job {} ({} scenes)", &job_id[..8], tasks.len());

    Ok(())
}

async fn process_task<S>(
    config: &WorkerConfig,
    write: &Arc<Mutex<S>>,
    active_inference: &Arc<Mutex<Option<runner::RunningInference>>>,
    task_id: &str,
    _job_id: &str,
    task_type: &str,
    scene: serde_json::Value,
    project: serde_json::Value,
    model_id: &str,
    model_config: crate::protocol::ModelConfig,
    pipeline_config: serde_json::Value,
    upload_url: &str,
    last_frame_url: Option<&str>,
    input_video_url: Option<&str>,
) -> anyhow::Result<()>
where
    S: SinkExt<Message> + Unpin + Send + 'static,
    <S as futures_util::Sink<Message>>::Error: std::error::Error + Send + Sync + 'static,
{
    // 1. Ensure model is cached
    if !models::is_model_cached(&config.models_dir, &model_config.local_dir) {
        info!("Model {} not cached, downloading...", model_id);
        let task_id_owned = task_id.to_string();
        let model_id_owned = model_id.to_string();
        let write_clone = write.clone();

        // Report model download progress
        let progress_cb = move |downloaded_gb: f64, total_gb: f64| {
            let msg = WorkerMessage::ModelProgress {
                task_id: task_id_owned.clone(),
                model_id: model_id_owned.clone(),
                downloaded_gb,
                total_gb,
            };
            // Fire-and-forget (blocking context)
            if let Ok(json) = serde_json::to_string(&msg) {
                let write = write_clone.clone();
                // Use Handle::current() to spawn from blocking context
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

    // 2. Download lastframe if needed for continuity
    let last_frame_path = if let Some(url) = last_frame_url {
        let dest = config
            .models_dir
            .parent()
            .unwrap_or(&config.models_dir)
            .join("tmp")
            .join(format!("{}_lastframe.png", task_id));
        upload::download_file(&config.server_url, url, &config.api_key, &dest).await?;
        Some(dest)
    } else {
        None
    };

    // 2b. Download input video if needed (for upscale tasks)
    let input_video_path = if let Some(url) = input_video_url {
        let dest = config
            .models_dir
            .parent()
            .unwrap_or(&config.models_dir)
            .join("tmp")
            .join(format!("{}_input.mp4", task_id));
        upload::download_file(&config.server_url, url, &config.api_key, &dest).await?;
        info!("Downloaded input video to: {}", dest.display());
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
        .join(task_id);
    tokio::fs::create_dir_all(&output_dir).await?;

    // 4. Spawn Python inference
    let model_path = config
        .models_dir
        .join(&model_config.local_dir)
        .to_string_lossy()
        .to_string();

    let job = InferenceJob {
        task_id: task_id.to_string(),
        task_type: task_type.to_string(),
        scene: Some(scene),  // Single scene for backward compat
        scenes: None,
        project,
        model_path,
        model_config: serde_json::to_value(&model_config)?, // Send FULL config, not just .extra
        pipeline_config,
        output_dir: output_dir.to_string_lossy().to_string(),
        last_frame_path: last_frame_path.map(|p| p.to_string_lossy().to_string()),
        input_video_path: input_video_path.map(|p| p.to_string_lossy().to_string()),
    };

    let (mut output_rx, mut handle) =
        runner::spawn_inference(&config.python_path, &config.python_scripts_dir, job).await?;

    *active_inference.lock().await = None; // We'll set it after extracting outputs

    // 5. Process Python output
    let mut result_files = Vec::new();
    let mut result_metadata = serde_json::Value::Null;

    while let Some(output) = output_rx.recv().await {
        match output {
            PythonOutput::Progress { pct, message } => {
                // Scale: model download = 0-10%, inference = 10-95%, upload = 95-100%
                let scaled_pct = 10.0 + pct * 0.85;
                let msg = WorkerMessage::TaskProgress {
                    task_id: task_id.to_string(),
                    progress: scaled_pct,
                    message,
                    phase: "generating".to_string(),
                };
                if let Ok(json) = serde_json::to_string(&msg) {
                    let _ = write.lock().await.send(Message::Text(json)).await;
                }
            }
            PythonOutput::Complete { files, metadata } => {
                result_files = files;
                result_metadata = metadata;
            }
            PythonOutput::Error { message } => {
                anyhow::bail!("Python inference error: {}", message);
            }
        }
    }

    // 6. Wait for process to finish
    let exit_code = runner::wait_for_completion(&mut handle).await?;
    if exit_code != 0 {
        anyhow::bail!("Python exited with code {}", exit_code);
    }

    // 7. Upload result files
    for file_name in &result_files {
        let file_path = output_dir.join(file_name);
        if file_path.exists() {
            // Report uploading phase
            let msg = WorkerMessage::TaskProgress {
                task_id: task_id.to_string(),
                progress: 96.0,
                message: format!("Uploading {}", file_name),
                phase: "uploading".to_string(),
            };
            if let Ok(json) = serde_json::to_string(&msg) {
                let _ = write.lock().await.send(Message::Text(json)).await;
            }

            upload::upload_file(&config.server_url, upload_url, &config.api_key, &file_path)
                .await?;

            // Upload lastframe if it exists
            let scene_id = result_metadata
                .get("scene_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let lastframe = output_dir.join(format!("scene_{:03}_lastframe.png", scene_id));
            if lastframe.exists() {
                upload::upload_lastframe(&config.server_url, task_id, &config.api_key, &lastframe)
                    .await?;
            }
        }
    }

    // 8. Report completion
    let complete_msg = WorkerMessage::TaskComplete {
        task_id: task_id.to_string(),
        result_filename: result_files.first().cloned().unwrap_or_default(),
        metadata: result_metadata,
    };
    if let Ok(json) = serde_json::to_string(&complete_msg) {
        let _ = write.lock().await.send(Message::Text(json)).await;
    }

    // 9. Cleanup output dir
    let _ = tokio::fs::remove_dir_all(&output_dir).await;

    Ok(())
}
