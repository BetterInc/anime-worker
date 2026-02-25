//! WebSocket client with auto-reconnect and message dispatch.

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
use crate::protocol::{ServerMessage, WorkerMessage};
use crate::runner::{self, InferenceJob, PythonOutput};
use crate::upload;

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
    let gpus = hardware::detect_gpus();
    let cached = models::list_cached_models(&config.models_dir);

    // Apply resource limits from constraints (if configured)
    let cpu_cores = config.constraints.cpu_limit.unwrap_or_else(|| hardware::cpu_cores());
    let ram_total_gb = config.constraints.ram_limit_gb.unwrap_or_else(|| hardware::total_ram_gb());
    let disk_space_gb = config.constraints.disk_limit_gb.unwrap_or_else(|| hardware::available_disk_space_gb(&config.models_dir));

    let hello = WorkerMessage::Hello {
        worker_id: config.worker_id.clone(),
        api_key: config.api_key.clone(),
        name: config.worker_name.clone(),
        gpus: gpus.clone(),
        ram_total_gb,
        cpu_cores,
        disk_space_gb,
        platform: hardware::platform().to_string(),
        models_cached: cached.clone(),
        constraints: Some(config.constraints.clone()),
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
            let gpus = hardware::detect_gpus();
            let cached = models::list_cached_models(&hb_config.models_dir);
            // Apply disk limit from constraints (if configured)
            let disk_space_gb = hb_config.constraints.disk_limit_gb
                .unwrap_or_else(|| hardware::available_disk_space_gb(&hb_config.models_dir));

            let msg = WorkerMessage::Heartbeat {
                worker_id: hb_config.worker_id.clone(),
                gpus,
                ram_free_gb: hardware::free_ram_gb(),
                disk_space_gb,
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
                            "Received task {} (scene {}, type {})",
                            &task_id[..8],
                            scene.get("id").and_then(|v| v.as_i64()).unwrap_or(-1),
                            task_type
                        );

                        let config = config.clone();
                        let write = write.clone();
                        let active = active_inference.clone();
                        let task_id_clone = task_id.clone();

                        tokio::spawn(async move {
                            let result = process_task(
                                &config,
                                &write,
                                &active,
                                &task_id,
                                &job_id,
                                &task_type,
                                scene,
                                project,
                                &model_id,
                                *model_config,
                                *pipeline_config,
                                &upload_url,
                                last_frame_url.as_deref(),
                                input_video_url.as_deref(),
                            )
                            .await;

                            if let Err(e) = result {
                                error!("Task {} failed: {}", &task_id_clone[..8], e);
                                let fail_msg = serde_json::to_string(&WorkerMessage::TaskFailed {
                                    task_id: task_id_clone.clone(),
                                    error_message: e.to_string(),
                                    phase: "processing".to_string(),
                                })
                                .unwrap_or_default();
                                let _ = write.lock().await.send(Message::Text(fail_msg)).await;
                            }

                            // Clear active inference
                            *active.lock().await = None;

                            // Request next task
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
        scene,
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
