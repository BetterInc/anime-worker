//! Python subprocess management for GPU inference.
//!
//! Spawns `python inference_runner.py`, sends job config on stdin,
//! reads JSON progress/completion lines from stdout.

use std::path::Path;
use std::process::Stdio;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tracing::{error, info, warn};

/// Messages emitted by the Python subprocess on stdout.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PythonOutput {
    Progress {
        pct: f64,
        message: String,
    },
    Complete {
        files: Vec<String>,
        metadata: serde_json::Value,
    },
    Error {
        message: String,
    },
}

/// The job configuration sent to Python on stdin.
#[derive(Debug, Serialize)]
pub struct InferenceJob {
    pub task_id: String,
    pub task_type: String,
    pub scene: serde_json::Value,
    pub project: serde_json::Value,
    pub model_path: String,
    pub model_config: serde_json::Value,
    pub pipeline_config: serde_json::Value,
    pub output_dir: String,
    pub last_frame_path: Option<String>,
}

/// Handle to a running Python inference process.
pub struct RunningInference {
    child: Child,
    cancel_tx: mpsc::Sender<()>,
}

impl RunningInference {
    /// Send a cancel signal to the Python process.
    pub async fn cancel(&self) {
        let _ = self.cancel_tx.send(()).await;
    }
}

/// Spawn the Python inference runner and stream its output.
///
/// Returns a receiver of PythonOutput messages and a handle to the running process.
pub async fn spawn_inference(
    python_path: &str,
    scripts_dir: &Path,
    job: InferenceJob,
) -> anyhow::Result<(mpsc::Receiver<PythonOutput>, RunningInference)> {
    let runner_script = scripts_dir.join("inference_runner.py");
    if !runner_script.exists() {
        anyhow::bail!(
            "inference_runner.py not found at {}",
            runner_script.display()
        );
    }

    info!("Spawning Python inference: {} {}", python_path, runner_script.display());

    let mut child = Command::new(python_path)
        .arg(&runner_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    // Send job config on stdin
    let job_json = serde_json::to_string(&job)?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(job_json.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        drop(stdin); // Close stdin so Python can start
    }

    let (output_tx, output_rx) = mpsc::channel::<PythonOutput>(64);
    let (cancel_tx, mut cancel_rx) = mpsc::channel::<()>(1);

    // Read stdout for JSON progress lines
    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");

    let tx = output_tx.clone();
    tokio::spawn(async move {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<PythonOutput>(trimmed) {
                Ok(msg) => {
                    if tx.send(msg).await.is_err() {
                        break;
                    }
                }
                Err(_) => {
                    // Non-JSON line, treat as log
                    info!("[python] {}", trimmed);
                }
            }
        }
    });

    // Forward stderr as log lines
    tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            warn!("[python:stderr] {}", line);
        }
    });

    // Cancel handler: kill the child process
    let child_id = child.id();
    tokio::spawn(async move {
        if cancel_rx.recv().await.is_some() {
            info!("Cancelling inference subprocess (pid: {:?})", child_id);
            #[cfg(unix)]
            {
                if let Some(pid) = child_id {
                    unsafe {
                        libc::kill(pid as i32, libc::SIGTERM);
                    }
                }
            }
            #[cfg(windows)]
            {
                // On Windows, we just drop the child which should terminate it
                // The child handle is moved into the wait task below
            }
        }
    });

    let handle = RunningInference {
        child,
        cancel_tx,
    };

    Ok((output_rx, handle))
}

/// Wait for the inference process to complete and return the exit code.
pub async fn wait_for_completion(handle: &mut RunningInference) -> anyhow::Result<i32> {
    let status = handle.child.wait().await?;
    Ok(status.code().unwrap_or(-1))
}
