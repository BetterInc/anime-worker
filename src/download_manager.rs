//! Manages active downloads with cancellation support

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::Mutex;

/// Manages active downloads with cancellation support
pub struct DownloadManager {
    active: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

impl DownloadManager {
    pub fn new() -> Self {
        Self {
            active: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a new download and get cancellation flag
    pub async fn register(&self, model_id: String) -> Arc<AtomicBool> {
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.active
            .lock()
            .await
            .insert(model_id, cancel_flag.clone());
        cancel_flag
    }

    /// Cancel a download by model_id
    pub async fn cancel(&self, model_id: &str) -> bool {
        if let Some(flag) = self.active.lock().await.get(model_id) {
            flag.store(true, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    /// Remove completed download from tracking
    #[allow(dead_code)]
    pub async fn remove(&self, model_id: &str) {
        self.active.lock().await.remove(model_id);
    }
}

impl Default for DownloadManager {
    fn default() -> Self {
        Self::new()
    }
}
