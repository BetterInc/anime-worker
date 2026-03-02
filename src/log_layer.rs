//! Custom tracing layer that forwards logs to the API via log_tx channel.

use tokio::sync::mpsc;
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer};

use crate::protocol::LogEntry;

/// Tracing layer that captures logs and sends them to a channel for API forwarding
pub struct ApiLogLayer {
    tx: mpsc::Sender<LogEntry>,
}

impl ApiLogLayer {
    pub fn new(tx: mpsc::Sender<LogEntry>) -> Self {
        Self { tx }
    }
}

impl<S: Subscriber> Layer<S> for ApiLogLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();

        // Extract log message
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);

        if visitor.message.is_empty() {
            return;
        }

        let log = LogEntry {
            job_id: None,
            task_id: None,
            level: metadata.level().to_string(),
            message: visitor.message,
            source: "worker".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            metadata: None,
        };

        // Send non-blocking (drop if channel full)
        let _ = self.tx.try_send(log);
    }
}

#[derive(Default)]
struct MessageVisitor {
    message: String,
}

impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{:?}", value);
            // Remove quotes from debug formatting
            if self.message.starts_with('"') && self.message.ends_with('"') {
                self.message = self.message[1..self.message.len() - 1].to_string();
            }
        }
    }
}
