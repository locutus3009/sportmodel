//! File watching for automatic data reload.
//!
//! Watches the Excel file for modifications and triggers a reload callback.
//! Implements debouncing to handle rapid successive events from editors and Syncthing.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use thiserror::Error;
use tokio::sync::{Mutex, mpsc};

/// Configuration for file watching.
#[derive(Debug, Clone)]
pub struct WatcherConfig {
    /// Minimum time between callbacks (default: 2 seconds).
    pub debounce_duration: Duration,
    /// Number of retry attempts for reload (default: 3).
    pub retry_attempts: u32,
    /// Delay between retry attempts (default: 500ms).
    pub retry_delay: Duration,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            debounce_duration: Duration::from_secs(2),
            retry_attempts: 3,
            retry_delay: Duration::from_millis(500),
        }
    }
}

/// Errors that can occur during file watching.
#[derive(Debug, Error)]
pub enum WatcherError {
    #[error("Failed to create watcher: {0}")]
    NotifyError(#[from] notify::Error),

    #[error("Watch path does not exist: {0}")]
    PathNotFound(PathBuf),

    #[error("Channel closed unexpectedly")]
    ChannelClosed,
}

/// Debouncer state for collapsing rapid events.
struct Debouncer {
    last_triggered: Instant,
    duration: Duration,
}

impl Debouncer {
    fn new(duration: Duration) -> Self {
        Self {
            // Start in the past so first event triggers immediately
            last_triggered: Instant::now() - duration - Duration::from_secs(1),
            duration,
        }
    }

    /// Returns true if enough time has passed since the last trigger.
    fn should_trigger(&mut self) -> bool {
        let now = Instant::now();
        if now.duration_since(self.last_triggered) >= self.duration {
            self.last_triggered = now;
            true
        } else {
            false
        }
    }

    /// Resets the debounce timer without triggering.
    fn reset(&mut self) {
        self.last_triggered = Instant::now();
    }
}

/// Starts watching a file for modifications.
///
/// Calls `on_change` when the file is modified, with debouncing to prevent
/// rapid successive calls. The callback runs in a tokio task.
///
/// This function blocks until an error occurs or the watcher is dropped.
pub async fn watch_file<F>(
    path: impl AsRef<Path>,
    config: WatcherConfig,
    on_change: F,
) -> Result<(), WatcherError>
where
    F: Fn() + Send + Sync + 'static,
{
    let path = path.as_ref();

    // Verify path exists
    if !path.exists() {
        return Err(WatcherError::PathNotFound(path.to_path_buf()));
    }

    // Get the canonical path and parent directory
    let canonical_path = path
        .canonicalize()
        .map_err(|_| WatcherError::PathNotFound(path.to_path_buf()))?;
    let watch_dir = canonical_path.parent().unwrap_or(&canonical_path);
    let file_name = canonical_path.file_name().map(|s| s.to_owned());

    log::info!("Watching file: {}", canonical_path.display());
    log::debug!("Watch directory: {}", watch_dir.display());

    // Create channel for events
    let (tx, mut rx) = mpsc::channel::<Event>(100);

    // Create the watcher
    let tx_clone = tx.clone();
    let mut watcher = RecommendedWatcher::new(
        move |result: Result<Event, notify::Error>| {
            if let Ok(event) = result {
                // Non-blocking send - if channel is full, drop the event
                let _ = tx_clone.try_send(event);
            }
        },
        notify::Config::default(),
    )?;

    // Watch the parent directory (more reliable for file replacements)
    watcher.watch(watch_dir, RecursiveMode::NonRecursive)?;

    let on_change = Arc::new(on_change);
    let debouncer = Arc::new(Mutex::new(Debouncer::new(config.debounce_duration)));

    // Track pending events for delayed debounce
    let pending_trigger = Arc::new(Mutex::new(false));

    // Spawn debounce timer task
    let debouncer_clone = debouncer.clone();
    let on_change_clone = on_change.clone();
    let pending_clone = pending_trigger.clone();

    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;

            let mut pending = pending_clone.lock().await;
            if *pending {
                let mut db = debouncer_clone.lock().await;
                if db.should_trigger() {
                    *pending = false;
                    drop(db);
                    drop(pending);
                    log::info!("Debounce timer triggered reload");
                    on_change_clone();
                }
            }
        }
    });

    // Process events
    while let Some(event) = rx.recv().await {
        // Check if this event relates to our file
        let is_our_file = event.paths.iter().any(|p| {
            if let Some(ref fname) = file_name {
                p.file_name() == Some(fname.as_os_str())
            } else {
                p == &canonical_path
            }
        });

        if !is_our_file {
            continue;
        }

        // Check event type - we care about creates, writes, and renames
        let is_relevant = matches!(
            event.kind,
            EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
        );

        if !is_relevant {
            continue;
        }

        log::debug!("File event: {:?}", event.kind);

        // Try immediate trigger or mark pending
        let mut db = debouncer.lock().await;
        if db.should_trigger() {
            drop(db);
            log::info!("File changed, triggering reload");
            on_change();
        } else {
            // Mark pending for the debounce timer
            db.reset();
            drop(db);
            let mut pending = pending_trigger.lock().await;
            *pending = true;
        }
    }

    Err(WatcherError::ChannelClosed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debouncer_first_event_triggers() {
        let mut debouncer = Debouncer::new(Duration::from_secs(2));
        assert!(debouncer.should_trigger());
    }

    #[test]
    fn test_debouncer_rapid_events_blocked() {
        let mut debouncer = Debouncer::new(Duration::from_secs(2));
        assert!(debouncer.should_trigger());
        assert!(!debouncer.should_trigger());
        assert!(!debouncer.should_trigger());
    }

    #[test]
    fn test_watcher_config_default() {
        let config = WatcherConfig::default();
        assert_eq!(config.debounce_duration, Duration::from_secs(2));
        assert_eq!(config.retry_attempts, 3);
        assert_eq!(config.retry_delay, Duration::from_millis(500));
    }
}
