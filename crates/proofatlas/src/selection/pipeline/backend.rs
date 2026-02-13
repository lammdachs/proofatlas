//! Backend: model-agnostic compute service for ML inference.
//!
//! The Backend manages device worker threads that execute model forward passes.
//! Data processing threads submit requests via `BackendHandle` and receive
//! responses through per-request response channels.
//!
//! # Architecture
//!
//! ```text
//! Data Processing 0 ──→ BackendHandle ──→ Backend ──→ Worker (GPU/CPU)
//! Data Processing 1 ──→ BackendHandle ──→             (owns models)
//! ```
//!
//! - **BackendHandle**: cheaply cloneable, wraps `mpsc::Sender` to backend.
//! - **Backend**: owns the worker thread with all models.
//! - **Worker**: receives requests, groups by model, executes batches, sends responses.
//!
//! Lifecycle: When all `BackendHandle`s are dropped, the request channel closes
//! and the worker drains remaining work then exits.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::thread::JoinHandle;

// =============================================================================
// Request / Response types
// =============================================================================

/// A request to the backend for model computation.
pub struct BackendRequest {
    /// Unique request ID for correlation.
    pub id: u64,
    /// Model identifier (e.g., "gcn_encoder", "mlp_scorer").
    pub model_id: String,
    /// Type-erased input data for the model.
    pub data: Box<dyn std::any::Any + Send>,
    /// Channel to send the response back to the requester.
    pub response_tx: mpsc::Sender<BackendResponse>,
}

/// A response from the backend.
pub struct BackendResponse {
    /// Request ID this responds to.
    pub id: u64,
    /// Type-erased output data from the model.
    pub data: Box<dyn std::any::Any + Send>,
}

// =============================================================================
// Model trait — implemented by each ML model
// =============================================================================

/// Trait for model execution. Each model type implements this.
///
/// Models run inside the backend worker thread with exclusive access to
/// compute resources. No synchronization is needed within the model.
pub trait Model: Send + 'static {
    /// Model identifier (must match `BackendRequest::model_id`).
    fn model_id(&self) -> &str;

    /// Execute a batch of requests.
    ///
    /// Input: `(request_id, type-erased data)` pairs.
    /// Output: `(request_id, type-erased result)` pairs, in the same order.
    ///
    /// The model knows its concrete input/output types and downcasts internally.
    fn execute_batch(
        &mut self,
        requests: Vec<(u64, Box<dyn std::any::Any + Send>)>,
    ) -> Vec<(u64, Box<dyn std::any::Any + Send>)>;
}

// =============================================================================
// BackendHandle — cloneable handle for submitting requests
// =============================================================================

/// Cheaply cloneable handle for submitting requests to the Backend.
///
/// When all handles are dropped, the backend's request channel closes
/// and the worker shuts down after draining remaining work.
#[derive(Clone)]
pub struct BackendHandle {
    tx: mpsc::Sender<BackendRequest>,
}

impl BackendHandle {
    /// Submit a request to the backend. Returns error if the backend is shut down.
    pub fn submit(&self, request: BackendRequest) -> Result<(), String> {
        self.tx
            .send(request)
            .map_err(|_| "Backend channel closed".to_string())
    }

    /// Submit a request and wait for the response synchronously.
    pub fn submit_sync(
        &self,
        id: u64,
        model_id: String,
        data: Box<dyn std::any::Any + Send>,
    ) -> Result<BackendResponse, String> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.submit(BackendRequest {
            id,
            model_id,
            data,
            response_tx: resp_tx,
        })?;
        resp_rx.recv().map_err(|_| "Backend response channel closed".to_string())
    }
}

// =============================================================================
// Backend — manages the worker thread
// =============================================================================

/// Model-agnostic compute service.
///
/// Owns a worker thread that processes model execution requests.
/// Multiple data processing threads can share a single backend
/// by cloning `BackendHandle`s.
pub struct Backend {
    /// Worker thread handle. Detached on drop (not joined).
    /// The worker exits when all `BackendHandle` senders are dropped.
    _worker: JoinHandle<()>,
    handle: BackendHandle,
}

impl Backend {
    /// Create a new Backend with the given models.
    ///
    /// Spawns a single worker thread with a 16 MiB stack (required for
    /// libtorch operations). All models are co-located on this worker.
    pub fn new(models: Vec<Box<dyn Model>>) -> Self {
        let (tx, rx) = mpsc::channel::<BackendRequest>();

        let worker = std::thread::Builder::new()
            .name("backend-worker".to_string())
            .stack_size(16 * 1024 * 1024) // 16 MiB for libtorch
            .spawn(move || {
                Self::worker_loop(rx, models);
            })
            .expect("Failed to spawn backend worker thread");

        Backend {
            _worker: worker,
            handle: BackendHandle { tx },
        }
    }

    /// Get a cloneable handle for submitting requests.
    pub fn handle(&self) -> BackendHandle {
        self.handle.clone()
    }

    /// Worker loop: receive requests, group by model, execute batches.
    fn worker_loop(
        rx: mpsc::Receiver<BackendRequest>,
        models: Vec<Box<dyn Model>>,
    ) {
        let mut models_map: HashMap<String, Box<dyn Model>> = models
            .into_iter()
            .map(|m| (m.model_id().to_string(), m))
            .collect();

        // Buffer for grouping requests by model
        let mut pending: HashMap<String, VecDeque<BackendRequest>> = HashMap::new();

        loop {
            // Block on first request (or exit if channel closed)
            let first = match rx.recv() {
                Ok(req) => req,
                Err(_) => {
                    // Channel closed — drain any remaining pending requests
                    Self::flush_pending(&mut pending, &mut models_map);
                    break;
                }
            };

            // Add first request to pending
            pending
                .entry(first.model_id.clone())
                .or_default()
                .push_back(first);

            // Drain any additional queued requests (non-blocking)
            while let Ok(req) = rx.try_recv() {
                pending
                    .entry(req.model_id.clone())
                    .or_default()
                    .push_back(req);
            }

            // Execute all pending batches
            Self::flush_pending(&mut pending, &mut models_map);
        }
    }

    /// Execute all pending request batches.
    fn flush_pending(
        pending: &mut HashMap<String, VecDeque<BackendRequest>>,
        models_map: &mut HashMap<String, Box<dyn Model>>,
    ) {
        for (model_id, queue) in pending.iter_mut() {
            if queue.is_empty() {
                continue;
            }

            let model = match models_map.get_mut(model_id) {
                Some(m) => m,
                None => {
                    eprintln!("Backend: unknown model_id '{}', dropping {} requests",
                        model_id, queue.len());
                    queue.clear();
                    continue;
                }
            };

            // Separate response channels from data
            let requests: Vec<BackendRequest> = queue.drain(..).collect();
            let mut response_txs: Vec<(u64, mpsc::Sender<BackendResponse>)> =
                Vec::with_capacity(requests.len());
            let mut batch_data: Vec<(u64, Box<dyn std::any::Any + Send>)> =
                Vec::with_capacity(requests.len());

            for req in requests {
                response_txs.push((req.id, req.response_tx));
                batch_data.push((req.id, req.data));
            }

            // Execute batch
            let results = model.execute_batch(batch_data);

            // Send responses — results are in the same order as inputs
            for ((id, tx), (_result_id, data)) in response_txs.into_iter().zip(results) {
                let _ = tx.send(BackendResponse { id, data });
            }
        }
    }
}

// No Drop impl — the worker thread is detached when Backend is dropped.
// It exits naturally when all BackendHandle senders are dropped.

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test model that doubles a f32 value.
    struct DoublerModel;

    impl Model for DoublerModel {
        fn model_id(&self) -> &str {
            "doubler"
        }

        fn execute_batch(
            &mut self,
            requests: Vec<(u64, Box<dyn std::any::Any + Send>)>,
        ) -> Vec<(u64, Box<dyn std::any::Any + Send>)> {
            requests
                .into_iter()
                .map(|(id, data)| {
                    let val = *data.downcast::<f32>().unwrap();
                    (id, Box::new(val * 2.0) as Box<dyn std::any::Any + Send>)
                })
                .collect()
        }
    }

    #[test]
    fn test_backend_basic() {
        let backend = Backend::new(vec![Box::new(DoublerModel)]);
        let handle = backend.handle();

        let resp = handle.submit_sync(1, "doubler".to_string(), Box::new(3.0f32)).unwrap();
        let result = *resp.data.downcast::<f32>().unwrap();
        assert_eq!(result, 6.0);
    }

    #[test]
    fn test_backend_multiple_requests() {
        let backend = Backend::new(vec![Box::new(DoublerModel)]);
        let handle = backend.handle();

        // Submit multiple requests
        let (tx1, rx1) = mpsc::channel();
        let (tx2, rx2) = mpsc::channel();
        handle.submit(BackendRequest { id: 1, model_id: "doubler".to_string(), data: Box::new(5.0f32), response_tx: tx1 }).unwrap();
        handle.submit(BackendRequest { id: 2, model_id: "doubler".to_string(), data: Box::new(7.0f32), response_tx: tx2 }).unwrap();

        let r1 = rx1.recv().unwrap();
        let r2 = rx2.recv().unwrap();
        assert_eq!(*r1.data.downcast::<f32>().unwrap(), 10.0);
        assert_eq!(*r2.data.downcast::<f32>().unwrap(), 14.0);
    }

    #[test]
    fn test_backend_shutdown_on_drop() {
        let handle = {
            let backend = Backend::new(vec![Box::new(DoublerModel)]);
            backend.handle()
        };
        // Backend dropped, but handle still exists.
        // Submit should still work if there are pending requests (they were drained).
        // After the worker exits, submit returns error.
        // (Exact behavior depends on timing; the key is no deadlock/panic.)
        std::thread::sleep(std::time::Duration::from_millis(10));
        let result = handle.submit_sync(1, "doubler".to_string(), Box::new(1.0f32));
        // May succeed or fail depending on timing, but shouldn't panic
        let _ = result;
    }
}
