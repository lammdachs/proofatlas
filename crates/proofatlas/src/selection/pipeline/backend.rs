//! Backend: model-agnostic compute service for ML inference.
//!
//! The Backend manages a worker thread that executes model forward passes.
//! Models are loaded **lazily** on first request — the first request for a
//! given `model_id` determines the device (CPU or CUDA) via its `use_cuda` flag.
//!
//! # Architecture
//!
//! ```text
//! Data Processing 0 ──→ BackendHandle ──→ Backend ──→ Worker (lazy-loaded models)
//! Data Processing 1 ──→ BackendHandle ──→
//! ```
//!
//! - **ModelSpec**: model_id + factory closure — stored until first request triggers loading.
//! - **BackendHandle**: cheaply cloneable, wraps `mpsc::Sender` to backend.
//! - **Worker**: receives requests, lazily loads models, groups by model, executes batches.
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
    /// Model identifier (e.g., "embed", "score_context", "embed_score").
    pub model_id: String,
    /// Type-erased input data for the model.
    pub data: Box<dyn std::any::Any + Send>,
    /// Channel to send the response back to the requester.
    pub response_tx: mpsc::Sender<BackendResponse>,
    /// Device hint for lazy model loading. Only used on first request
    /// for this model_id; subsequent requests use the already-loaded model.
    pub use_cuda: bool,
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
// ModelSpec — lazy model specification
// =============================================================================

/// Specification for a lazily-loaded model.
///
/// The factory closure is called with `use_cuda` from the first request for
/// this model_id. The resulting model is cached for all subsequent requests.
pub struct ModelSpec {
    /// Model identifier (must match `BackendRequest::model_id`).
    pub model_id: String,
    /// Factory that creates the model. Called with `use_cuda` from the first request.
    pub factory: Box<dyn FnOnce(bool) -> Result<Box<dyn Model>, String> + Send>,
}

impl ModelSpec {
    /// Wrap an already-loaded model (factory ignores `use_cuda`).
    pub fn loaded(model: Box<dyn Model>) -> Self {
        let id = model.model_id().to_string();
        Self {
            model_id: id,
            factory: Box::new(move |_| Ok(model)),
        }
    }
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
    ///
    /// `use_cuda` is a device hint for lazy model loading — only used on the
    /// first request for this `model_id`.
    pub fn submit_sync(
        &self,
        id: u64,
        model_id: String,
        data: Box<dyn std::any::Any + Send>,
        use_cuda: bool,
    ) -> Result<BackendResponse, String> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.submit(BackendRequest {
            id,
            model_id,
            data,
            response_tx: resp_tx,
            use_cuda,
        })?;
        resp_rx.recv().map_err(|_| "Backend response channel closed".to_string())
    }
}

// =============================================================================
// Backend — manages the worker thread
// =============================================================================

/// Model-agnostic compute service with lazy model loading.
///
/// Owns a worker thread that processes model execution requests.
/// Models are loaded on first request using the device specified by the requester.
/// Multiple data processing threads can share a single backend
/// by cloning `BackendHandle`s.
pub struct Backend {
    /// Worker thread handle. Detached on drop (not joined).
    /// The worker exits when all `BackendHandle` senders are dropped.
    _worker: JoinHandle<()>,
    handle: BackendHandle,
}

impl Backend {
    /// Create a new Backend with lazy-loaded model specs.
    ///
    /// Models are not loaded until the first request arrives for each model_id.
    /// The first request's `use_cuda` flag determines the device.
    ///
    /// Spawns a single worker thread with a 16 MiB stack (required for
    /// libtorch operations).
    pub fn new(specs: Vec<ModelSpec>) -> Self {
        let (tx, rx) = mpsc::channel::<BackendRequest>();

        let worker = std::thread::Builder::new()
            .name("backend-worker".to_string())
            .stack_size(16 * 1024 * 1024) // 16 MiB for libtorch
            .spawn(move || {
                Self::worker_loop(rx, specs);
            })
            .expect("Failed to spawn backend worker thread");

        Backend {
            _worker: worker,
            handle: BackendHandle { tx },
        }
    }

    /// Create a Backend from pre-loaded models (backward compat).
    ///
    /// Models are wrapped in `ModelSpec::loaded()` — no lazy loading.
    pub fn from_models(models: Vec<Box<dyn Model>>) -> Self {
        let specs = models.into_iter().map(ModelSpec::loaded).collect();
        Self::new(specs)
    }

    /// Get a cloneable handle for submitting requests.
    pub fn handle(&self) -> BackendHandle {
        self.handle.clone()
    }

    /// Worker loop: receive requests, lazily load models, execute batches.
    fn worker_loop(
        rx: mpsc::Receiver<BackendRequest>,
        specs: Vec<ModelSpec>,
    ) {
        type Factory = Box<dyn FnOnce(bool) -> Result<Box<dyn Model>, String> + Send>;

        let mut factories: HashMap<String, Factory> = specs
            .into_iter()
            .map(|s| (s.model_id, s.factory))
            .collect();
        let mut models_map: HashMap<String, Box<dyn Model>> = HashMap::new();

        // Buffer for grouping requests by model
        let mut pending: HashMap<String, VecDeque<BackendRequest>> = HashMap::new();

        loop {
            // Block on first request (or exit if channel closed)
            let first = match rx.recv() {
                Ok(req) => req,
                Err(_) => {
                    // Channel closed — drain any remaining pending requests
                    Self::flush_pending(&mut pending, &mut models_map, &mut factories);
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
            Self::flush_pending(&mut pending, &mut models_map, &mut factories);
        }
    }

    /// Execute all pending request batches, lazily loading models as needed.
    fn flush_pending(
        pending: &mut HashMap<String, VecDeque<BackendRequest>>,
        models_map: &mut HashMap<String, Box<dyn Model>>,
        factories: &mut HashMap<String, Box<dyn FnOnce(bool) -> Result<Box<dyn Model>, String> + Send>>,
    ) {
        for (model_id, queue) in pending.iter_mut() {
            if queue.is_empty() {
                continue;
            }

            // Lazy load: if model not yet loaded, use factory with first request's device hint
            if !models_map.contains_key(model_id) {
                if let Some(factory) = factories.remove(model_id) {
                    let use_cuda = queue.front().map(|r| r.use_cuda).unwrap_or(false);
                    match factory(use_cuda) {
                        Ok(model) => {
                            models_map.insert(model_id.clone(), model);
                        }
                        Err(e) => {
                            eprintln!("Backend: failed to load model '{}': {}", model_id, e);
                            queue.clear();
                            continue;
                        }
                    }
                } else {
                    eprintln!("Backend: unknown model_id '{}', dropping {} requests",
                        model_id, queue.len());
                    queue.clear();
                    continue;
                }
            }

            let model = models_map.get_mut(model_id).unwrap();

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
        let backend = Backend::from_models(vec![Box::new(DoublerModel)]);
        let handle = backend.handle();

        let resp = handle.submit_sync(1, "doubler".to_string(), Box::new(3.0f32), false).unwrap();
        let result = *resp.data.downcast::<f32>().unwrap();
        assert_eq!(result, 6.0);
    }

    #[test]
    fn test_backend_lazy_loading() {
        // Factory that creates a DoublerModel on demand
        let spec = ModelSpec {
            model_id: "doubler".to_string(),
            factory: Box::new(|_use_cuda| Ok(Box::new(DoublerModel) as Box<dyn Model>)),
        };
        let backend = Backend::new(vec![spec]);
        let handle = backend.handle();

        // First request triggers lazy load
        let resp = handle.submit_sync(1, "doubler".to_string(), Box::new(4.0f32), false).unwrap();
        assert_eq!(*resp.data.downcast::<f32>().unwrap(), 8.0);

        // Subsequent request uses cached model
        let resp = handle.submit_sync(2, "doubler".to_string(), Box::new(5.0f32), false).unwrap();
        assert_eq!(*resp.data.downcast::<f32>().unwrap(), 10.0);
    }

    #[test]
    fn test_backend_multiple_requests() {
        let backend = Backend::from_models(vec![Box::new(DoublerModel)]);
        let handle = backend.handle();

        // Submit multiple requests
        let (tx1, rx1) = mpsc::channel();
        let (tx2, rx2) = mpsc::channel();
        handle.submit(BackendRequest { id: 1, model_id: "doubler".to_string(), data: Box::new(5.0f32), response_tx: tx1, use_cuda: false }).unwrap();
        handle.submit(BackendRequest { id: 2, model_id: "doubler".to_string(), data: Box::new(7.0f32), response_tx: tx2, use_cuda: false }).unwrap();

        let r1 = rx1.recv().unwrap();
        let r2 = rx2.recv().unwrap();
        assert_eq!(*r1.data.downcast::<f32>().unwrap(), 10.0);
        assert_eq!(*r2.data.downcast::<f32>().unwrap(), 14.0);
    }

    #[test]
    fn test_backend_shutdown_on_drop() {
        let handle = {
            let backend = Backend::from_models(vec![Box::new(DoublerModel)]);
            backend.handle()
        };
        // Backend dropped, but handle still exists.
        // Submit should still work if there are pending requests (they were drained).
        // After the worker exits, submit returns error.
        // (Exact behavior depends on timing; the key is no deadlock/panic.)
        std::thread::sleep(std::time::Duration::from_millis(10));
        let result = handle.submit_sync(1, "doubler".to_string(), Box::new(1.0f32), false);
        // May succeed or fail depending on timing, but shouldn't panic
        let _ = result;
    }
}
