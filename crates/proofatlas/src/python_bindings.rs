//! Python bindings for ProofAtlas using PyO3

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::logic::{Clause, Interner};
use crate::prover::Prover;
use crate::atlas::ProofAtlas;
use crate::state::{clause_indices, ProofResult, StateChange as RustStateChange};
use crate::config::LiteralSelectionStrategy;

/// Single step in proof trace
#[pyclass]
#[derive(Clone)]
pub struct ProofStep {
    #[pyo3(get)]
    pub clause_id: usize,
    #[pyo3(get)]
    pub clause_string: String,
    #[pyo3(get)]
    pub parent_ids: Vec<usize>,
    #[pyo3(get)]
    pub rule_name: String,
}

// =============================================================================
// PyProofAtlas — Python-facing "ProofAtlas" class
// =============================================================================

/// Python-accessible orchestrator — wraps Rust ProofAtlas.
///
/// Created once with config, reused across multiple problems.
/// Each `prove()` call returns a `Prover` with the result.
#[pyclass(name = "ProofAtlas", unsendable)]
pub struct PyProofAtlas {
    atlas: Arc<ProofAtlas>,
    /// Initial clause count from the last prove() call
    initial_count: Option<usize>,
    /// Worker pool (created by start_workers, None initially)
    pool_tx: Option<crossbeam_channel::Sender<WorkerTask>>,
    pool_rx: Option<crossbeam_channel::Receiver<WorkerResult>>,
    pool_threads: Vec<std::thread::JoinHandle<()>>,
}

#[pymethods]
impl PyProofAtlas {
    /// Create a new ProofAtlas orchestrator.
    ///
    /// Args:
    ///     timeout: Timeout in seconds (default: 300)
    ///     max_iterations: Maximum saturation steps (0 = no limit)
    ///     literal_selection: Literal selection strategy: 0/20/21/22 (default: 0)
    ///     age_weight_ratio: Age probability for age-weight selector (default: 0.5)
    ///     encoder: Encoder name: None (default, uses age_weight), "gcn", "gat", "graphsage", or "sentence"
    ///     scorer: Scorer name (e.g., "mlp", "attention")
    ///     weights_path: Path to model weights directory
    ///     memory_limit: Memory limit in MB
    ///     use_cuda: Whether to use CUDA (default: false)
    ///     enable_profiling: Enable structured profiling (default: false)
    ///     include_dir: Directory for resolving TPTP include() directives
    ///     max_clause_size: Maximum clause size (default: 100)
    ///     enable_trace: Enable MiniLM backend for trace embedding (default: false)
    ///     model_name: Model name for weight files (default: "{encoder}_{scorer}")
    ///     temperature: Softmax temperature for ML clause selection (default: 1.0)
    #[new]
    #[pyo3(signature = (timeout=None, max_iterations=None, literal_selection=None, age_weight_ratio=None, encoder=None, scorer=None, weights_path=None, memory_limit=None, use_cuda=None, enable_profiling=None, include_dir=None, max_clause_size=None, enable_trace=None, model_name=None, temperature=None))]
    pub fn new(
        timeout: Option<f64>,
        max_iterations: Option<usize>,
        literal_selection: Option<u32>,
        age_weight_ratio: Option<f64>,
        encoder: Option<String>,
        scorer: Option<String>,
        weights_path: Option<String>,
        memory_limit: Option<usize>,
        use_cuda: Option<bool>,
        enable_profiling: Option<bool>,
        include_dir: Option<String>,
        max_clause_size: Option<usize>,
        enable_trace: Option<bool>,
        model_name: Option<String>,
        temperature: Option<f32>,
    ) -> PyResult<Self> {
        let timeout_dur = timeout
            .map(|s| Duration::from_secs_f64(s))
            .unwrap_or(Duration::from_secs(300));

        let lit_sel = match literal_selection {
            Some(20) => LiteralSelectionStrategy::Sel20,
            Some(21) => LiteralSelectionStrategy::Sel21,
            Some(22) => LiteralSelectionStrategy::Sel22,
            _ => LiteralSelectionStrategy::Sel0,
        };

        let config = crate::config::ProverConfig {
            max_clauses: 0,
            max_iterations: max_iterations.unwrap_or(0),
            max_clause_size: max_clause_size.unwrap_or(100),
            timeout: timeout_dur,
            literal_selection: lit_sel,
            memory_limit,
            enable_profiling: enable_profiling.unwrap_or(false),
        };

        let mut builder = ProofAtlas::builder(config);

        if let Some(dir) = include_dir {
            builder = builder.include_dir(dir);
        }

        if let Some(enc) = encoder {
            builder = builder.encoder(enc);
        }
        if let Some(sc) = scorer {
            builder = builder.scorer(sc);
        }
        if let Some(path) = weights_path {
            builder = builder.weights_path(path);
        }
        if let Some(name) = model_name {
            builder = builder.model_name(name);
        }
        if let Some(cuda) = use_cuda {
            builder = builder.use_cuda(cuda);
        }
        if let Some(ratio) = age_weight_ratio {
            builder = builder.age_weight_ratio(ratio);
        }
        if let Some(trace) = enable_trace {
            builder = builder.enable_trace(trace);
        }
        if let Some(temp) = temperature {
            builder = builder.temperature(temp);
        }

        let atlas = builder.build()
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(PyProofAtlas {
            atlas: Arc::new(atlas),
            initial_count: None,
            pool_tx: None,
            pool_rx: None,
            pool_threads: Vec::new(),
        })
    }

    /// Prove a TPTP file.
    ///
    /// Args:
    ///     path: Path to TPTP problem file
    ///     timeout: Override timeout in seconds (currently unused, set on constructor)
    ///     memory_limit: Override memory limit in MB (currently unused, set on constructor)
    ///
    /// Returns:
    ///     Prover object with result, proof steps, and statistics
    #[pyo3(signature = (path, timeout=None, memory_limit=None))]
    pub fn prove(&mut self, path: &str, timeout: Option<f64>, memory_limit: Option<usize>) -> PyResult<PyProver> {
        let _ = (timeout, memory_limit);
        let parsed = self.parse_file_threaded(path)?;
        self.run_prover(parsed)
    }

    /// Prove TPTP content from a string.
    ///
    /// Args:
    ///     content: TPTP content as string
    ///     timeout: Override timeout in seconds (currently unused, set on constructor)
    ///     memory_limit: Override memory limit in MB (currently unused, set on constructor)
    ///
    /// Returns:
    ///     Prover object with result, proof steps, and statistics
    #[pyo3(signature = (content, timeout=None, memory_limit=None))]
    pub fn prove_string(&mut self, content: &str, timeout: Option<f64>, memory_limit: Option<usize>) -> PyResult<PyProver> {
        let _ = (timeout, memory_limit);
        let parsed = self.parse_string_threaded(content)?;
        self.run_prover(parsed)
    }

    // ── Worker pool ──────────────────────────────────────────────────

    /// Start N prover threads sharing this atlas's Backend.
    #[pyo3(signature = (n_workers, collect_traces=false, traces_dir=None, trace_preset=None, fallback_dirs=None))]
    pub fn start_workers(&mut self, n_workers: usize, collect_traces: bool,
                         traces_dir: Option<String>, trace_preset: Option<String>,
                         fallback_dirs: Option<Vec<String>>) -> PyResult<()> {
        let _ = fallback_dirs; // Deprecated: relabeling is now offline
        if self.pool_tx.is_some() {
            return Err(PyValueError::new_err("Workers already started"));
        }

        let (task_tx, task_rx) = crossbeam_channel::unbounded::<WorkerTask>();
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<WorkerResult>();

        let trace_config = match (traces_dir, trace_preset) {
            (Some(dir), Some(preset)) => Some(TraceConfig {
                traces_dir: dir,
                preset,
            }),
            _ => None,
        };

        for i in 0..n_workers {
            let atlas = Arc::clone(&self.atlas);
            let task_rx = task_rx.clone();
            let result_tx = result_tx.clone();
            let trace_config = trace_config.clone();

            let thread = std::thread::Builder::new()
                .name(format!("prover-worker-{}", i))
                .stack_size(128 * 1024 * 1024)
                .spawn(move || {
                    worker_loop(&atlas, &task_rx, &result_tx, collect_traces, trace_config.as_ref());
                })
                .map_err(|e| PyValueError::new_err(format!("Failed to spawn worker: {}", e)))?;

            self.pool_threads.push(thread);
        }

        self.pool_tx = Some(task_tx);
        self.pool_rx = Some(result_rx);
        Ok(())
    }

    /// Submit a problem to the worker pool.
    pub fn submit(&self, path: String) -> PyResult<()> {
        self.pool_tx.as_ref()
            .ok_or_else(|| PyValueError::new_err("No workers started. Call start_workers() first."))?
            .send(WorkerTask::Prove(path))
            .map_err(|_| PyValueError::new_err("Worker channel closed"))
    }

    /// Collect one result from the worker pool. Blocks until available or timeout.
    #[pyo3(signature = (timeout=None))]
    pub fn collect(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<Option<PyBatchResult>> {
        let rx = self.pool_rx.as_ref()
            .ok_or_else(|| PyValueError::new_err("No workers started. Call start_workers() first."))?;

        let deadline = timeout.map(|t| Instant::now() + Duration::from_secs_f64(t));

        loop {
            match rx.try_recv() {
                Ok(r) => return Ok(Some(PyBatchResult {
                    problem: r.problem,
                    status: r.status,
                    time_s: r.time_s,
                    clause_strings: r.clause_strings,
                    num_clauses: r.num_clauses,
                    iterations: r.iterations,
                    clause_bytes: r.clause_bytes,
                })),
                Err(crossbeam_channel::TryRecvError::Empty) => {},
                Err(crossbeam_channel::TryRecvError::Disconnected) => return Ok(None),
            }

            if let Some(d) = deadline {
                if Instant::now() >= d {
                    return Ok(None);
                }
            }

            py.check_signals()?;
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Shut down the worker pool.
    pub fn shutdown_workers(&mut self) {
        if let Some(tx) = self.pool_tx.take() {
            for _ in &self.pool_threads {
                let _ = tx.send(WorkerTask::Shutdown);
            }
            drop(tx);
        }
        self.pool_rx = None;
        for thread in self.pool_threads.drain(..) {
            let _ = thread.join();
        }
    }
}

impl PyProofAtlas {
    /// Parse in a large-stack thread, polling for Python signals.
    fn parse_file_threaded(&self, path: &str) -> PyResult<crate::parser::ParsedProblem> {
        let include_dirs: Vec<String> = self.atlas.include_dirs().to_vec();
        let config = self.atlas.config().clone();
        let path_owned = path.to_string();

        let handle = std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(move || {
                use crate::parser::parse_tptp_file;
                let include_refs: Vec<&str> = include_dirs.iter().map(|s| s.as_str()).collect();
                let timeout_instant = Some(Instant::now() + config.timeout);
                let max_clauses = if config.max_clauses > 0 { Some(config.max_clauses) } else { None };
                parse_tptp_file(&path_owned, &include_refs, timeout_instant, max_clauses, config.memory_limit)
                    .map_err(|e| e.message)
            })
            .map_err(|e| PyValueError::new_err(format!("Failed to spawn parser thread: {}", e)))?;

        self.poll_result_thread(handle)
    }

    /// Parse string content in a large-stack thread, polling for Python signals.
    fn parse_string_threaded(&self, content: &str) -> PyResult<crate::parser::ParsedProblem> {
        let include_dirs: Vec<String> = self.atlas.include_dirs().to_vec();
        let config = self.atlas.config().clone();
        let content_owned = content.to_string();

        let handle = std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(move || {
                use crate::parser::parse_tptp;
                let include_refs: Vec<&str> = include_dirs.iter().map(|s| s.as_str()).collect();
                let timeout_instant = Some(Instant::now() + config.timeout);
                let max_clauses = if config.max_clauses > 0 { Some(config.max_clauses) } else { None };
                parse_tptp(&content_owned, &include_refs, timeout_instant, max_clauses, config.memory_limit)
                    .map_err(|e| e.message)
            })
            .map_err(|e| PyValueError::new_err(format!("Failed to spawn parser thread: {}", e)))?;

        self.poll_result_thread(handle)
    }

    /// Poll a thread returning Result<T, String> for completion, checking Python signals.
    fn poll_result_thread<T>(&self, handle: std::thread::JoinHandle<Result<T, String>>) -> PyResult<T> {
        loop {
            if handle.is_finished() {
                return handle.join()
                    .map_err(|_| PyValueError::new_err("Thread panicked"))?
                    .map_err(|e| PyValueError::new_err(e));
            }
            Python::with_gil(|py| py.check_signals())?;
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Create sink, build prover, and run saturation in a large-stack thread.
    fn run_prover(&mut self, parsed: crate::parser::ParsedProblem) -> PyResult<PyProver> {
        let sink = self.atlas.create_sink(&parsed.interner)
            .map_err(|e| PyValueError::new_err(e))?;

        let config = self.atlas.config().clone();
        let clauses = parsed.formula.clauses;
        let interner = parsed.interner;

        // Run saturation in a large-stack thread
        let handle = std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(move || {
                let mut prover = Prover::new(clauses, config, sink, interner);
                let result = prover.prove();
                (result, prover)
            })
            .map_err(|e| PyValueError::new_err(format!("Failed to spawn saturation thread: {}", e)))?;

        let (result, prover) = loop {
            if handle.is_finished() {
                break handle.join()
                    .map_err(|_| PyValueError::new_err("Saturation thread panicked (possible stack overflow)"))?;
            }
            Python::with_gil(|py| py.check_signals())?;
            std::thread::sleep(Duration::from_millis(50));
        };

        let initial_count = prover.state.initial_clause_count;
        self.initial_count = Some(initial_count);

        let profile_json = prover.profile()
            .and_then(|p| serde_json::to_string(p).ok());

        #[cfg(feature = "ml")]
        let backend_handle = self.atlas.backend_handle();

        Ok(PyProver {
            prover,
            result,
            initial_count,
            profile_json,
            #[cfg(feature = "ml")]
            backend_handle,
        })
    }
}

impl Drop for PyProofAtlas {
    fn drop(&mut self) {
        self.shutdown_workers();
    }
}

// =============================================================================
// PyProver — holds prover state after prove()
// =============================================================================

/// Python-accessible prover result — holds the Prover and ProofResult.
///
/// Created by ProofAtlas.prove() / prove_string(). Provides access to
/// proof steps, statistics, traces, and profiling data.
#[pyclass(name = "Prover", unsendable)]
pub struct PyProver {
    prover: Prover,
    result: ProofResult,
    initial_count: usize,
    profile_json: Option<String>,
    #[cfg(feature = "ml")]
    backend_handle: Option<crate::selection::BackendHandle>,
}

impl PyProver {
    fn interner(&self) -> &Interner {
        self.prover.interner()
    }

    fn clauses(&self) -> &[std::sync::Arc<Clause>] {
        self.prover.clauses()
    }

    fn event_log(&self) -> &[RustStateChange] {
        self.prover.event_log()
    }

    fn empty_clause_idx(&self) -> Option<usize> {
        match &self.result {
            ProofResult::Proof { empty_clause_idx } => Some(*empty_clause_idx),
            _ => None,
        }
    }
}

#[pymethods]
impl PyProver {
    /// Whether a proof was found.
    #[getter]
    pub fn proof_found(&self) -> bool {
        matches!(self.result, ProofResult::Proof { .. })
    }

    /// Status string: "proof", "saturated", or "resource_limit".
    #[getter]
    pub fn status(&self) -> String {
        match &self.result {
            ProofResult::Proof { .. } => "proof".to_string(),
            ProofResult::Saturated => "saturated".to_string(),
            ProofResult::ResourceLimit => "resource_limit".to_string(),
        }
    }

    /// Number of initial clauses (from parsing).
    #[getter]
    pub fn initial_count(&self) -> usize {
        self.initial_count
    }

    /// Get profile JSON from the proof run.
    ///
    /// Returns None if profiling was not enabled.
    pub fn profile_json(&self) -> Option<String> {
        self.profile_json.clone()
    }

    /// Get the raw event log as JSON.
    pub fn trace_json(&self) -> PyResult<Option<String>> {
        let events = self.event_log();
        if events.is_empty() {
            return Ok(None);
        }
        let json = serde_json::to_string(events)
            .map_err(|e| PyValueError::new_err(format!("Trace serialization failed: {}", e)))?;
        Ok(Some(json))
    }

    /// Get the structured trace (iterations with simplification/selection/generation)
    /// as JSON, ready for the frontend ProofInspector.
    pub fn trace_iterations_json(&self) -> PyResult<Option<String>> {
        let events = self.event_log();
        if events.is_empty() {
            return Ok(None);
        }
        let interner = self.interner();
        let trace = crate::state::build_trace(events, |c| c.display(interner).to_string());
        let json = serde_json::to_string(&trace)
            .map_err(|e| PyValueError::new_err(format!("Trace serialization failed: {}", e)))?;
        Ok(Some(json))
    }

    /// Build the complete prove result as JSON (shared format with WASM).
    ///
    /// This is the single source of truth for the web API response format.
    pub fn prove_result_json(&self, elapsed_ms: u32) -> PyResult<String> {
        use crate::state::{
            build_trace, steps_to_wire, all_steps_wire, status_message,
            ProveResult as CoreResult, ProveStatistics,
        };

        let interner = self.interner();
        let events = self.event_log();
        let (status, msg) = status_message(&self.result);

        let (success, proof, message, final_clauses) = match &self.result {
            ProofResult::Proof { empty_clause_idx } => {
                let steps = self.prover.extract_proof(*empty_clause_idx);
                let wire = steps_to_wire(&steps, interner);
                let message = format!("Proof found with {} steps", wire.len());
                let final_count = self.clauses().len();
                (true, Some(wire), message, final_count)
            }
            _ => (false, None, msg.to_string(), self.clauses().len()),
        };

        let all_clauses = all_steps_wire(events, interner);

        let result = CoreResult {
            success,
            status: status.to_string(),
            message,
            proof,
            all_clauses: Some(all_clauses),
            statistics: ProveStatistics {
                initial_clauses: self.initial_count,
                generated_clauses: self.clauses().len(),
                final_clauses,
                time_ms: elapsed_ms,
            },
            trace: if !events.is_empty() {
                Some(build_trace(events, |c| c.display(interner).to_string()))
            } else {
                None
            },
            profile: self.profile_json.as_ref()
                .and_then(|s| serde_json::from_str(s).ok()),
        };

        serde_json::to_string(&result)
            .map_err(|e| PyValueError::new_err(format!("Serialization failed: {}", e)))
    }

    /// Get statistics.
    pub fn statistics(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        let clauses = self.clauses();
        stats.insert("total".to_string(), clauses.len());
        stats.insert("processed".to_string(), self.prover.state.processed.len());
        stats.insert("empty_clauses".to_string(), clauses.iter().filter(|c| c.is_empty()).count());
        stats.insert("unit_clauses".to_string(), clauses.iter().filter(|c| c.literals.len() == 1).count());
        stats.insert("iterations".to_string(), self.prover.iterations());
        stats.insert("clause_bytes".to_string(), self.prover.clause_bytes());
        stats
    }

    /// Get all proof steps from the saturation run.
    pub fn all_steps(&self) -> Vec<ProofStep> {
        let interner = self.interner();
        let events = self.event_log();
        let mut steps = Vec::new();
        for event in events {
            match event {
                RustStateChange::Add(clause, rule_name, premises) => {
                    if let Some(idx) = clause.id {
                        steps.push(ProofStep {
                            clause_id: idx,
                            clause_string: clause.display(interner).to_string(),
                            parent_ids: clause_indices(premises),
                            rule_name: rule_name.clone(),
                        });
                    }
                }
                RustStateChange::Simplify(_, Some(clause), rule_name, premises) => {
                    if let Some(idx) = clause.id {
                        steps.push(ProofStep {
                            clause_id: idx,
                            clause_string: clause.display(interner).to_string(),
                            parent_ids: clause_indices(premises),
                            rule_name: rule_name.clone(),
                        });
                    }
                }
                _ => {}
            }
        }
        steps
    }

    /// Get minimal proof trace (only steps in the proof DAG).
    pub fn proof_steps(&self) -> Vec<ProofStep> {
        let empty_id = match self.empty_clause_idx() {
            Some(idx) => idx,
            None => return Vec::new(),
        };

        let rust_steps = self.prover.extract_proof(empty_id);
        let interner = self.interner();

        rust_steps.iter().map(|step| {
            ProofStep {
                clause_id: step.clause_idx,
                clause_string: step.conclusion.display(interner).to_string(),
                parent_ids: clause_indices(&step.premises),
                rule_name: step.rule_name.clone(),
            }
        }).collect()
    }

    /// Save trace data as per-problem NPZ files.
    ///
    /// Writes {traces_dir}/{preset}/{stem}.graph.npz and
    /// {traces_dir}/{preset}/{stem}.sentence.npz with lifecycle encoding.
    ///
    /// If the ProofAtlas was built with `enable_trace=True`, pre-computes
    /// 384-D MiniLM embeddings for node names (graph) and clause strings (sentence).
    /// Save a proof trace for training.
    ///
    /// If `external_labels` is provided, those labels are used instead of deriving
    /// them from this run's proof. This allows saving traces for failed runs using
    /// labels from a different proof (e.g., the baseline).
    #[cfg(feature = "ml")]
    #[pyo3(signature = (traces_dir, preset, problem, time_seconds, external_labels=None))]
    pub fn save_trace(
        &self,
        traces_dir: &str,
        preset: &str,
        problem: &str,
        time_seconds: f64,
        external_labels: Option<Vec<u8>>,
    ) -> PyResult<()> {
        let _ = time_seconds;
        // Node/clause embeddings disabled — pass None for backend handle
        crate::selection::training::trace::save_trace(
            &self.prover, &self.result, None,
            traces_dir, preset, problem, external_labels,
        ).map_err(|e| PyValueError::new_err(e))
    }
}


// =============================================================================
// PyMiniLMBackend — wraps Backend + MiniLMEncoderModel for trace embedding
// =============================================================================

#[cfg(feature = "ml")]
#[pyclass(unsendable, name = "MiniLMBackend")]
pub struct PyMiniLMBackend {
    handle: crate::selection::BackendHandle,
    _backend: crate::selection::Backend,
}

#[cfg(feature = "ml")]
#[pymethods]
impl PyMiniLMBackend {
    #[new]
    #[pyo3(signature = (model_path, tokenizer_path, use_cuda=false))]
    fn new(model_path: String, tokenizer_path: String, use_cuda: bool) -> PyResult<Self> {
        let model = crate::selection::MiniLMEncoderModel::new(&model_path, &tokenizer_path, use_cuda)
            .map_err(|e| PyValueError::new_err(e))?;
        let backend = crate::selection::Backend::from_models(vec![Box::new(model)]);
        let handle = backend.handle();
        Ok(PyMiniLMBackend {
            handle,
            _backend: backend,
        })
    }
}

#[cfg(feature = "ml")]
impl PyMiniLMBackend {
    /// Encode strings via the backend (blocking).
    #[allow(dead_code)]
    fn encode_strings(&self, strings: Vec<String>) -> Vec<Vec<f32>> {
        if strings.is_empty() {
            return vec![];
        }
        let resp = self
            .handle
            .submit_sync(0, "minilm".to_string(), Box::new(strings), true)
            .expect("MiniLM backend submission failed");
        *resp.data.downcast::<Vec<Vec<f32>>>().expect("unexpected response type")
    }
}

// =============================================================================
// Worker pool types
// =============================================================================

/// Result from a worker pool problem.
#[pyclass(name = "BatchResult")]
#[derive(Clone)]
pub struct PyBatchResult {
    #[pyo3(get)]
    pub problem: String,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub time_s: f64,
    /// Clause strings from the prover (for trace fallback labeling).
    /// Only populated when trace collection is enabled.
    #[pyo3(get)]
    pub clause_strings: Option<Vec<String>>,
    /// Number of clauses in the prover state.
    #[pyo3(get)]
    pub num_clauses: usize,
    /// Number of given-clause iterations completed.
    #[pyo3(get)]
    pub iterations: usize,
    /// Estimated clause storage in bytes.
    #[pyo3(get)]
    pub clause_bytes: usize,
}

#[derive(Clone)]
struct TraceConfig {
    traces_dir: String,
    preset: String,
}

enum WorkerTask {
    Prove(String), // problem path
    Shutdown,
}

struct WorkerResult {
    problem: String,
    status: String,
    time_s: f64,
    clause_strings: Option<Vec<String>>,
    num_clauses: usize,
    iterations: usize,
    clause_bytes: usize,
}

/// Worker loop for prover threads (used by PyProofAtlas::start_workers).
fn worker_loop(
        atlas: &ProofAtlas,
        task_rx: &crossbeam_channel::Receiver<WorkerTask>,
        result_tx: &crossbeam_channel::Sender<WorkerResult>,
        collect_traces: bool,
        trace_config: Option<&TraceConfig>,
    ) {
        while let Ok(task) = task_rx.recv() {
            match task {
                WorkerTask::Shutdown => break,
                WorkerTask::Prove(path) => {
                    let start = Instant::now();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let parsed = match atlas.parse_file(&path) {
                            Ok(p) => p,
                            Err(e) => return Err((e.message, e.clause_count, e.clause_bytes)),
                        };
                        let parse_clause_count = parsed.clause_count;
                        let parse_clause_bytes = parsed.clause_bytes;
                        match atlas.prove_parsed(parsed) {
                            Ok((proof_result, prover)) => Ok((proof_result, prover, parse_clause_count, parse_clause_bytes)),
                            Err(e) => Err((e, parse_clause_count, parse_clause_bytes)),
                        }
                    }));

                    let elapsed = start.elapsed().as_secs_f64();

                    let (status, clause_strings, num_clauses, iterations, clause_bytes) = match result {
                        Ok(Ok((proof_result, prover, _parse_nc, _parse_cb))) => {
                            use crate::state::ProofResult;
                            let proof_found = matches!(proof_result, ProofResult::Proof { .. });
                            let status = match &proof_result {
                                ProofResult::Proof { .. } => "proof",
                                ProofResult::Saturated => "saturated",
                                ProofResult::ResourceLimit => "resource_limit",
                            };
                            let iterations = prover.iterations();
                            let clause_bytes = prover.clause_bytes();
                            // Save trace
                            #[cfg(feature = "ml")]
                            if let Some(tc) = trace_config {
                                let problem_name = std::path::Path::new(&path)
                                    .file_name()
                                    .map(|f| f.to_string_lossy().to_string())
                                    .unwrap_or_default();
                                // Save trace for all results (proofs get real labels,
                                // failures get all-zero labels). Offline relabeling
                                // can later assign labels from other configs' proofs.
                                // No MiniLM embeddings — gcn_struct doesn't use them.
                                let _ = crate::selection::training::trace::save_trace(
                                    &prover, &proof_result, None,
                                    &tc.traces_dir, &tc.preset, &problem_name, None,
                                );
                            }
                            #[cfg(not(feature = "ml"))]
                            let _ = (proof_found, trace_config);

                            let strings = if collect_traces {
                                let interner = prover.interner();
                                let clauses = prover.clauses();
                                Some(clauses.iter()
                                    .map(|c| c.display(interner).to_string())
                                    .collect())
                            } else {
                                None
                            };
                            let nc = prover.clauses().len();
                            (status.to_string(), strings, nc, iterations, clause_bytes)
                        }
                        Ok(Err((e, parse_nc, parse_cb))) => {
                            let s = e.to_lowercase();
                            if s.contains("timed out") || s.contains("memory limit") || s.contains("clause limit") {
                                ("resource_limit".to_string(), None, parse_nc, 0, parse_cb)
                            } else {
                                ("error".to_string(), None, parse_nc, 0, parse_cb)
                            }
                        }
                        Err(_) => ("error".to_string(), None, 0, 0, 0),
                    };

                    let problem = std::path::Path::new(&path)
                        .file_name()
                        .map(|f| f.to_string_lossy().to_string())
                        .unwrap_or(path.clone());

                    let _ = result_tx.send(WorkerResult {
                        problem,
                        status,
                        time_s: elapsed,
                        clause_strings,
                        num_clauses,
                        iterations,
                        clause_bytes,
                    });
                }
            }
        }
    }

#[pymodule]
fn proofatlas(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProofAtlas>()?;
    m.add_class::<PyProver>()?;
    m.add_class::<ProofStep>()?;
    m.add_class::<PyBatchResult>()?;
    #[cfg(feature = "ml")]
    m.add_class::<PyMiniLMBackend>()?;
    Ok(())
}
