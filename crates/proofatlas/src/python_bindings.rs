//! Python bindings for ProofAtlas using PyO3

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
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
// PyOrchestrator — Python-facing "ProofAtlas" class
// =============================================================================

/// Python-accessible orchestrator — wraps Rust ProofAtlas.
///
/// Created once with config, reused across multiple problems.
/// Each `prove()` call returns a `Prover` with the result.
#[pyclass(name = "ProofAtlas", unsendable)]
pub struct PyOrchestrator {
    atlas: ProofAtlas,
    /// Initial clause count from the last prove() call
    initial_count: Option<usize>,
}

#[pymethods]
impl PyOrchestrator {
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
    ///     socket_path: Path to scoring server Unix socket
    ///     include_dir: Directory for resolving TPTP include() directives
    ///     max_clause_size: Maximum clause size (default: 100)
    #[new]
    #[pyo3(signature = (timeout=None, max_iterations=None, literal_selection=None, age_weight_ratio=None, encoder=None, scorer=None, weights_path=None, memory_limit=None, use_cuda=None, enable_profiling=None, socket_path=None, include_dir=None, max_clause_size=None))]
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
        socket_path: Option<String>,
        include_dir: Option<String>,
        max_clause_size: Option<usize>,
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
        if let Some(cuda) = use_cuda {
            builder = builder.use_cuda(cuda);
        }
        if let Some(ratio) = age_weight_ratio {
            builder = builder.age_weight_ratio(ratio);
        }
        if let Some(path) = socket_path {
            builder = builder.socket_path(path);
        }

        let atlas = builder.build()
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(PyOrchestrator {
            atlas,
            initial_count: None,
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
}

impl PyOrchestrator {
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
                parse_tptp_file(&path_owned, &include_refs, timeout_instant, config.memory_limit)
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
                parse_tptp(&content_owned, &include_refs, timeout_instant, config.memory_limit)
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

        Ok(PyProver {
            prover,
            result,
            initial_count,
            profile_json,
        })
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

    /// Get statistics.
    pub fn statistics(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        let clauses = self.clauses();
        stats.insert("total".to_string(), clauses.len());
        stats.insert("processed".to_string(), self.prover.state.processed.len());
        stats.insert("empty_clauses".to_string(), clauses.iter().filter(|c| c.is_empty()).count());
        stats.insert("unit_clauses".to_string(), clauses.iter().filter(|c| c.literals.len() == 1).count());
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
    /// If a MiniLMBackend is provided, pre-computes 384-D MiniLM embeddings
    /// for node names (graph) and clause strings (sentence).
    #[cfg(feature = "ml")]
    #[pyo3(signature = (traces_dir, preset, problem, time_seconds, backend=None))]
    pub fn save_trace(
        &self,
        traces_dir: &str,
        preset: &str,
        problem: &str,
        time_seconds: f64,
        backend: Option<&PyMiniLMBackend>,
    ) -> PyResult<()> {
        use crate::selection::ml::graph::GraphBuilder;
        use crate::selection::training::npz::NpzWriter;
        use std::path::PathBuf;

        let interner = self.interner();
        let clauses = self.clauses();
        let num_clauses = clauses.len();

        if num_clauses == 0 {
            return Ok(());
        }

        // Only save traces for proofs
        if self.empty_clause_idx().is_none() {
            return Ok(());
        }

        let stem = std::path::Path::new(problem)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(problem);

        let preset_dir = PathBuf::from(traces_dir).join(preset);
        std::fs::create_dir_all(&preset_dir)
            .map_err(|e| PyValueError::new_err(format!("Failed to create traces dir: {}", e)))?;

        // --- Lifecycle arrays ---
        let (transfer_step, activate_step, simplify_step, num_steps, _) =
            self.build_clause_lifecycle();

        // --- Proof labels ---
        let proof_clauses = self.get_proof_clause_set();
        let labels: Vec<u8> = (0..num_clauses)
            .map(|i| if proof_clauses.contains(&i) { 1 } else { 0 })
            .collect();

        // --- Clause features [C, 9] ---
        let clause_features_flat = self.compute_clause_features_flat();

        // --- Graph data ---
        let clause_refs: Vec<&Clause> = clauses.iter().map(|c| c.as_ref()).collect();
        let batch_graph = GraphBuilder::build_from_clauses(&clause_refs);
        let node_names = GraphBuilder::collect_node_names(&clause_refs, interner);

        // Node features [N, 3] flat
        let node_features_flat: Vec<f32> = batch_graph
            .node_features
            .iter()
            .flat_map(|nf| nf.iter().copied())
            .collect();
        let num_nodes = batch_graph.num_nodes;

        // Edges
        let _total_edges = batch_graph.edge_indices.len();
        let edge_src: Vec<i32> = batch_graph.edge_indices.iter().map(|&(s, _)| s as i32).collect();
        let edge_dst: Vec<i32> = batch_graph.edge_indices.iter().map(|&(_, d)| d as i32).collect();

        // Offsets
        let mut node_offsets = Vec::with_capacity(num_clauses + 1);
        let mut edge_offsets = Vec::with_capacity(num_clauses + 1);
        node_offsets.push(0i64);
        edge_offsets.push(0i64);
        for i in 0..num_clauses {
            node_offsets.push(batch_graph.clause_boundaries[i].1 as i64);
            edge_offsets.push(batch_graph.edge_boundaries[i].1 as i64);
        }

        let num_steps_arr = [num_steps];

        // --- Pre-compute embeddings if backend available ---
        let node_embeddings: Option<Vec<f32>>;
        let node_sentinel_type: Option<Vec<i8>>;
        let clause_embeddings: Option<Vec<f32>>;

        if let Some(be) = backend {
            // Graph: classify nodes as sentinel or real, encode real symbols
            let mut sentinel_types = vec![-1i8; num_nodes]; // -1 = real symbol
            let mut real_indices = Vec::new();
            let mut real_names = Vec::new();

            for (i, name) in node_names.iter().enumerate() {
                match name.as_str() {
                    "VAR" => sentinel_types[i] = 0,
                    "CLAUSE" => sentinel_types[i] = 1,
                    "LIT" => sentinel_types[i] = 2,
                    _ => {
                        real_indices.push(i);
                        real_names.push(name.clone());
                    }
                }
            }

            // Deduplicate real names for efficient encoding
            let mut unique_names: Vec<String> = Vec::new();
            let mut name_to_uid: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for name in &real_names {
                if !name_to_uid.contains_key(name) {
                    name_to_uid.insert(name.clone(), unique_names.len());
                    unique_names.push(name.clone());
                }
            }

            let unique_embs = if unique_names.is_empty() {
                vec![]
            } else {
                be.encode_strings(unique_names.clone())
            };

            // Scatter back to node positions (384-D)
            let emb_dim = 384;
            let mut flat_node_emb = vec![0.0f32; num_nodes * emb_dim];
            for (j, &node_idx) in real_indices.iter().enumerate() {
                let uid = name_to_uid[&real_names[j]];
                let src = &unique_embs[uid];
                let dst_start = node_idx * emb_dim;
                flat_node_emb[dst_start..dst_start + emb_dim].copy_from_slice(src);
            }

            node_embeddings = Some(flat_node_emb);
            node_sentinel_type = Some(sentinel_types);

            // Sentence: encode clause display strings
            let clause_strings: Vec<String> = clauses
                .iter()
                .map(|c| c.display(interner).to_string())
                .collect();
            let clause_embs = be.encode_strings(clause_strings);
            let flat_clause_emb: Vec<f32> = clause_embs.into_iter().flatten().collect();
            clause_embeddings = Some(flat_clause_emb);
        } else {
            node_embeddings = None;
            node_sentinel_type = None;
            clause_embeddings = None;
        }

        // --- Write graph NPZ ---
        let graph_path = preset_dir.join(format!("{}.graph.npz", stem));
        let mut gw = NpzWriter::new(&graph_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to create graph NPZ: {}", e)))?;

        gw.write_array_2d("node_features", &node_features_flat, num_nodes, 3)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        gw.write_array_1d("edge_src", &edge_src)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        gw.write_array_1d("edge_dst", &edge_dst)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        gw.write_array_1d("node_offsets", &node_offsets)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        gw.write_array_1d("edge_offsets", &edge_offsets)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        gw.write_array_2d("clause_features", &clause_features_flat, num_clauses, 9)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        gw.write_array_1d("labels", &labels)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        gw.write_array_1d("transfer_step", &transfer_step)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        gw.write_array_1d("activate_step", &activate_step)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        gw.write_array_1d("simplify_step", &simplify_step)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        gw.write_array_1d("num_steps", &num_steps_arr)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;

        if let Some(ref embs) = node_embeddings {
            gw.write_array_2d("node_embeddings", embs, num_nodes, 384)
                .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        }
        if let Some(ref st) = node_sentinel_type {
            gw.write_array_1d("node_sentinel_type", st)
                .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        }

        gw.finish()
            .map_err(|e| PyValueError::new_err(format!("NPZ finish error: {}", e)))?;

        // --- Write sentence NPZ ---
        let sentence_path = preset_dir.join(format!("{}.sentence.npz", stem));
        let mut sw = NpzWriter::new(&sentence_path)
            .map_err(|e| PyValueError::new_err(format!("Failed to create sentence NPZ: {}", e)))?;

        sw.write_array_2d("clause_features", &clause_features_flat, num_clauses, 9)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        sw.write_array_1d("labels", &labels)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        sw.write_array_1d("transfer_step", &transfer_step)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        sw.write_array_1d("activate_step", &activate_step)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        sw.write_array_1d("simplify_step", &simplify_step)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        sw.write_array_1d("num_steps", &num_steps_arr)
            .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;

        if let Some(ref embs) = clause_embeddings {
            sw.write_array_2d("clause_embeddings", embs, num_clauses, 384)
                .map_err(|e| PyValueError::new_err(format!("NPZ write error: {}", e)))?;
        }

        sw.finish()
            .map_err(|e| PyValueError::new_err(format!("NPZ finish error: {}", e)))?;

        let _ = time_seconds; // reserved for future metadata

        Ok(())
    }
}

impl PyProver {
    /// Get the set of clause indices in the proof DAG
    fn get_proof_clause_set(&self) -> HashSet<usize> {
        let empty_id = match self.empty_clause_idx() {
            Some(idx) => idx,
            None => return HashSet::new(),
        };

        let events = self.event_log();

        let mut derivation_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for event in events {
            match event {
                RustStateChange::Add(clause, _, premises) => {
                    if let Some(idx) = clause.id {
                        derivation_map.insert(idx, clause_indices(premises));
                    }
                }
                RustStateChange::Simplify(_, Some(clause), _, premises) => {
                    if let Some(idx) = clause.id {
                        derivation_map.insert(idx, clause_indices(premises));
                    }
                }
                _ => {}
            }
        }

        let mut proof_clauses = HashSet::new();
        let mut to_visit = vec![empty_id];

        while let Some(current_id) = to_visit.pop() {
            if proof_clauses.contains(&current_id) {
                continue;
            }
            proof_clauses.insert(current_id);

            if let Some(parents) = derivation_map.get(&current_id) {
                to_visit.extend(parents);
            }
        }

        proof_clauses
    }

}

/// Recursively compute term statistics.
fn term_stats(
    term: &crate::logic::Term,
    distinct_symbols: &mut HashSet<u64>,
    distinct_variables: &mut HashSet<u64>,
) -> (usize, usize, usize) {
    use crate::logic::Term;
    match term {
        Term::Variable(v) => {
            distinct_variables.insert(v.id.0 as u64);
            (0, 0, 1)
        }
        Term::Constant(c) => {
            distinct_symbols.insert(c.id.0 as u64 | (2u64 << 32));
            (0, 1, 0)
        }
        Term::Function(f, args) => {
            distinct_symbols.insert(f.id.0 as u64 | (3u64 << 32));
            let mut max_depth = 0usize;
            let mut sc = 1usize;
            let mut vc = 0usize;
            for arg in args {
                let (d, s, v) = term_stats(arg, distinct_symbols, distinct_variables);
                max_depth = max_depth.max(d);
                sc += s;
                vc += v;
            }
            (max_depth + 1, sc, vc)
        }
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
    fn encode_strings(&self, strings: Vec<String>) -> Vec<Vec<f32>> {
        if strings.is_empty() {
            return vec![];
        }
        let resp = self
            .handle
            .submit_sync(0, "minilm".to_string(), Box::new(strings), false)
            .expect("MiniLM backend submission failed");
        *resp.data.downcast::<Vec<Vec<f32>>>().expect("unexpected response type")
    }
}

// =============================================================================
// save_trace() method on PyProver
// =============================================================================

impl PyProver {
    /// Build lifecycle arrays from the event log.
    ///
    /// Returns (transfer_step, activate_step, simplify_step, num_steps, num_clauses).
    fn build_clause_lifecycle(&self) -> (Vec<i32>, Vec<i32>, Vec<i32>, i32, usize) {
        let events = self.event_log();
        let num_clauses = self.clauses().len();

        let mut transfer_step = vec![-1i32; num_clauses];
        let mut activate_step = vec![-1i32; num_clauses];
        let mut simplify_step = vec![-1i32; num_clauses];
        let mut step: i32 = 0;

        for event in events {
            match event {
                RustStateChange::Transfer(idx) => {
                    if *idx < num_clauses {
                        transfer_step[*idx] = step;
                    }
                }
                RustStateChange::Activate(idx) => {
                    if *idx < num_clauses {
                        activate_step[*idx] = step;
                    }
                    step += 1;
                }
                RustStateChange::Simplify(idx, _replacement, _, _) => {
                    if *idx < num_clauses {
                        simplify_step[*idx] = step;
                    }
                    // replacement enters N, not U — no transfer_step set
                }
                RustStateChange::Add(_, _, _) => {
                    // enters N only
                }
            }
        }

        (transfer_step, activate_step, simplify_step, step, num_clauses)
    }

    /// Compute clause features [num_clauses, 9] as flat Vec<f32>.
    fn compute_clause_features_flat(&self) -> Vec<f32> {
        let clauses = self.clauses();
        let events = self.event_log();
        let num_clauses = clauses.len();

        // Build derivation info
        let mut derivation_info: HashMap<usize, String> = HashMap::new();
        for event in events {
            match event {
                RustStateChange::Add(clause, rule_name, _) => {
                    if let Some(idx) = clause.id {
                        derivation_info.insert(idx, rule_name.clone());
                    }
                }
                RustStateChange::Simplify(_, Some(clause), rule_name, _) => {
                    if let Some(idx) = clause.id {
                        derivation_info.insert(idx, rule_name.clone());
                    }
                }
                _ => {}
            }
        }

        let mut features = Vec::with_capacity(num_clauses * 9);
        for (idx, clause) in clauses.iter().enumerate() {
            let rule_name = derivation_info.get(&idx).map(|s| s.as_str()).unwrap_or("input");
            let rule_id = match rule_name {
                "input" => 0.0f32,
                "resolution" => 1.0,
                "factoring" => 2.0,
                "superposition" => 3.0,
                "equality_resolution" => 4.0,
                "equality_factoring" => 5.0,
                "demodulation" => 6.0,
                _ => 0.0,
            };

            let mut depth: usize = 0;
            let mut symbol_count: usize = 0;
            let mut variable_count: usize = 0;
            let mut distinct_symbols: HashSet<u64> = HashSet::new();
            let mut distinct_variables: HashSet<u64> = HashSet::new();

            for lit in &clause.literals {
                symbol_count += 1;
                distinct_symbols.insert(lit.predicate.id.0 as u64 | (1u64 << 32));
                for arg in &lit.args {
                    let (d, sc, vc) = term_stats(arg, &mut distinct_symbols, &mut distinct_variables);
                    depth = depth.max(d);
                    symbol_count += sc;
                    variable_count += vc;
                }
            }

            features.extend_from_slice(&[
                clause.age as f32,
                clause.role.to_feature_value(),
                rule_id,
                clause.literals.len() as f32,
                depth as f32,
                symbol_count as f32,
                distinct_symbols.len() as f32,
                variable_count as f32,
                distinct_variables.len() as f32,
            ]);
        }
        features
    }
}

/// Start a scoring server that blocks until the process is terminated.
#[cfg(feature = "ml")]
#[pyfunction]
#[pyo3(signature = (encoder, scorer, weights_path, socket_path, use_cuda=None))]
fn start_scoring_server(
    encoder: &str,
    scorer: &str,
    weights_path: &str,
    socket_path: &str,
    use_cuda: Option<bool>,
) -> PyResult<()> {
    let weights_dir = std::path::PathBuf::from(weights_path);
    let model_name = format!("{}_{}", encoder, scorer);
    let cuda = use_cuda.unwrap_or(false);

    let (embedder, scorer_box): (
        Box<dyn crate::selection::cached::ClauseEmbedder>,
        Box<dyn crate::selection::cached::EmbeddingScorer>,
    ) = if encoder == "sentence" {
        let model_path = weights_dir.join(format!("{}.pt", model_name));
        let tokenizer_path = weights_dir.join(format!("{}_tokenizer/tokenizer.json", model_name));
        if !model_path.exists() {
            return Err(PyValueError::new_err(format!(
                "Model not found at {}",
                model_path.display()
            )));
        }
        let emb = crate::selection::load_sentence_embedder(&model_path, &tokenizer_path, cuda)
            .map_err(|e| PyValueError::new_err(format!("Failed to load model: {}", e)))?;
        (
            Box::new(emb),
            Box::new(crate::selection::PassThroughScorer),
        )
    } else {
        let model_path = weights_dir.join(format!("{}.pt", model_name));
        if !model_path.exists() {
            return Err(PyValueError::new_err(format!(
                "Model not found at {}",
                model_path.display()
            )));
        }
        let emb = crate::selection::load_gcn_embedder(&model_path, cuda)
            .map_err(|e| PyValueError::new_err(format!("Failed to load model: {}", e)))?;
        (Box::new(emb), Box::new(crate::selection::GcnScorer))
    };

    let server = crate::selection::ScoringServer::new(
        embedder,
        scorer_box,
        socket_path.to_string(),
    );

    server.run();
    Ok(())
}

#[pymodule]
fn proofatlas(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOrchestrator>()?;
    m.add_class::<PyProver>()?;
    m.add_class::<ProofStep>()?;
    #[cfg(feature = "ml")]
    m.add_class::<PyMiniLMBackend>()?;
    #[cfg(feature = "ml")]
    m.add_function(wrap_pyfunction!(start_scoring_server, m)?)?;
    Ok(())
}
