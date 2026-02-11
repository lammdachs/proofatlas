//! Python bindings for ProofAtlas using PyO3

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::logic::{Clause, Interner};
use crate::prover::ProofAtlas;
use crate::state::{clause_indices, ProofResult, StateChange as RustStateChange};
use crate::parser::parse_tptp;
use crate::config::LiteralSelectionStrategy;

/// Python-accessible prover — thin wrapper around ProofAtlas.
///
/// Pre-prove: holds initial clauses and interner from parsing.
/// Post-prove: retains the ProofAtlas instance and delegates all queries to it.
#[pyclass(name = "ProofAtlas", unsendable)]
pub struct PyProofAtlas {
    /// Pre-prove state: initial clauses from add_clauses_from_tptp
    initial_clauses: Vec<Clause>,
    /// Pre-prove state: interner from parsing
    interner: Interner,

    /// Post-prove state: the retained prover instance
    prover: Option<ProofAtlas>,
    /// Post-prove state: the saturation result
    result: Option<ProofResult>,
    /// Profile JSON (serialized once after prove())
    profile_json: Option<String>,
}

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

impl PyProofAtlas {
    /// Get the interner (works pre- and post-prove)
    fn interner(&self) -> &Interner {
        self.prover.as_ref()
            .map(|p| p.interner())
            .unwrap_or(&self.interner)
    }

    /// Get the clauses (works pre- and post-prove)
    fn clauses(&self) -> &[Clause] {
        self.prover.as_ref()
            .map(|p| p.clauses())
            .unwrap_or(&self.initial_clauses)
    }

    /// Get the event log (post-prove only)
    fn event_log(&self) -> &[RustStateChange] {
        self.prover.as_ref()
            .map(|p| p.event_log())
            .unwrap_or(&[])
    }

    /// Get the empty clause index (post-prove, proof case only)
    fn empty_clause_idx(&self) -> Option<usize> {
        match &self.result {
            Some(ProofResult::Proof { empty_clause_idx }) => Some(*empty_clause_idx),
            _ => None,
        }
    }
}

#[pymethods]
impl PyProofAtlas {
    /// Create empty proof state
    #[new]
    pub fn new() -> Self {
        PyProofAtlas {
            initial_clauses: Vec::new(),
            interner: Interner::new(),
            prover: None,
            result: None,
            profile_json: None,
        }
    }

    /// Parse TPTP content and add clauses, return clause IDs
    ///
    /// Args:
    ///     content: TPTP file content as string
    ///     include_dir: Optional directory to search for included files (e.g., TPTP root)
    ///     timeout: Optional timeout in seconds for CNF conversion (prevents hangs on complex formulas)
    ///     memory_limit: Optional memory limit in MB (checked via process RSS during CNF conversion)
    #[pyo3(signature = (content, include_dir=None, timeout=None, memory_limit=None))]
    pub fn add_clauses_from_tptp(
        &mut self,
        content: &str,
        include_dir: Option<&str>,
        timeout: Option<f64>,
        memory_limit: Option<usize>,
    ) -> PyResult<Vec<usize>> {
        let timeout_instant = timeout.map(|t| Instant::now() + Duration::from_secs_f64(t));
        let include_dirs: Vec<String> = include_dir.into_iter().map(|s| s.to_string()).collect();
        let content_owned = content.to_string();

        // Run parsing in a thread with larger stack to handle deeply nested formulas.
        // Default Python thread stack is too small for formulas with depth > 2000.
        // Poll the join handle so Ctrl+C works during parsing.
        let handle = std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)  // 128MB stack
            .spawn(move || {
                let include_refs: Vec<&str> = include_dirs.iter().map(|s| s.as_str()).collect();
                parse_tptp(&content_owned, &include_refs, timeout_instant, memory_limit)
            })
            .map_err(|e| PyValueError::new_err(format!("Failed to spawn parser thread: {}", e)))?;

        let parsed = loop {
            if handle.is_finished() {
                break handle.join()
                    .map_err(|_| PyValueError::new_err("Parser thread panicked"))?
                    .map_err(|e| PyValueError::new_err(format!("Parse error: {}", e)))?;
            }
            Python::with_gil(|py| py.check_signals())?;
            std::thread::sleep(Duration::from_millis(50));
        };

        // Extract interner and formula from parsed problem
        self.interner = parsed.interner;
        let cnf = parsed.formula;

        let mut ids = Vec::new();
        for mut clause in cnf.clauses {
            let id = self.initial_clauses.len();
            clause.id = Some(id);
            ids.push(id);
            self.initial_clauses.push(clause);
        }

        Ok(ids)
    }

    /// Run full saturation using the Rust saturation engine
    ///
    /// Args:
    ///     timeout: Optional timeout in seconds
    ///     max_iterations: Maximum number of saturation steps (0 or None = no limit)
    ///     literal_selection: Literal selection strategy: 0/20/21/22 (default: 0)
    ///     age_weight_ratio: Age probability for age-weight clause selector (default: 0.5)
    ///     encoder: Encoder name: None (default, uses age_weight), "gcn", "gat", "graphsage", or "sentence"
    ///     scorer: Name of the scorer. Model file is "{encoder}_{scorer}.pt" (e.g., "gcn_mlp").
    ///     weights_path: Path to model weights directory
    ///     memory_limit: Memory limit for clause storage in MB
    ///     enable_profiling: Enable structured profiling (default: false).
    ///     socket_path: Path to scoring server Unix socket. If set, uses RemoteSelector.
    ///                  If None with an ML encoder, auto-launches a local scoring server.
    ///
    /// Returns:
    ///     Tuple of (proof_found: bool, status: str)
    #[pyo3(signature = (timeout=None, max_iterations=None, literal_selection=None, age_weight_ratio=None, encoder=None, scorer=None, weights_path=None, memory_limit=None, use_cuda=None, enable_profiling=None, socket_path=None))]
    pub fn prove(
        &mut self,
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
    ) -> PyResult<(bool, String)> {
        use crate::config::ProverConfig;
        use crate::selection::AgeWeightSelector;

        // Create clause selector based on encoder type
        let model_name = match (encoder.as_deref(), scorer.as_deref()) {
            (Some(enc), Some(sc)) => Some(format!("{}_{}", enc, sc)),
            (Some(_), None) => return Err(PyValueError::new_err("scorer required when encoder is set")),
            _ => None,
        };

        // Keep auto-launched server handle alive until saturation completes
        #[cfg(feature = "ml")]
        let _server_handle: Option<std::thread::JoinHandle<()>> = None;
        #[cfg(feature = "ml")]
        let mut _server_handle = _server_handle;
        let auto_socket_path: Option<String>;

        let clause_selector: Box<dyn crate::selection::ClauseSelector> = match encoder.as_deref() {
            None => {
                auto_socket_path = None;
                // No encoder = heuristic selector
                let ratio = age_weight_ratio.unwrap_or(0.5);
                Box::new(AgeWeightSelector::new(ratio))
            }
            #[cfg(feature = "ml")]
            Some(enc @ ("gcn" | "gat" | "graphsage" | "sentence")) => {
                if let Some(ref path) = socket_path {
                    // Connect to existing scoring server
                    auto_socket_path = None;
                    let selector = crate::selection::RemoteSelector::connect(path)
                        .map_err(|e| PyValueError::new_err(e))?;
                    Box::new(selector)
                } else {
                    // Auto-launch a local scoring server
                    let weights_dir = if let Some(path) = weights_path.as_ref() {
                        std::path::PathBuf::from(path)
                    } else {
                        std::path::PathBuf::from(".weights")
                    };
                    let name = model_name.as_deref().unwrap();

                    let (embedder, scorer_box): (Box<dyn crate::selection::cached::ClauseEmbedder>, Box<dyn crate::selection::cached::EmbeddingScorer>) = if enc == "sentence" {
                        let model_path = weights_dir.join(format!("{}.pt", name));
                        let tokenizer_path = weights_dir.join(format!("{}_tokenizer/tokenizer.json", name));
                        if !model_path.exists() {
                            return Err(PyValueError::new_err(format!(
                                "Model not found at {}",
                                model_path.display()
                            )));
                        }
                        let emb = crate::selection::load_sentence_embedder(
                            &model_path,
                            &tokenizer_path,
                            use_cuda.unwrap_or(true),
                        ).map_err(|e| PyValueError::new_err(format!("Failed to load model: {}", e)))?;
                        (Box::new(emb), Box::new(crate::selection::PassThroughScorer))
                    } else {
                        let model_path = weights_dir.join(format!("{}.pt", name));
                        if !model_path.exists() {
                            return Err(PyValueError::new_err(format!(
                                "Model not found at {}",
                                model_path.display()
                            )));
                        }
                        let emb = crate::selection::load_gcn_embedder(
                            &model_path,
                            use_cuda.unwrap_or(true),
                        ).map_err(|e| PyValueError::new_err(format!("Failed to load model: {}", e)))?;
                        (Box::new(emb), Box::new(crate::selection::GcnScorer))
                    };

                    let sock_path = format!("/tmp/proofatlas-scoring-{}.sock", std::process::id());
                    let server = crate::selection::ScoringServer::new(
                        embedder,
                        scorer_box,
                        sock_path.clone(),
                    );
                    _server_handle = Some(server.spawn());
                    std::thread::sleep(Duration::from_millis(50));

                    auto_socket_path = Some(sock_path.clone());
                    let selector = crate::selection::RemoteSelector::connect(&sock_path)
                        .map_err(|e| PyValueError::new_err(e))?;
                    Box::new(selector)
                }
            }
            Some(other) => {
                #[cfg(feature = "ml")]
                let available = "None, 'gcn', 'gat', 'graphsage', or 'sentence'";
                #[cfg(not(feature = "ml"))]
                let available = "None (ML features not enabled)";
                return Err(PyValueError::new_err(format!(
                    "Unknown encoder: '{}'. Use {}",
                    other, available
                )));
            }
        };

        // Build config
        let timeout_dur = timeout
            .map(|s| Duration::from_secs_f64(s))
            .unwrap_or(Duration::from_secs(300));

        let lit_sel = match literal_selection {
            Some(20) => LiteralSelectionStrategy::Sel20,
            Some(21) => LiteralSelectionStrategy::Sel21,
            Some(22) => LiteralSelectionStrategy::Sel22,
            _ => LiteralSelectionStrategy::Sel0,
        };

        let config = ProverConfig {
            max_clauses: 0,
            max_iterations: max_iterations.unwrap_or(0),
            max_clause_size: 100,
            timeout: timeout_dur,
            literal_selection: lit_sel,
            memory_limit,
            enable_profiling: enable_profiling.unwrap_or(false),
        };

        // Create prover from current clauses — move initial_clauses/interner into prover
        let clauses = std::mem::take(&mut self.initial_clauses);
        let interner = std::mem::take(&mut self.interner);
        let mut prover = ProofAtlas::new(clauses, config, clause_selector, interner);

        // Run saturation in a thread with larger stack, move prover in and back out.
        // Poll the join handle so we can check for Python signals (Ctrl+C) between polls.
        let cancel = prover.cancel.clone();
        let handle = std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)  // 128MB stack
            .spawn(move || {
                let result = prover.prove();
                (prover, result)
            })
            .map_err(|e| PyValueError::new_err(format!("Failed to spawn saturation thread: {}", e)))?;

        let (prover, result) = loop {
            if handle.is_finished() {
                break handle.join()
                    .map_err(|_| PyValueError::new_err("Saturation thread panicked (possible stack overflow)"))?;
            }
            // Check for Python signals (KeyboardInterrupt from Ctrl+C)
            Python::with_gil(|py| py.check_signals())
                .map_err(|e| {
                    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    e
                })?;
            std::thread::sleep(Duration::from_millis(50));
        };

        // Serialize profile to JSON if present
        self.profile_json = prover.profile()
            .map(|p| serde_json::to_string(p).ok())
            .flatten();

        let (proof_found, status) = match &result {
            ProofResult::Proof { .. } => (true, "proof"),
            ProofResult::Saturated => (false, "saturated"),
            ProofResult::ResourceLimit => (false, "resource_limit"),
        };

        self.result = Some(result);
        self.prover = Some(prover);

        // Clean up auto-launched server socket
        if let Some(ref path) = auto_socket_path {
            let _ = std::fs::remove_file(path);
        }

        Ok((proof_found, status.to_string()))
    }

    /// Get profile JSON from the last prove() call.
    ///
    /// Returns None if profiling was not enabled.
    pub fn profile_json(&self) -> Option<String> {
        self.profile_json.clone()
    }

    /// Get the raw event log from the last prove() call as JSON.
    ///
    /// Returns the serialized Vec<StateChange> used by the web trace converter.
    pub fn trace_json(&self) -> PyResult<Option<String>> {
        let events = self.event_log();
        if events.is_empty() {
            return Ok(None);
        }
        let json = serde_json::to_string(events)
            .map_err(|e| PyValueError::new_err(format!("Trace serialization failed: {}", e)))?;
        Ok(Some(json))
    }

    /// Get statistics
    pub fn statistics(&self) -> HashMap<String, usize> {
        let clauses = self.clauses();
        let mut stats = HashMap::new();
        stats.insert("total".to_string(), clauses.len());

        // processed count from prover state if available
        let processed_count = self.prover.as_ref()
            .map(|p| p.state.processed.len())
            .unwrap_or(0);
        stats.insert("processed".to_string(), processed_count);

        // Count empty clauses
        let empty_count = clauses.iter().filter(|c| c.is_empty()).count();
        stats.insert("empty_clauses".to_string(), empty_count);

        // Count unit clauses
        let unit_count = clauses.iter().filter(|c| c.literals.len() == 1).count();
        stats.insert("unit_clauses".to_string(), unit_count);

        stats
    }

    /// Get all proof steps from the last saturation run.
    /// Returns every step including all derivations.
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

        let prover = match &self.prover {
            Some(p) => p,
            None => return Vec::new(),
        };

        // Use extract_proof on the prover to get proof steps
        let rust_steps = prover.extract_proof(empty_id);
        let interner = prover.interner();

        rust_steps.iter().map(|step| {
            ProofStep {
                clause_id: step.clause_idx,
                clause_string: step.conclusion.display(interner).to_string(),
                parent_ids: clause_indices(&step.premises),
                rule_name: step.rule_name.clone(),
            }
        }).collect()
    }

    /// Extract tensor trace data for ML training.
    ///
    /// Returns a tuple of two Python dicts (graph_dict, sentence_dict), each containing
    /// numpy arrays ready to be saved as .npz files.
    ///
    /// The graph dict uses `GraphBuilder::build_from_clauses()` — the same function
    /// used at inference time by the scoring server — as the single source of truth
    /// for clause→graph conversion.
    ///
    /// The sentence dict uses `clause.display(interner)` — the same method used at
    /// inference time by the sentence encoder.
    pub fn extract_tensor_trace<'py>(&self, py: Python<'py>, time_seconds: f64)
        -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)>
    {
        use numpy::{PyArray1, PyArray2};
        use crate::selection::graph::GraphBuilder;

        let interner = self.interner();
        let clauses = self.clauses();
        let events = self.event_log();
        let num_clauses = clauses.len();

        // --- Proof clause set and derivation info ---
        let proof_clauses = self.get_proof_clause_set();

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

        // --- Graph data via GraphBuilder::build_from_clauses ---
        let clause_refs: Vec<&Clause> = clauses.iter().collect();
        let batch_graph = GraphBuilder::build_from_clauses(&clause_refs);

        // node_features [total_nodes, 3]
        let node_feat_2d: Vec<Vec<f32>> = batch_graph.node_features.iter()
            .map(|nf| nf.to_vec())
            .collect();
        let node_features = PyArray2::from_vec2(py, &node_feat_2d)
            .map_err(|e| PyValueError::new_err(format!("Failed to create node_features: {}", e)))?;

        // edge_src, edge_dst [total_edges]
        let total_edges = batch_graph.edge_indices.len();
        let mut edge_src = Vec::with_capacity(total_edges);
        let mut edge_dst = Vec::with_capacity(total_edges);
        for &(s, d) in &batch_graph.edge_indices {
            edge_src.push(s as i32);
            edge_dst.push(d as i32);
        }
        let edge_src_arr = PyArray1::from_vec(py, edge_src);
        let edge_dst_arr = PyArray1::from_vec(py, edge_dst);

        // node_offsets [num_clauses + 1] and edge_offsets [num_clauses + 1]
        let mut node_offsets = Vec::with_capacity(num_clauses + 1);
        let mut edge_offsets = Vec::with_capacity(num_clauses + 1);
        node_offsets.push(0i64);
        edge_offsets.push(0i64);
        for i in 0..num_clauses {
            node_offsets.push(batch_graph.clause_boundaries[i].1 as i64);
            edge_offsets.push(batch_graph.edge_boundaries[i].1 as i64);
        }
        let node_offsets_arr = PyArray1::from_vec(py, node_offsets);
        let edge_offsets_arr = PyArray1::from_vec(py, edge_offsets);

        // --- Clause features [num_clauses, 9] ---
        let mut clause_features_2d = Vec::with_capacity(num_clauses);
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
                _ => 0.0, // Unknown rules default to input
            };

            // Compute term statistics (reusing logic from json.rs::term_stats)
            let mut depth: usize = 0;
            let mut symbol_count: usize = 0;
            let mut variable_count: usize = 0;
            let mut distinct_symbols: HashSet<u64> = HashSet::new();
            let mut distinct_variables: HashSet<u64> = HashSet::new();

            for lit in &clause.literals {
                symbol_count += 1;
                distinct_symbols.insert(lit.predicate.id.0 as u64 | (1u64 << 32));
                for arg in &lit.args {
                    let (d, sc, vc) = Self::term_stats(arg, &mut distinct_symbols, &mut distinct_variables);
                    depth = depth.max(d);
                    symbol_count += sc;
                    variable_count += vc;
                }
            }

            clause_features_2d.push(vec![
                clause.age as f32,                  // [0] age
                clause.role.to_feature_value(),     // [1] role
                rule_id,                            // [2] rule
                clause.literals.len() as f32,       // [3] size
                depth as f32,                       // [4] depth
                symbol_count as f32,                // [5] symbol_count
                distinct_symbols.len() as f32,      // [6] distinct_symbols
                variable_count as f32,              // [7] variable_count
                distinct_variables.len() as f32,    // [8] distinct_vars
            ]);
        }
        let clause_features = PyArray2::from_vec2(py, &clause_features_2d)
            .map_err(|e| PyValueError::new_err(format!("Failed to create clause_features: {}", e)))?;

        // --- Labels [num_clauses] ---
        let labels: Vec<u8> = (0..num_clauses)
            .map(|i| if proof_clauses.contains(&i) { 1 } else { 0 })
            .collect();
        let labels_arr = PyArray1::from_vec(py, labels);

        // --- Selection states (offset-encoded) ---
        let states = self.build_selection_states_raw();
        let num_states = states.len();

        let mut state_selected = Vec::with_capacity(num_states);
        let mut state_u_offsets = Vec::with_capacity(num_states + 1);
        let mut state_u_indices = Vec::new();
        let mut state_p_offsets = Vec::with_capacity(num_states + 1);
        let mut state_p_indices = Vec::new();

        state_u_offsets.push(0i64);
        state_p_offsets.push(0i64);

        for (selected, u_set, p_set) in &states {
            state_selected.push(*selected as i32);
            state_u_indices.extend(u_set.iter().map(|&x| x as i32));
            state_u_offsets.push(state_u_indices.len() as i64);
            state_p_indices.extend(p_set.iter().map(|&x| x as i32));
            state_p_offsets.push(state_p_indices.len() as i64);
        }

        let state_selected_arr = PyArray1::from_vec(py, state_selected);
        let state_u_offsets_arr = PyArray1::from_vec(py, state_u_offsets);
        let state_u_indices_arr = PyArray1::from_vec(py, state_u_indices);
        let state_p_offsets_arr = PyArray1::from_vec(py, state_p_offsets);
        let state_p_indices_arr = PyArray1::from_vec(py, state_p_indices);

        // --- Metadata ---
        let proof_found = self.empty_clause_idx().is_some();
        let proof_found_arr = PyArray1::from_vec(py, vec![proof_found]);
        let time_arr = PyArray1::from_vec(py, vec![time_seconds]);

        // --- Clause strings for sentence trace ---
        let clause_strings: Vec<String> = clauses.iter()
            .map(|c| c.display(interner).to_string())
            .collect();

        // --- Build graph dict ---
        let graph_dict = PyDict::new(py);
        graph_dict.set_item("node_features", node_features)?;
        graph_dict.set_item("edge_src", edge_src_arr)?;
        graph_dict.set_item("edge_dst", edge_dst_arr)?;
        graph_dict.set_item("node_offsets", node_offsets_arr)?;
        graph_dict.set_item("edge_offsets", edge_offsets_arr)?;
        graph_dict.set_item("clause_features", &clause_features)?;
        graph_dict.set_item("labels", &labels_arr)?;
        graph_dict.set_item("state_selected", &state_selected_arr)?;
        graph_dict.set_item("state_u_offsets", &state_u_offsets_arr)?;
        graph_dict.set_item("state_u_indices", &state_u_indices_arr)?;
        graph_dict.set_item("state_p_offsets", &state_p_offsets_arr)?;
        graph_dict.set_item("state_p_indices", &state_p_indices_arr)?;
        graph_dict.set_item("proof_found", &proof_found_arr)?;
        graph_dict.set_item("time_seconds", &time_arr)?;

        // --- Build sentence dict ---
        let sentence_dict = PyDict::new(py);
        // clause_strings as Python list of strings
        let py_strings = pyo3::types::PyList::new(py, &clause_strings)
            .map_err(|e| PyValueError::new_err(format!("Failed to create clause_strings: {}", e)))?;
        sentence_dict.set_item("clause_strings", py_strings)?;
        sentence_dict.set_item("clause_features", clause_features)?;
        sentence_dict.set_item("labels", labels_arr)?;
        sentence_dict.set_item("state_selected", state_selected_arr)?;
        sentence_dict.set_item("state_u_offsets", state_u_offsets_arr)?;
        sentence_dict.set_item("state_u_indices", state_u_indices_arr)?;
        sentence_dict.set_item("state_p_offsets", state_p_offsets_arr)?;
        sentence_dict.set_item("state_p_indices", state_p_indices_arr)?;
        sentence_dict.set_item("proof_found", proof_found_arr)?;
        sentence_dict.set_item("time_seconds", time_arr)?;

        Ok((graph_dict, sentence_dict))
    }
}

impl PyProofAtlas {
    /// Get the set of clause indices in the proof DAG
    fn get_proof_clause_set(&self) -> HashSet<usize> {
        let empty_id = match self.empty_clause_idx() {
            Some(idx) => idx,
            None => return HashSet::new(),
        };

        let events = self.event_log();

        // Build derivation map from event log
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

    /// Replay event log to build selection state snapshots (U/P at each Activate)
    /// Returns raw tuples: (selected, u_indices, p_indices)
    fn build_selection_states_raw(&self) -> Vec<(usize, Vec<usize>, Vec<usize>)> {
        use std::collections::BTreeSet;

        let events = self.event_log();
        let mut n: BTreeSet<usize> = BTreeSet::new();
        let mut u: BTreeSet<usize> = BTreeSet::new();
        let mut p: BTreeSet<usize> = BTreeSet::new();
        let mut states = Vec::new();

        for event in events {
            match event {
                RustStateChange::Add(clause, _, _) => {
                    if let Some(idx) = clause.id {
                        n.insert(idx);
                    }
                }
                RustStateChange::Simplify(clause_idx, replacement, _, _) => {
                    n.remove(clause_idx);
                    u.remove(clause_idx);
                    p.remove(clause_idx);
                    if let Some(clause) = replacement {
                        if let Some(idx) = clause.id {
                            n.insert(idx);
                        }
                    }
                }
                RustStateChange::Transfer(clause_idx) => {
                    n.remove(clause_idx);
                    u.insert(*clause_idx);
                }
                RustStateChange::Activate(clause_idx) => {
                    // Snapshot U and P before moving the selected clause
                    states.push((
                        *clause_idx,
                        u.iter().copied().collect(),
                        p.iter().copied().collect(),
                    ));
                    u.remove(clause_idx);
                    p.insert(*clause_idx);
                }
            }
        }

        states
    }

    /// Recursively compute term statistics.
    /// Returns (depth, symbol_count, variable_count).
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
                    let (d, s, v) = Self::term_stats(arg, distinct_symbols, distinct_variables);
                    max_depth = max_depth.max(d);
                    sc += s;
                    vc += v;
                }
                (max_depth + 1, sc, vc)
            }
        }
    }
}

/// Start a scoring server that blocks until the process is terminated.
///
/// Intended to be called from a subprocess (e.g., by bench.py).
/// The server listens on the given Unix socket and serves scoring requests
/// from worker processes.
///
/// Args:
///     encoder: Encoder type ("gcn", "gat", "graphsage", or "sentence")
///     scorer: Scorer name (used to locate model file as "{encoder}_{scorer}.pt")
///     weights_path: Path to weights directory
///     socket_path: Path for the Unix domain socket
///     use_cuda: Whether to use CUDA (default: false)
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

    // This blocks until the process is terminated
    server.run();
    Ok(())
}

#[pymodule]
fn proofatlas(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProofAtlas>()?;
    m.add_class::<ProofStep>()?;
    #[cfg(feature = "ml")]
    m.add_function(wrap_pyfunction!(start_scoring_server, m)?)?;
    Ok(())
}
