//! Python bindings for ProofAtlas using PyO3

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

#[cfg(feature = "python")]
use numpy::{PyArray1, PyArray2, ToPyArray};

use crate::logic::{Clause, Interner};
use crate::generating::{
    equality_factoring, equality_resolution, factoring, resolution, superposition,
};
use crate::state::{clause_indices, StateChange as RustStateChange};
use crate::selection::graph::{ClauseGraph, GraphBuilder};
use crate::parser::parse_tptp;
use crate::config::LiteralSelectionStrategy;
use crate::selection::{LiteralSelector, SelectAll, SelectMaximal};

/// Python-accessible proof state
#[pyclass]
pub struct ProofState {
    /// Symbol interner for resolving symbol names
    interner: Interner,
    /// All clauses (processed and unprocessed)
    clauses: Vec<Clause>,
    /// Indices of processed clauses
    processed: HashSet<usize>,
    /// Queue of unprocessed clause indices
    unprocessed: VecDeque<usize>,
    /// Literal selector
    literal_selector: Box<dyn LiteralSelector + Send>,
    /// Proof trace
    proof_trace: Vec<ProofStep>,
}

/// Information about a clause (lightweight)
#[pyclass]
#[derive(Clone)]
pub struct ClauseInfo {
    #[pyo3(get)]
    pub clause_id: usize,
    #[pyo3(get)]
    pub clause_string: String,
    #[pyo3(get)]
    pub num_literals: usize,
    #[pyo3(get)]
    pub literal_strings: Vec<String>,
    #[pyo3(get)]
    pub is_unit: bool,
    #[pyo3(get)]
    pub is_horn: bool,
    #[pyo3(get)]
    pub is_equality: bool,
    #[pyo3(get)]
    pub weight: usize,
    #[pyo3(get)]
    pub variables: Vec<String>,
    #[pyo3(get)]
    pub age: usize,
    #[pyo3(get)]
    pub role: String,
    #[pyo3(get)]
    pub is_goal: bool,
}

/// Result of an inference
#[pyclass]
#[derive(Clone)]
pub struct InferenceResult {
    #[pyo3(get)]
    pub clause_string: String,
    #[pyo3(get)]
    pub parent_ids: Vec<usize>,
    #[pyo3(get)]
    pub rule_name: String,
    /// Internal: the actual clause for adding to state
    pub(crate) clause: Clause,
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

/// Graph representation of a clause (Python-accessible)
#[pyclass]
pub struct ClauseGraphData {
    graph: ClauseGraph,
}

#[cfg(feature = "python")]
#[pymethods]
impl ClauseGraphData {
    /// Get edge indices as numpy array (2, num_edges)
    fn edge_indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        let num_edges = self.graph.edge_indices.len();

        // Build separate source and target arrays
        let mut sources = Vec::with_capacity(num_edges);
        let mut targets = Vec::with_capacity(num_edges);

        for (src, tgt) in &self.graph.edge_indices {
            sources.push(*src as i64);
            targets.push(*tgt as i64);
        }

        // Create 2D array with shape (2, num_edges)
        PyArray2::from_vec2(py, &[sources, targets]).unwrap()
    }

    /// Get node features as numpy array (num_nodes, feature_dim)
    fn node_features<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        // Convert Vec<[f32; FEATURE_DIM]> to Vec<Vec<f32>>
        let features_vec: Vec<Vec<f32>> = self.graph.node_features
            .iter()
            .map(|arr| arr.to_vec())
            .collect();

        PyArray2::from_vec2(py, &features_vec).unwrap()
    }

    /// Get node types as numpy array (num_nodes,)
    fn node_types<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        self.graph.node_types.to_pyarray(py)
    }

    /// Number of nodes
    fn num_nodes(&self) -> usize {
        self.graph.num_nodes
    }

    /// Number of edges
    fn num_edges(&self) -> usize {
        self.graph.edge_indices.len()
    }

    /// Feature dimension
    fn feature_dim(&self) -> usize {
        if self.graph.node_features.is_empty() {
            0
        } else {
            self.graph.node_features[0].len()
        }
    }

    /// Get node names for debugging
    fn node_names(&self) -> Vec<String> {
        self.graph.node_names.clone()
    }
}

#[pymethods]
impl ProofState {
    /// Create empty proof state
    #[new]
    pub fn new() -> Self {
        ProofState {
            interner: Interner::new(),
            clauses: Vec::new(),
            processed: HashSet::new(),
            unprocessed: VecDeque::new(),
            literal_selector: Box::new(SelectAll) as Box<dyn LiteralSelector + Send>,
            proof_trace: Vec::new(),
        }
    }

    /// Parse TPTP content and add clauses, return clause IDs
    ///
    /// Args:
    ///     content: TPTP file content as string
    ///     include_dir: Optional directory to search for included files (e.g., TPTP root)
    ///     timeout: Optional timeout in seconds for CNF conversion (prevents hangs on complex formulas)
    #[pyo3(signature = (content, include_dir=None, timeout=None))]
    pub fn add_clauses_from_tptp(
        &mut self,
        content: &str,
        include_dir: Option<&str>,
        timeout: Option<f64>,
    ) -> PyResult<Vec<usize>> {
        let timeout_instant = timeout.map(|t| Instant::now() + Duration::from_secs_f64(t));
        let include_dirs: Vec<String> = include_dir.into_iter().map(|s| s.to_string()).collect();
        let content_owned = content.to_string();

        // Run parsing in a thread with larger stack to handle deeply nested formulas
        // Default Python thread stack is too small for formulas with depth > 2000
        let parsed = std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)  // 128MB stack
            .spawn(move || {
                let include_refs: Vec<&str> = include_dirs.iter().map(|s| s.as_str()).collect();
                parse_tptp(&content_owned, &include_refs, timeout_instant)
            })
            .map_err(|e| PyValueError::new_err(format!("Failed to spawn parser thread: {}", e)))?
            .join()
            .map_err(|_| PyValueError::new_err("Parser thread panicked"))?
            .map_err(|e| PyValueError::new_err(format!("Parse error: {}", e)))?;

        // Extract interner and formula from parsed problem
        self.interner = parsed.interner;
        let cnf = parsed.formula;

        let mut ids = Vec::new();
        for mut clause in cnf.clauses {
            let id = self.clauses.len();
            clause.id = Some(id);
            ids.push(id);
            self.clauses.push(clause);
            self.unprocessed.push_back(id);
        }

        Ok(ids)
    }

    /// Total number of clauses
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Number of processed clauses
    pub fn num_processed(&self) -> usize {
        self.processed.len()
    }

    /// Number of unprocessed clauses
    pub fn num_unprocessed(&self) -> usize {
        self.unprocessed.len()
    }

    /// Check if proof found (contains empty clause)
    pub fn contains_empty_clause(&self) -> bool {
        self.clauses.iter().any(|c| c.is_empty())
    }

    /// Get clause as formatted string
    pub fn clause_to_string(&self, clause_id: usize) -> PyResult<String> {
        self.clauses
            .get(clause_id)
            .map(|c| c.display(&self.interner).to_string())
            .ok_or_else(|| PyValueError::new_err(format!("Invalid clause ID: {}", clause_id)))
    }

    /// Get basic clause information
    pub fn get_clause_info(&self, clause_id: usize) -> PyResult<ClauseInfo> {
        let clause = self
            .clauses
            .get(clause_id)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid clause ID: {}", clause_id)))?;

        // Extract literal strings
        let literal_strings: Vec<String> =
            clause.literals.iter().map(|lit| lit.display(&self.interner).to_string()).collect();

        // Extract variables
        let mut variables = HashSet::new();
        for lit in &clause.literals {
            lit.collect_variables(&mut variables);
        }
        let variables: Vec<String> = variables.into_iter().map(|v| v.name(&self.interner).to_string()).collect();

        // Count symbols for weight
        let weight = clause.symbol_count();

        // Check properties
        let is_unit = clause.literals.len() == 1;
        let is_horn = clause.literals.iter().filter(|l| l.polarity).count() <= 1;
        let is_equality = clause.literals.iter().any(|l| l.is_equality(&self.interner));

        // Get role as string
        let role = match clause.role {
            crate::logic::ClauseRole::Axiom => "axiom",
            crate::logic::ClauseRole::Hypothesis => "hypothesis",
            crate::logic::ClauseRole::Definition => "definition",
            crate::logic::ClauseRole::NegatedConjecture => "negated_conjecture",
            crate::logic::ClauseRole::Derived => "derived",
        }
        .to_string();

        Ok(ClauseInfo {
            clause_id,
            clause_string: clause.display(&self.interner).to_string(),
            num_literals: clause.literals.len(),
            literal_strings,
            is_unit,
            is_horn,
            is_equality,
            weight,
            variables,
            age: clause.age,
            role,
            is_goal: clause.role.is_goal(),
        })
    }

    /// Get literals as strings
    pub fn get_literals_as_strings(&self, clause_id: usize) -> PyResult<Vec<String>> {
        let clause = self
            .clauses
            .get(clause_id)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid clause ID: {}", clause_id)))?;

        Ok(clause.literals.iter().map(|lit| lit.display(&self.interner).to_string()).collect())
    }

    /// Get parent clause IDs and inference rule name
    pub fn get_clause_parents(&self, clause_id: usize) -> PyResult<(Vec<usize>, String)> {
        // Find in proof trace
        for step in &self.proof_trace {
            if step.clause_id == clause_id {
                return Ok((step.parent_ids.clone(), step.rule_name.clone()));
            }
        }

        // Initial clause has no parents
        Ok((Vec::new(), "input".to_string()))
    }

    /// Select next given clause
    #[pyo3(signature = (strategy=None))]
    pub fn select_given_clause(
        &mut self,
        strategy: Option<&str>,
    ) -> PyResult<Option<usize>> {
        let strategy = strategy.unwrap_or("age");

        match strategy {
            "age" | "fifo" => Ok(self.unprocessed.pop_front()), // Accept both for backward compatibility
            "smallest" => {
                // Find smallest clause
                let mut best_idx = None;
                let mut best_size = usize::MAX;
                let mut best_pos = None;

                for (pos, &idx) in self.unprocessed.iter().enumerate() {
                    let size = self.clauses[idx].literals.len();
                    if size < best_size {
                        best_size = size;
                        best_idx = Some(idx);
                        best_pos = Some(pos);
                    }
                }

                if let Some(pos) = best_pos {
                    self.unprocessed.remove(pos);
                    Ok(best_idx)
                } else {
                    Ok(None)
                }
            }
            "size" => {
                // Alias for "smallest"
                self.select_given_clause(Some("smallest"))
            }
            _ => Ok(self.unprocessed.pop_front()), // Default to age-based
        }
    }

    /// Generate inferences with given clause
    pub fn generate_inferences(&mut self, given_id: usize) -> PyResult<Vec<InferenceResult>> {
        let given_clause = self
            .clauses
            .get(given_id)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid clause ID: {}", given_id)))?
            .clone();

        let mut results = Vec::new();
        let selector = self.literal_selector.as_ref();

        // Self-inferences on given clause
        for rust_result in factoring(&given_clause, given_id, selector) {
            results.push(self.convert_state_change(rust_result).unwrap());
        }

        for rust_result in equality_resolution(&given_clause, given_id, selector, &self.interner) {
            results.push(self.convert_state_change(rust_result).unwrap());
        }

        for rust_result in equality_factoring(&given_clause, given_id, selector, &mut self.interner) {
            results.push(self.convert_state_change(rust_result).unwrap());
        }

        // Binary inferences with processed clauses
        let processed_ids: Vec<usize> = self.processed.iter().copied().collect();
        for processed_id in processed_ids {
            let processed_clause = self.clauses[processed_id].clone();

            // Resolution in both directions
            for rust_result in resolution(
                &given_clause,
                &processed_clause,
                given_id,
                processed_id,
                selector,
                &mut self.interner,
            ) {
                results.push(self.convert_state_change(rust_result).unwrap());
            }
            for rust_result in resolution(
                &processed_clause,
                &given_clause,
                processed_id,
                given_id,
                selector,
                &mut self.interner,
            ) {
                results.push(self.convert_state_change(rust_result).unwrap());
            }

            // Superposition
            for rust_result in superposition(
                &given_clause,
                &processed_clause,
                given_id,
                processed_id,
                selector,
                &mut self.interner,
            ) {
                results.push(self.convert_state_change(rust_result).unwrap());
            }
            for rust_result in superposition(
                &processed_clause,
                &given_clause,
                processed_id,
                given_id,
                selector,
                &mut self.interner,
            ) {
                results.push(self.convert_state_change(rust_result).unwrap());
            }
        }

        Ok(results)
    }

    /// Add inference result if not redundant, return new clause ID
    pub fn add_inference(&mut self, inference: InferenceResult) -> PyResult<Option<usize>> {
        let clause = inference.clause;

        // Check if tautology
        if clause.is_tautology(&self.interner) {
            return Ok(None);
        }

        // Simple duplicate check (could be improved with subsumption)
        for existing in &self.clauses {
            if existing.literals == clause.literals {
                return Ok(None);
            }
        }

        // Add the clause
        let new_id = self.clauses.len();
        let mut clause_with_id = clause;
        clause_with_id.id = Some(new_id);

        self.clauses.push(clause_with_id);
        self.unprocessed.push_back(new_id);

        // Record in proof trace
        self.proof_trace.push(ProofStep {
            clause_id: new_id,
            clause_string: inference.clause_string,
            parent_ids: inference.parent_ids,
            rule_name: inference.rule_name,
        });

        Ok(Some(new_id))
    }

    /// Move clause from unprocessed to processed
    pub fn process_clause(&mut self, clause_id: usize) -> PyResult<()> {
        // Remove from unprocessed if present (may already be removed by select_given_clause)
        if let Some(pos) = self.unprocessed.iter().position(|&id| id == clause_id) {
            self.unprocessed.remove(pos);
        }

        // Add to processed
        self.processed.insert(clause_id);
        Ok(())
    }

    /// Set literal selection strategy
    pub fn set_literal_selection(&mut self, strategy: &str) -> PyResult<()> {
        match strategy {
            "all" | "select_all" | "0" => {
                self.literal_selector = Box::new(SelectAll) as Box<dyn LiteralSelector + Send>;
                Ok(())
            }
            "maximal" | "20" => {
                self.literal_selector =
                    Box::new(SelectMaximal::new()) as Box<dyn LiteralSelector + Send>;
                Ok(())
            }
            "unique" | "21" => {
                self.literal_selector =
                    Box::new(crate::selection::SelectUniqueMaximalOrNegOrMaximal::new()) as Box<dyn LiteralSelector + Send>;
                Ok(())
            }
            "neg_max_weight" | "22" => {
                self.literal_selector =
                    Box::new(crate::selection::SelectNegMaxWeightOrMaximal::new()) as Box<dyn LiteralSelector + Send>;
                Ok(())
            }
            _ => Err(PyValueError::new_err(format!(
                "Unknown literal selection: {}. Use 0/20/21/22 or all/maximal/unique/neg_max_weight",
                strategy
            ))),
        }
    }

    /// Run full saturation using the Rust saturation engine
    ///
    /// This is more efficient than calling generate_inferences/add_inference in a loop
    /// because it uses demodulation and other optimizations.
    ///
    /// Args:
    ///     timeout: Optional timeout in seconds
    ///     max_iterations: Maximum number of saturation steps (0 or None = no limit)
    ///     literal_selection: Literal selection strategy: 0/20/21/22 (default: 0)
    ///     age_weight_ratio: Age probability for age-weight clause selector (default: 0.5)
    ///     encoder: Encoder name: None (default, uses age_weight), "gcn", "gat", "graphsage", or "sentence"
    ///     scorer: Name of the scorer. Model file is "{encoder}_{scorer}.pt" (e.g., "gcn_mlp").
    ///     weights_path: Path to model weights directory
    ///     max_clause_memory_mb: Clause memory limit in MB (directly comparable across provers)
    ///     enable_profiling: Enable structured profiling (default: false).
    ///                       When enabled, the third element of the return tuple is a JSON string.
    ///
    /// Returns:
    ///     Tuple of (proof_found: bool, status: str, profile_json: Optional[str], trace_json: Optional[str])
    #[pyo3(signature = (timeout=None, max_iterations=None, literal_selection=None, age_weight_ratio=None, encoder=None, scorer=None, weights_path=None, max_clause_memory_mb=None, use_cuda=None, enable_profiling=None))]
    pub fn run_saturation(
        &mut self,
        timeout: Option<f64>,
        max_iterations: Option<usize>,
        literal_selection: Option<u32>,
        age_weight_ratio: Option<f64>,
        encoder: Option<String>,
        scorer: Option<String>,
        weights_path: Option<String>,
        max_clause_memory_mb: Option<usize>,
        use_cuda: Option<bool>,
        enable_profiling: Option<bool>,
    ) -> PyResult<(bool, String, Option<String>, Option<String>)> {
        use crate::prover::ProofAtlas;
        use crate::config::ProverConfig;
        use crate::state::ProofResult;
        use crate::selection::AgeWeightSelector;
        use std::time::Duration;

        // Create clause selector based on encoder type
        let model_name = match (encoder.as_deref(), scorer.as_deref()) {
            (Some(enc), Some(sc)) => Some(format!("{}_{}", enc, sc)),
            (Some(_), None) => return Err(PyValueError::new_err("scorer required when encoder is set")),
            _ => None,
        };

        let clause_selector: Box<dyn crate::selection::ClauseSelector> = match encoder.as_deref() {
            None => {
                // No encoder = heuristic selector
                let ratio = age_weight_ratio.unwrap_or(0.5);
                Box::new(AgeWeightSelector::new(ratio))
            }
            #[cfg(feature = "ml")]
            Some("gcn") | Some("gat") | Some("graphsage") => {
                // Graph encoders
                let weights_dir = if let Some(path) = weights_path.as_ref() {
                    std::path::PathBuf::from(path)
                } else {
                    std::path::PathBuf::from(".weights")
                };

                let name = model_name.as_deref().unwrap();
                let model_path = weights_dir.join(format!("{}.pt", name));

                if !model_path.exists() {
                    return Err(PyValueError::new_err(format!(
                        "Model not found at {}. Export with model.export_torchscript(path)",
                        model_path.display()
                    )));
                }

                let selector = crate::selection::load_gcn_selector(
                    &model_path,
                    use_cuda.unwrap_or(true),
                ).map_err(|e| PyValueError::new_err(format!("Failed to load model: {}", e)))?;
                Box::new(selector)
            }
            #[cfg(feature = "ml")]
            Some("sentence") => {
                // String encoder (sentence transformer)
                let weights_dir = if let Some(path) = weights_path.as_ref() {
                    std::path::PathBuf::from(path)
                } else {
                    std::path::PathBuf::from(".weights")
                };

                let name = model_name.as_deref().unwrap();
                let model_path = weights_dir.join(format!("{}.pt", name));
                let tokenizer_path = weights_dir.join(format!("{}_tokenizer/tokenizer.json", name));

                if !model_path.exists() {
                    return Err(PyValueError::new_err(format!(
                        "Model not found at {}. Export with model.export_torchscript(path)",
                        model_path.display()
                    )));
                }

                let selector = crate::selection::load_sentence_selector(
                    &model_path,
                    &tokenizer_path,
                    use_cuda.unwrap_or(true),
                ).map_err(|e| PyValueError::new_err(format!("Failed to load model: {}", e)))?;
                Box::new(selector)
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
            max_clause_memory_mb,
            enable_profiling: enable_profiling.unwrap_or(false),
        };

        // Create prover from current clauses
        let initial_clauses: Vec<Clause> = self.clauses.clone();
        let interner = self.interner.clone();
        let prover = ProofAtlas::new(initial_clauses, config, clause_selector, interner);

        // Run saturation in a thread with larger stack to handle deep recursion
        let (result, profile, sat_trace, returned_interner) = std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)  // 128MB stack
            .spawn(move || prover.prove())
            .map_err(|e| PyValueError::new_err(format!("Failed to spawn saturation thread: {}", e)))?
            .join()
            .map_err(|_| PyValueError::new_err("Saturation thread panicked (possible stack overflow)"))?;

        // Update interner with new symbols created during saturation
        self.interner = returned_interner;

        // Serialize profile to JSON if present
        let profile_json = profile
            .map(|p| serde_json::to_string(&p))
            .transpose()
            .map_err(|e| PyValueError::new_err(format!("Profile serialization failed: {}", e)))?;

        // Copy back the results
        let (proof_found, status, final_clauses, proof_steps) = match result {
            ProofResult::Proof(proof) => {
                (true, "proof", proof.all_clauses, proof.steps)
            }
            ProofResult::Saturated(steps, clauses) => (false, "saturated", clauses, steps),
            ProofResult::ResourceLimit(steps, clauses) => (false, "resource_limit", clauses, steps),
            ProofResult::Timeout(steps, clauses) => (false, "resource_limit", clauses, steps),
        };

        // Update our state with the results
        self.clauses = final_clauses;
        self.processed.clear();
        self.unprocessed.clear();

        // Rebuild processed/unprocessed from the proof steps
        for step in &proof_steps {
            self.processed.insert(step.clause_idx);
        }

        // Rebuild proof trace
        self.proof_trace.clear();
        for step in proof_steps {
            let clause_string = step.conclusion.display(&self.interner).to_string();
            self.proof_trace.push(ProofStep {
                clause_id: step.clause_idx,
                parent_ids: clause_indices(&step.premises),
                rule_name: step.rule_name.clone(),
                clause_string,
            });
        }

        // Serialize trace to JSON
        let trace_json = serde_json::to_string(&sat_trace)
            .map_err(|e| PyValueError::new_err(format!("Trace serialization failed: {}", e)))?;

        Ok((proof_found, status.to_string(), profile_json, Some(trace_json)))
    }

    /// Get statistics
    pub fn get_statistics(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total".to_string(), self.clauses.len());
        stats.insert("processed".to_string(), self.processed.len());
        stats.insert("unprocessed".to_string(), self.unprocessed.len());

        // Count empty clauses
        let empty_count = self.clauses.iter().filter(|c| c.is_empty()).count();
        stats.insert("empty_clauses".to_string(), empty_count);

        // Count unit clauses
        let unit_count = self
            .clauses
            .iter()
            .filter(|c| c.literals.len() == 1)
            .count();
        stats.insert("unit_clauses".to_string(), unit_count);

        stats
    }

    /// Get all proof steps from the last saturation run.
    /// Unlike get_proof_trace() which returns only the minimal proof path,
    /// this returns every step including GivenClauseSelection steps.
    pub fn get_all_steps(&self) -> Vec<ProofStep> {
        self.proof_trace.clone()
    }

    /// Get proof trace
    pub fn get_proof_trace(&self) -> Vec<ProofStep> {
        // Find empty clause
        let empty_clause_id = self.clauses.iter().position(|c| c.is_empty());

        if let Some(empty_id) = empty_clause_id {
            // Build proof trace backwards from empty clause
            let mut trace = Vec::new();
            let mut to_visit = vec![empty_id];
            let mut visited = HashSet::new();

            while let Some(current_id) = to_visit.pop() {
                if visited.contains(&current_id) {
                    continue;
                }
                visited.insert(current_id);

                // Find this clause in proof trace
                if let Some(step) = self.proof_trace.iter().find(|s| s.clause_id == current_id) {
                    trace.push(step.clone());
                    to_visit.extend(&step.parent_ids);
                } else {
                    // Input clause
                    trace.push(ProofStep {
                        clause_id: current_id,
                        clause_string: self.clauses[current_id].display(&self.interner).to_string(),
                        parent_ids: Vec::new(),
                        rule_name: "input".to_string(),
                    });
                }
            }

            // Reverse to get forward trace
            trace.reverse();
            trace
        } else {
            Vec::new()
        }
    }

    /// Convert clause to sparse graph representation
    #[cfg(feature = "python")]
    pub fn clause_to_graph(&self, clause_id: usize) -> PyResult<ClauseGraphData> {
        // Use current number of clauses as max_age for normalization
        let max_age = self.clauses.len().max(1);
        self.clause_to_graph_with_context(clause_id, max_age)
    }

    /// Convert clause to sparse graph with specific max_age for normalization
    #[cfg(feature = "python")]
    pub fn clause_to_graph_with_context(
        &self,
        clause_id: usize,
        max_age: usize,
    ) -> PyResult<ClauseGraphData> {
        let clause = self
            .clauses
            .get(clause_id)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid clause ID: {}", clause_id)))?;

        let graph = GraphBuilder::build_from_clause_with_context(clause, max_age, &self.interner);

        Ok(ClauseGraphData { graph })
    }

    /// Convert multiple clauses to batch of graphs
    #[cfg(feature = "python")]
    pub fn clauses_to_graphs(&self, clause_ids: Vec<usize>) -> PyResult<Vec<ClauseGraphData>> {
        // Use current number of clauses as max_age for normalization
        let max_age = self.clauses.len().max(1);

        let mut graphs = Vec::new();
        for clause_id in clause_ids {
            let graph_data = self.clause_to_graph_with_context(clause_id, max_age)?;
            graphs.push(graph_data);
        }

        Ok(graphs)
    }

    /// Extract training examples: all clauses labeled by whether they're in the proof
    /// Returns list of (clause_id, label) where label=1 if in proof, 0 otherwise
    pub fn extract_training_examples(&self) -> Vec<TrainingExample> {
        // Find empty clause
        let empty_clause_id = self.clauses.iter().position(|c| c.is_empty());

        if empty_clause_id.is_none() {
            return Vec::new(); // No proof found
        }

        let empty_id = empty_clause_id.unwrap();

        // Build set of clauses in proof DAG
        let mut proof_clauses = HashSet::new();
        let mut to_visit = vec![empty_id];

        while let Some(current_id) = to_visit.pop() {
            if proof_clauses.contains(&current_id) {
                continue;
            }
            proof_clauses.insert(current_id);

            // Find parents in proof trace
            if let Some(step) = self.proof_trace.iter().find(|s| s.clause_id == current_id) {
                to_visit.extend(&step.parent_ids);
            }
        }

        // Create training examples for all clauses
        let mut examples = Vec::new();
        for clause_id in 0..self.clauses.len() {
            let label = if proof_clauses.contains(&clause_id) { 1 } else { 0 };
            examples.push(TrainingExample { clause_idx: clause_id, label });
        }

        examples
    }

    /// Get all clause IDs
    pub fn all_clause_ids(&self) -> Vec<usize> {
        (0..self.clauses.len()).collect()
    }

    /// Get clause IDs that are in the proof (if proof found)
    pub fn proof_clause_ids(&self) -> Vec<usize> {
        self.extract_training_examples()
            .into_iter()
            .filter(|e| e.label == 1)
            .map(|e| e.clause_idx)
            .collect()
    }

    /// Get proof statistics
    pub fn get_proof_statistics(&self) -> HashMap<String, usize> {
        let examples = self.extract_training_examples();
        let mut stats = HashMap::new();

        let total = examples.len();
        let in_proof = examples.iter().filter(|e| e.label == 1).count();
        let not_in_proof = total - in_proof;

        stats.insert("total_clauses".to_string(), total);
        stats.insert("proof_clauses".to_string(), in_proof);
        stats.insert("non_proof_clauses".to_string(), not_in_proof);

        if total > 0 {
            stats.insert("proof_percentage".to_string(), (in_proof * 100) / total);
        }

        stats
    }

    /// Extract training data in structured JSON format (model-independent)
    ///
    /// Returns a JSON string with the trace data that can be converted
    /// to graphs or strings at training time.
    ///
    /// Format:
    /// ```json
    /// {
    ///   "proof_found": true,
    ///   "time_seconds": 1.23,
    ///   "clauses": [
    ///     {
    ///       "literals": [{"polarity": true, "atom": {"predicate": "=", "args": [...]}}],
    ///       "label": 1,
    ///       "age": 0,
    ///       "role": "axiom"
    ///     }
    ///   ]
    /// }
    /// ```
    pub fn extract_structured_trace(&self, time_seconds: f64) -> PyResult<String> {
        use crate::json::{TraceJson, TrainingClauseJson};
        use std::collections::HashMap;

        let examples = self.extract_training_examples();

        // Build proof clause set for labeling
        let proof_clauses: HashSet<usize> = examples
            .iter()
            .filter(|e| e.label == 1)
            .map(|e| e.clause_idx)
            .collect();

        // Build derivation info map from proof trace
        // All proof trace entries are real derivations (no trace-only events to filter)
        let mut derivation_info: HashMap<usize, (Vec<usize>, String)> = HashMap::new();
        for step in &self.proof_trace {
            derivation_info.insert(
                step.clause_id,
                (step.parent_ids.clone(), step.rule_name.clone()),
            );
        }

        // Build structured clauses with derivation info
        let clauses: Vec<TrainingClauseJson> = self
            .clauses
            .iter()
            .enumerate()
            .map(|(idx, clause)| {
                let in_proof = proof_clauses.contains(&idx);
                let (parents, rule) = derivation_info
                    .get(&idx)
                    .cloned()
                    .unwrap_or_else(|| (vec![], String::new()));
                TrainingClauseJson::from_clause(clause, &self.interner, in_proof, parents, rule)
            })
            .collect();

        let trace = TraceJson {
            proof_found: self.contains_empty_clause(),
            time_seconds,
            clauses,
        };

        serde_json::to_string(&trace)
            .map_err(|e| PyValueError::new_err(format!("JSON serialization failed: {}", e)))
    }
}

impl ProofState {
    /// Convert Rust StateChange::Add to Python inference result
    fn convert_state_change(&self, change: RustStateChange) -> Option<InferenceResult> {
        if let RustStateChange::Add(clause, rule_name, premises) = change {
            let clause_string = clause.display(&self.interner).to_string();
            let parent_ids = clause_indices(&premises);
            Some(InferenceResult {
                clause_string,
                parent_ids,
                rule_name: rule_name.to_lowercase(),
                clause,
            })
        } else {
            None
        }
    }
}

/// Training example for ML
#[pyclass]
#[derive(Clone)]
pub struct TrainingExample {
    #[pyo3(get)]
    pub clause_idx: usize,
    #[pyo3(get)]
    pub label: u8,  // 1 = in proof, 0 = not in proof
}

/// Python module definition
#[pymodule]
fn proofatlas(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ProofState>()?;
    m.add_class::<ClauseInfo>()?;
    m.add_class::<InferenceResult>()?;
    m.add_class::<ProofStep>()?;
    #[cfg(feature = "python")]
    m.add_class::<ClauseGraphData>()?;
    m.add_class::<TrainingExample>()?;
    Ok(())
}
