//! Python bindings for ProofAtlas using PyO3

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};

#[cfg(feature = "python")]
use numpy::{PyArray1, PyArray2, ToPyArray};

use crate::core::{CNFFormula, Clause, Term, Variable};
use crate::inference::{
    equality_factoring, equality_resolution, factoring, resolution, superposition,
    InferenceResult as RustInferenceResult, InferenceRule,
};
use crate::ml::{ClauseGraph, GraphBuilder};
use crate::parser::parse_tptp;
use crate::saturation::{LiteralSelectionStrategy, SaturationConfig};
use crate::selection::{LiteralSelector, SelectAll, SelectMaxWeight};

/// Python-accessible proof state
#[pyclass]
pub struct ProofState {
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
    fn edge_indices<'py>(&self, py: Python<'py>) -> &'py PyArray2<i64> {
        let num_edges = self.graph.edge_indices.len();

        // Build separate source and target arrays
        let mut sources = Vec::with_capacity(num_edges);
        let mut targets = Vec::with_capacity(num_edges);

        for (src, tgt) in &self.graph.edge_indices {
            sources.push(*src as i64);
            targets.push(*tgt as i64);
        }

        // Create 2D array with shape (2, num_edges)
        PyArray2::from_vec2(py, &vec![sources, targets]).unwrap()
    }

    /// Get node features as numpy array (num_nodes, feature_dim)
    fn node_features<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        // Convert Vec<[f32; FEATURE_DIM]> to Vec<Vec<f32>>
        let features_vec: Vec<Vec<f32>> = self.graph.node_features
            .iter()
            .map(|arr| arr.to_vec())
            .collect();

        PyArray2::from_vec2(py, &features_vec).unwrap()
    }

    /// Get node types as numpy array (num_nodes,)
    fn node_types<'py>(&self, py: Python<'py>) -> &'py PyArray1<u8> {
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
            clauses: Vec::new(),
            processed: HashSet::new(),
            unprocessed: VecDeque::new(),
            literal_selector: Box::new(SelectAll) as Box<dyn LiteralSelector + Send>,
            proof_trace: Vec::new(),
        }
    }

    /// Parse TPTP content and add clauses, return clause IDs
    pub fn add_clauses_from_tptp(&mut self, content: &str) -> PyResult<Vec<usize>> {
        let cnf = parse_tptp(content)
            .map_err(|e| PyValueError::new_err(format!("Parse error: {}", e)))?;

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
            .map(|c| c.to_string())
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
            clause.literals.iter().map(|lit| lit.to_string()).collect();

        // Extract variables
        let mut variables = HashSet::new();
        for lit in &clause.literals {
            lit.collect_variables(&mut variables);
        }
        let variables: Vec<String> = variables.into_iter().map(|v| v.name.clone()).collect();

        // Count symbols for weight
        let weight = clause.symbol_count();

        // Check properties
        let is_unit = clause.literals.len() == 1;
        let is_horn = clause.literals.iter().filter(|l| l.polarity).count() <= 1;
        let is_equality = clause.literals.iter().any(|l| l.atom.is_equality());

        // Get role as string
        let role = match clause.role {
            crate::core::ClauseRole::Axiom => "axiom",
            crate::core::ClauseRole::Hypothesis => "hypothesis",
            crate::core::ClauseRole::Definition => "definition",
            crate::core::ClauseRole::NegatedConjecture => "negated_conjecture",
            crate::core::ClauseRole::Derived => "derived",
        }
        .to_string();

        Ok(ClauseInfo {
            clause_id,
            clause_string: clause.to_string(),
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

        Ok(clause.literals.iter().map(|lit| lit.to_string()).collect())
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
    pub fn select_given_clause(
        &mut self,
        _py: Python,
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
                self.select_given_clause(_py, Some("smallest"))
            }
            _ => Ok(self.unprocessed.pop_front()), // Default to age-based
        }
    }

    /// Generate inferences with given clause
    pub fn generate_inferences(&self, given_id: usize) -> PyResult<Vec<InferenceResult>> {
        let given_clause = self
            .clauses
            .get(given_id)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid clause ID: {}", given_id)))?;

        let mut results = Vec::new();
        let selector = self.literal_selector.as_ref();

        // Self-inferences on given clause
        for rust_result in factoring(given_clause, given_id, selector) {
            results.push(self.convert_inference_result(rust_result));
        }

        for rust_result in equality_resolution(given_clause, given_id, selector) {
            results.push(self.convert_inference_result(rust_result));
        }

        for rust_result in equality_factoring(given_clause, given_id, selector) {
            results.push(self.convert_inference_result(rust_result));
        }

        // Binary inferences with processed clauses
        for &processed_id in &self.processed {
            let processed_clause = &self.clauses[processed_id];

            // Resolution in both directions
            for rust_result in resolution(
                given_clause,
                processed_clause,
                given_id,
                processed_id,
                selector,
            ) {
                results.push(self.convert_inference_result(rust_result));
            }
            for rust_result in resolution(
                processed_clause,
                given_clause,
                processed_id,
                given_id,
                selector,
            ) {
                results.push(self.convert_inference_result(rust_result));
            }

            // Superposition
            for rust_result in superposition(
                given_clause,
                processed_clause,
                given_id,
                processed_id,
                selector,
            ) {
                results.push(self.convert_inference_result(rust_result));
            }
            for rust_result in superposition(
                processed_clause,
                given_clause,
                processed_id,
                given_id,
                selector,
            ) {
                results.push(self.convert_inference_result(rust_result));
            }
        }

        Ok(results)
    }

    /// Add inference result if not redundant, return new clause ID
    pub fn add_inference(&mut self, inference: InferenceResult) -> PyResult<Option<usize>> {
        let clause = inference.clause;

        // Check if tautology
        if clause.is_tautology() {
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
            "all" | "select_all" => {
                self.literal_selector = Box::new(SelectAll) as Box<dyn LiteralSelector + Send>;
                Ok(())
            }
            "max_weight" => {
                self.literal_selector =
                    Box::new(SelectMaxWeight::new()) as Box<dyn LiteralSelector + Send>;
                Ok(())
            }
            _ => Err(PyValueError::new_err(format!(
                "Unknown literal selection: {}",
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
    ///     max_iterations: Maximum number of saturation steps
    ///     timeout_secs: Optional timeout in seconds
    ///     onnx_model_path: Optional path to ONNX clause selector model
    ///
    /// Returns:
    ///     True if proof found, False otherwise
    pub fn run_saturation(
        &mut self,
        max_iterations: usize,
        timeout_secs: Option<f64>,
        onnx_model_path: Option<String>,
    ) -> PyResult<bool> {
        use crate::saturation::{SaturationConfig, SaturationResult, SaturationState};
        use crate::selection::OnnxClauseSelector;
        use std::time::Duration;

        // Build config
        let timeout = timeout_secs
            .map(|s| Duration::from_secs_f64(s))
            .unwrap_or(Duration::from_secs(300));

        let literal_selection = match self.literal_selector.as_ref().name() {
            "SelectMaxWeight" => LiteralSelectionStrategy::SelectMaxWeight,
            "SelectLargestNegative" => LiteralSelectionStrategy::SelectLargestNegative,
            _ => LiteralSelectionStrategy::SelectAll,
        };

        let config = SaturationConfig {
            max_clauses: 100000,
            max_iterations,
            max_clause_size: 100,
            timeout,
            literal_selection,
            step_limit: None,
        };

        // Create saturation state from current clauses
        let initial_clauses: Vec<Clause> = self.clauses.clone();
        let mut state = SaturationState::new(initial_clauses, config);

        // Set ONNX clause selector if provided
        if let Some(model_path) = onnx_model_path {
            match OnnxClauseSelector::new(&model_path) {
                Ok(selector) => {
                    state.set_clause_selector(Box::new(selector));
                }
                Err(e) => {
                    return Err(PyValueError::new_err(format!(
                        "Failed to load ONNX model '{}': {}",
                        model_path, e
                    )));
                }
            }
        }

        // Run saturation
        let result = state.saturate();

        // Copy back the results
        let (proof_found, final_clauses, proof_steps) = match result {
            SaturationResult::Proof(proof) => {
                (true, proof.all_clauses, proof.steps)
            }
            SaturationResult::Saturated(steps, clauses) => (false, clauses, steps),
            SaturationResult::ResourceLimit(steps, clauses) => (false, clauses, steps),
            SaturationResult::Timeout(steps, clauses) => (false, clauses, steps),
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
            self.proof_trace.push(ProofStep {
                clause_id: step.clause_idx,
                parent_ids: step.inference.premises.clone(),
                rule_name: format!("{:?}", step.inference.rule),
                clause_string: step.inference.conclusion.to_string(),
            });
        }

        Ok(proof_found)
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
                        clause_string: self.clauses[current_id].to_string(),
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

        let graph = GraphBuilder::build_from_clause_with_context(clause, max_age);

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
}

impl ProofState {
    /// Convert Rust inference result to Python inference result
    fn convert_inference_result(&self, rust_result: RustInferenceResult) -> InferenceResult {
        let clause_string = rust_result.conclusion.to_string();
        let rule_name = match rust_result.rule {
            InferenceRule::Input => "input",
            InferenceRule::GivenClauseSelection => "given_clause_selection",
            InferenceRule::Resolution => "resolution",
            InferenceRule::Factoring => "factoring",
            InferenceRule::Superposition => "superposition",
            InferenceRule::EqualityResolution => "equality_resolution",
            InferenceRule::EqualityFactoring => "equality_factoring",
            InferenceRule::Demodulation => "demodulation",
        }
        .to_string();

        InferenceResult {
            clause_string,
            parent_ids: rust_result.premises,
            rule_name,
            clause: rust_result.conclusion,
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
fn proofatlas(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ProofState>()?;
    m.add_class::<ClauseInfo>()?;
    m.add_class::<InferenceResult>()?;
    m.add_class::<ProofStep>()?;
    #[cfg(feature = "python")]
    m.add_class::<ClauseGraphData>()?;
    m.add_class::<TrainingExample>()?;
    Ok(())
}
