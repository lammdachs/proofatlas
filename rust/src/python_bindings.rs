//! Python bindings for ProofAtlas using PyO3

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::collections::{HashMap, VecDeque, HashSet};

use crate::core::{Clause, CNFFormula, Term, Variable};
use crate::parser::parse_tptp;
use crate::saturation::{SaturationConfig, LiteralSelectionStrategy};
use crate::selection::{LiteralSelector, SelectAll, SelectMaxWeight};
use crate::inference::{
    resolution, factoring, superposition, equality_resolution, equality_factoring,
    InferenceResult as RustInferenceResult, InferenceRule
};

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
    /// Which inference rules to use
    use_superposition: bool,
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
            use_superposition: true,
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
        self.clauses.get(clause_id)
            .map(|c| c.to_string())
            .ok_or_else(|| PyValueError::new_err(format!("Invalid clause ID: {}", clause_id)))
    }
    
    /// Get basic clause information
    pub fn get_clause_info(&self, clause_id: usize) -> PyResult<ClauseInfo> {
        let clause = self.clauses.get(clause_id)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid clause ID: {}", clause_id)))?;
        
        // Extract literal strings
        let literal_strings: Vec<String> = clause.literals.iter()
            .map(|lit| lit.to_string())
            .collect();
        
        // Extract variables
        let mut variables = HashSet::new();
        for lit in &clause.literals {
            lit.collect_variables(&mut variables);
        }
        let variables: Vec<String> = variables.into_iter()
            .map(|v| v.name.clone())
            .collect();
        
        // Count symbols for weight
        let weight = clause.symbol_count();
        
        // Check properties
        let is_unit = clause.literals.len() == 1;
        let is_horn = clause.literals.iter().filter(|l| l.polarity).count() <= 1;
        let is_equality = clause.literals.iter().any(|l| l.atom.is_equality());
        
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
        })
    }
    
    /// Get literals as strings
    pub fn get_literals_as_strings(&self, clause_id: usize) -> PyResult<Vec<String>> {
        let clause = self.clauses.get(clause_id)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid clause ID: {}", clause_id)))?;
        
        Ok(clause.literals.iter()
            .map(|lit| lit.to_string())
            .collect())
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
    pub fn select_given_clause(&mut self, _py: Python, strategy: Option<&str>) -> PyResult<Option<usize>> {
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
            _ => Ok(self.unprocessed.pop_front()) // Default to age-based
        }
    }
    
    /// Generate inferences with given clause
    pub fn generate_inferences(&self, given_id: usize) -> PyResult<Vec<InferenceResult>> {
        let given_clause = self.clauses.get(given_id)
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
        
        if self.use_superposition {
            for rust_result in equality_factoring(given_clause, given_id, selector) {
                results.push(self.convert_inference_result(rust_result));
            }
        }
        
        // Binary inferences with processed clauses
        for &processed_id in &self.processed {
            let processed_clause = &self.clauses[processed_id];
            
            // Resolution in both directions
            for rust_result in resolution(given_clause, processed_clause, given_id, processed_id, selector) {
                results.push(self.convert_inference_result(rust_result));
            }
            for rust_result in resolution(processed_clause, given_clause, processed_id, given_id, selector) {
                results.push(self.convert_inference_result(rust_result));
            }
            
            // Superposition if enabled
            if self.use_superposition {
                for rust_result in superposition(given_clause, processed_clause, given_id, processed_id, selector) {
                    results.push(self.convert_inference_result(rust_result));
                }
                for rust_result in superposition(processed_clause, given_clause, processed_id, given_id, selector) {
                    results.push(self.convert_inference_result(rust_result));
                }
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
                self.literal_selector = Box::new(SelectMaxWeight::new()) as Box<dyn LiteralSelector + Send>;
                Ok(())
            }
            _ => Err(PyValueError::new_err(format!("Unknown literal selection: {}", strategy)))
        }
    }
    
    /// Set whether to use superposition
    pub fn set_use_superposition(&mut self, use_superposition: bool) {
        self.use_superposition = use_superposition;
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
        let unit_count = self.clauses.iter().filter(|c| c.literals.len() == 1).count();
        stats.insert("unit_clauses".to_string(), unit_count);
        
        stats
    }
    
    /// Get proof trace
    pub fn get_proof_trace(&self) -> Vec<ProofStep> {
        // Find empty clause
        let empty_clause_id = self.clauses.iter()
            .position(|c| c.is_empty());
        
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
}

impl ProofState {
    /// Convert Rust inference result to Python inference result
    fn convert_inference_result(&self, rust_result: RustInferenceResult) -> InferenceResult {
        let clause_string = rust_result.conclusion.to_string();
        let rule_name = match rust_result.rule {
            InferenceRule::Resolution => "resolution",
            InferenceRule::Factoring => "factoring",
            InferenceRule::Superposition => "superposition",
            InferenceRule::EqualityResolution => "equality_resolution",
            InferenceRule::EqualityFactoring => "equality_factoring",
        }.to_string();
        
        InferenceResult {
            clause_string,
            parent_ids: rust_result.premises,
            rule_name,
            clause: rust_result.conclusion,
        }
    }
}

/// Python module definition
#[pymodule]
fn proofatlas(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ProofState>()?;
    m.add_class::<ClauseInfo>()?;
    m.add_class::<InferenceResult>()?;
    m.add_class::<ProofStep>()?;
    Ok(())
}