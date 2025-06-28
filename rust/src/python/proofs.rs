//! Python bindings for proof types

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyList;
use std::collections::HashMap;
use serde_json::Value;

use crate::proofs::{ProofState, ProofStep, Proof, RuleApplication};

/// Python wrapper for RuleApplication
#[pyclass(name = "RuleApplication")]
#[derive(Clone)]
pub struct PyRuleApplication {
    pub inner: RuleApplication,
}

#[pymethods]
impl PyRuleApplication {
    #[new]
    #[pyo3(signature = (rule_name, parents, generated_clauses=None, deleted_clause_indices=None, metadata=None))]
    fn new(
        rule_name: String,
        parents: Vec<usize>,
        generated_clauses: Option<Vec<String>>,
        deleted_clause_indices: Option<Vec<usize>>,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> PyResult<Self> {
        let mut rule_app = RuleApplication::new(rule_name, parents);
        
        // For now, we'll skip parsing generated clauses from strings
        // In full implementation, would parse clause strings
        
        if let Some(deleted) = deleted_clause_indices {
            rule_app = rule_app.with_deleted(deleted);
        }
        
        // Convert metadata to serde_json::Value
        if let Some(meta) = metadata {
            Python::with_gil(|py| {
                for (key, value) in meta {
                    // Simple conversion - in full implementation would handle all Python types
                    if let Ok(s) = value.extract::<String>(py) {
                        rule_app.metadata.insert(key, Value::String(s));
                    }
                }
            });
        }
        
        Ok(PyRuleApplication { inner: rule_app })
    }
    
    #[getter]
    fn rule_name(&self) -> &str {
        &self.inner.rule_name
    }
    
    #[getter]
    fn parents(&self) -> Vec<usize> {
        self.inner.parents.clone()
    }
    
    #[getter]
    fn deleted_clause_indices(&self) -> Vec<usize> {
        self.inner.deleted_clause_indices.clone()
    }
}

/// Python wrapper for ProofState
#[pyclass(name = "ProofState")]
#[derive(Clone)]
pub struct PyProofState {
    pub inner: ProofState,
}

#[pymethods]
impl PyProofState {
    #[new]
    fn new(processed: Vec<String>, unprocessed: Vec<String>) -> PyResult<Self> {
        // For now, create empty vectors
        // Full implementation would parse clause strings
        let state = ProofState::new(Vec::new(), Vec::new());
        Ok(PyProofState { inner: state })
    }
    
    /// Get all clauses
    #[getter]
    fn all_clauses(&self, py: Python) -> PyResult<PyObject> {
        let all = PyList::empty(py);
        for clause in self.inner.all_clauses() {
            all.append(format!("{}", clause))?;
        }
        Ok(all.into())
    }
    
    /// Get processed clauses
    #[getter]
    fn processed(&self, py: Python) -> PyResult<PyObject> {
        let processed = PyList::empty(py);
        for clause in &self.inner.processed {
            processed.append(format!("{}", clause))?;
        }
        Ok(processed.into())
    }
    
    /// Get unprocessed clauses
    #[getter]
    fn unprocessed(&self, py: Python) -> PyResult<PyObject> {
        let unprocessed = PyList::empty(py);
        for clause in &self.inner.unprocessed {
            unprocessed.append(format!("{}", clause))?;
        }
        Ok(unprocessed.into())
    }
    
    /// Check if contains empty clause
    #[getter]
    fn contains_empty_clause(&self) -> bool {
        self.inner.contains_empty_clause()
    }
    
    /// Move clause to processed
    fn move_to_processed(&mut self, index: usize) -> PyResult<()> {
        self.inner.move_to_processed(index)
            .ok_or_else(|| PyValueError::new_err("Invalid clause index"))?;
        Ok(())
    }
}

/// Python wrapper for ProofStep
#[pyclass(name = "ProofStep")]
#[derive(Clone)]
pub struct PyProofStep {
    pub inner: ProofStep,
}

#[pymethods]
impl PyProofStep {
    #[new]
    #[pyo3(signature = (state, selected_clause=None, applied_rules=None, metadata=None))]
    fn new(
        state: PyProofState,
        selected_clause: Option<usize>,
        applied_rules: Option<Vec<PyRuleApplication>>,
        metadata: Option<HashMap<String, PyObject>>,
    ) -> PyResult<Self> {
        let mut step = ProofStep::new(state.inner.clone());
        
        if let Some(selected) = selected_clause {
            step = step.with_selected_clause(selected);
        }
        
        if let Some(rules) = applied_rules {
            let rule_apps: Vec<RuleApplication> = rules.into_iter()
                .map(|r| r.inner.clone())
                .collect();
            step = step.with_applied_rules(rule_apps);
        }
        
        // Handle metadata similar to RuleApplication
        
        Ok(PyProofStep { inner: step })
    }
    
    #[getter]
    fn state(&self) -> PyProofState {
        PyProofState { inner: self.inner.state.clone() }
    }
    
    #[getter]
    fn selected_clause(&self) -> Option<usize> {
        self.inner.selected_clause
    }
    
    #[getter]
    fn applied_rules(&self) -> Vec<PyRuleApplication> {
        self.inner.applied_rules.iter()
            .map(|r| PyRuleApplication { inner: r.clone() })
            .collect()
    }
}

/// Python wrapper for Proof
#[pyclass(name = "Proof")]
#[derive(Clone)]
pub struct PyProof {
    pub inner: Proof,
}

#[pymethods]
impl PyProof {
    #[new]
    #[pyo3(signature = (initial_state=None))]
    fn new(initial_state: Option<PyProofState>) -> PyResult<Self> {
        let proof = if let Some(state) = initial_state {
            Proof::new(state.inner)
        } else {
            Proof::empty()
        };
        Ok(PyProof { inner: proof })
    }
    
    /// Add a step to the proof
    fn add_step(&mut self, step: PyProofStep) {
        self.inner.add_step(step.inner);
    }
    
    /// Finalize the proof
    fn finalize(&mut self, final_state: PyProofState) {
        self.inner.finalize(final_state.inner);
    }
    
    /// Get initial state
    #[getter]
    fn initial_state(&self) -> Option<PyProofState> {
        self.inner.initial_state()
            .map(|s| PyProofState { inner: s.clone() })
    }
    
    /// Get final state
    #[getter]
    fn final_state(&self) -> Option<PyProofState> {
        self.inner.final_state()
            .map(|s| PyProofState { inner: s.clone() })
    }
    
    /// Get length
    #[getter]
    fn length(&self) -> usize {
        self.inner.length()
    }
    
    /// Get all steps
    #[getter]
    fn steps(&self) -> Vec<PyProofStep> {
        self.inner.steps.iter()
            .map(|s| PyProofStep { inner: s.clone() })
            .collect()
    }
    
    /// Get a specific step
    fn get_step(&self, index: usize) -> Option<PyProofStep> {
        self.inner.get_step(index)
            .map(|s| PyProofStep { inner: s.clone() })
    }
    
    /// Check if found contradiction
    fn found_contradiction(&self) -> bool {
        self.inner.found_contradiction()
    }
    
    /// Get selected clauses
    fn get_selected_clauses(&self) -> Vec<usize> {
        self.inner.get_selected_clauses()
    }
}

/// Register the proofs module with Python
pub fn register_proofs_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRuleApplication>()?;
    m.add_class::<PyProofState>()?;
    m.add_class::<PyProofStep>()?;
    m.add_class::<PyProof>()?;
    Ok(())
}