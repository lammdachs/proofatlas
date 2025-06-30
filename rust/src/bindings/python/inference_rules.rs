//! Python bindings for inference rules

use pyo3::prelude::*;
use crate::rules::{
    resolve_clauses, factor_clause, superpose_clauses,
    equality_resolve, equality_factor
};
use crate::saturation::{apply_literal_selection, SelectNegative};
use super::array_bindings::PyProblem;

/// Python wrapper for inference result
#[pyclass(name = "InferenceResult")]
#[derive(Clone)]
pub struct PyInferenceResult {
    pub rule: String,
    pub parent_clauses: Vec<usize>,
    pub new_clause_idx: Option<usize>,
}

#[pymethods]
impl PyInferenceResult {
    #[getter]
    fn rule(&self) -> &str {
        &self.rule
    }
    
    #[getter]
    fn parent_clauses(&self) -> Vec<usize> {
        self.parent_clauses.clone()
    }
    
    #[getter]
    fn new_clause_idx(&self) -> Option<usize> {
        self.new_clause_idx
    }
}

/// Apply resolution between two clauses
#[pyfunction]
#[pyo3(signature = (problem, clause1_idx, clause2_idx, apply_selection=false))]
pub fn py_resolve_clauses(
    problem: &mut PyProblem,
    clause1_idx: usize,
    clause2_idx: usize,
    apply_selection: bool,
) -> PyResult<Vec<PyInferenceResult>> {
    // Apply literal selection if requested
    if apply_selection {
        apply_literal_selection(&mut problem.inner, clause1_idx, &SelectNegative);
        apply_literal_selection(&mut problem.inner, clause2_idx, &SelectNegative);
    }
    
    let results = resolve_clauses(&mut problem.inner, clause1_idx, clause2_idx);
    
    Ok(results.into_iter().map(|r| PyInferenceResult {
        rule: r.applied_rule.clone(),
        parent_clauses: r.parent_clauses,
        new_clause_idx: r.new_clause_idx,
    }).collect())
}

/// Apply factoring to a clause
#[pyfunction]
#[pyo3(signature = (problem, clause_idx, apply_selection=false))]
pub fn py_factor_clause(
    problem: &mut PyProblem,
    clause_idx: usize,
    apply_selection: bool,
) -> PyResult<Vec<PyInferenceResult>> {
    if apply_selection {
        apply_literal_selection(&mut problem.inner, clause_idx, &SelectNegative);
    }
    
    let results = factor_clause(&mut problem.inner, clause_idx);
    
    Ok(results.into_iter().map(|r| PyInferenceResult {
        rule: r.applied_rule.clone(),
        parent_clauses: r.parent_clauses,
        new_clause_idx: r.new_clause_idx,
    }).collect())
}

/// Apply superposition between two clauses
#[pyfunction]
#[pyo3(signature = (problem, clause1_idx, clause2_idx, apply_selection=false))]
pub fn py_superpose_clauses(
    problem: &mut PyProblem,
    clause1_idx: usize,
    clause2_idx: usize,
    apply_selection: bool,
) -> PyResult<Vec<PyInferenceResult>> {
    if apply_selection {
        apply_literal_selection(&mut problem.inner, clause1_idx, &SelectNegative);
        apply_literal_selection(&mut problem.inner, clause2_idx, &SelectNegative);
    }
    
    let results = superpose_clauses(&mut problem.inner, clause1_idx, clause2_idx);
    
    Ok(results.into_iter().map(|r| PyInferenceResult {
        rule: r.applied_rule.clone(),
        parent_clauses: r.parent_clauses,
        new_clause_idx: r.new_clause_idx,
    }).collect())
}

/// Apply equality resolution to a clause
#[pyfunction]
#[pyo3(signature = (problem, clause_idx, apply_selection=false))]
pub fn py_equality_resolve(
    problem: &mut PyProblem,
    clause_idx: usize,
    apply_selection: bool,
) -> PyResult<Vec<PyInferenceResult>> {
    if apply_selection {
        apply_literal_selection(&mut problem.inner, clause_idx, &SelectNegative);
    }
    
    let results = equality_resolve(&mut problem.inner, clause_idx);
    
    Ok(results.into_iter().map(|r| PyInferenceResult {
        rule: r.applied_rule.clone(),
        parent_clauses: r.parent_clauses,
        new_clause_idx: r.new_clause_idx,
    }).collect())
}

/// Apply equality factoring to a clause
#[pyfunction]
#[pyo3(signature = (problem, clause_idx, apply_selection=false))]
pub fn py_equality_factor(
    problem: &mut PyProblem,
    clause_idx: usize,
    apply_selection: bool,
) -> PyResult<Vec<PyInferenceResult>> {
    if apply_selection {
        apply_literal_selection(&mut problem.inner, clause_idx, &SelectNegative);
    }
    
    let results = equality_factor(&mut problem.inner, clause_idx);
    
    Ok(results.into_iter().map(|r| PyInferenceResult {
        rule: r.applied_rule.clone(),
        parent_clauses: r.parent_clauses,
        new_clause_idx: r.new_clause_idx,
    }).collect())
}

/// Check if problem has empty clause
#[pyfunction]
pub fn py_has_empty_clause(problem: &PyProblem) -> bool {
    problem.inner.has_empty_clause()
}

/// Get clause literal count
#[pyfunction]
pub fn py_clause_literal_count(problem: &PyProblem, clause_idx: usize) -> PyResult<usize> {
    if clause_idx >= problem.inner.num_clauses {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            format!("Clause index {} out of range", clause_idx)
        ));
    }
    
    Ok(problem.inner.clause_literals(clause_idx).len())
}

/// Python module for inference rules
pub fn add_inference_rules(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyInferenceResult>()?;
    m.add_function(wrap_pyfunction!(py_resolve_clauses, m)?)?;
    m.add_function(wrap_pyfunction!(py_factor_clause, m)?)?;
    m.add_function(wrap_pyfunction!(py_superpose_clauses, m)?)?;
    m.add_function(wrap_pyfunction!(py_equality_resolve, m)?)?;
    m.add_function(wrap_pyfunction!(py_equality_factor, m)?)?;
    m.add_function(wrap_pyfunction!(py_has_empty_clause, m)?)?;
    m.add_function(wrap_pyfunction!(py_clause_literal_count, m)?)?;
    Ok(())
}