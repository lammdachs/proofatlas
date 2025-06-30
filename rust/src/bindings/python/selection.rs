//! Python bindings for clause and literal selection

use pyo3::prelude::*;
use crate::saturation::{
    apply_literal_selection,
    SelectAll, SelectNegative, SelectFirstNegative, SelectLargestNegative
};
use super::array_bindings::PyProblem;

/// Python wrapper for literal selection strategies
#[pyclass(name = "LiteralSelectionStrategy")]
#[derive(Clone)]
pub enum PyLiteralSelectionStrategy {
    All,
    Negative,
    FirstNegative,
    LargestNegative,
}

/// Apply literal selection to a clause
#[pyfunction]
pub fn py_apply_literal_selection(
    problem: &mut PyProblem,
    clause_idx: usize,
    strategy: PyLiteralSelectionStrategy,
) -> PyResult<()> {
    if clause_idx >= problem.inner.num_clauses {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            format!("Clause index {} out of range", clause_idx)
        ));
    }
    
    match strategy {
        PyLiteralSelectionStrategy::All => {
            apply_literal_selection(&mut problem.inner, clause_idx, &SelectAll);
        }
        PyLiteralSelectionStrategy::Negative => {
            apply_literal_selection(&mut problem.inner, clause_idx, &SelectNegative);
        }
        PyLiteralSelectionStrategy::FirstNegative => {
            apply_literal_selection(&mut problem.inner, clause_idx, &SelectFirstNegative);
        }
        PyLiteralSelectionStrategy::LargestNegative => {
            apply_literal_selection(&mut problem.inner, clause_idx, &SelectLargestNegative);
        }
    }
    
    Ok(())
}

/// Get selected literals for a clause
#[pyfunction]
pub fn py_get_selected_literals(
    problem: &PyProblem,
    clause_idx: usize,
) -> PyResult<Vec<usize>> {
    if clause_idx >= problem.inner.num_clauses {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            format!("Clause index {} out of range", clause_idx)
        ));
    }
    
    let literals = problem.inner.clause_literals(clause_idx);
    let selected: Vec<usize> = literals.iter()
        .enumerate()
        .filter(|(_, &lit)| problem.inner.node_selected[lit])
        .map(|(idx, _)| idx)
        .collect();
    
    Ok(selected)
}

/// Clear all literal selections
#[pyfunction]
pub fn py_clear_literal_selections(problem: &mut PyProblem) {
    for i in 0..problem.inner.node_selected.len() {
        problem.inner.node_selected[i] = false;
    }
}

/// Set specific literals as selected
#[pyfunction]
pub fn py_set_selected_literals(
    problem: &mut PyProblem,
    clause_idx: usize,
    literal_indices: Vec<usize>,
) -> PyResult<()> {
    if clause_idx >= problem.inner.num_clauses {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            format!("Clause index {} out of range", clause_idx)
        ));
    }
    
    let literals = problem.inner.clause_literals(clause_idx);
    
    // First clear all selections for this clause
    for &lit in &literals {
        problem.inner.node_selected[lit] = false;
    }
    
    // Then set the specified ones
    for idx in literal_indices {
        if idx >= literals.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                format!("Literal index {} out of range for clause", idx)
            ));
        }
        problem.inner.node_selected[literals[idx]] = true;
    }
    
    Ok(())
}

/// Check if a clause has selected literals
#[pyfunction]
pub fn py_has_selected_literals(
    problem: &PyProblem,
    clause_idx: usize,
) -> PyResult<bool> {
    if clause_idx >= problem.inner.num_clauses {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            format!("Clause index {} out of range", clause_idx)
        ));
    }
    
    let literals = problem.inner.clause_literals(clause_idx);
    let has_selected = literals.iter().any(|&lit| problem.inner.node_selected[lit]);
    
    Ok(has_selected)
}

/// Python module for selection functions
pub fn add_selection_functions(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyLiteralSelectionStrategy>()?;
    m.add_function(wrap_pyfunction!(py_apply_literal_selection, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_selected_literals, m)?)?;
    m.add_function(wrap_pyfunction!(py_clear_literal_selections, m)?)?;
    m.add_function(wrap_pyfunction!(py_set_selected_literals, m)?)?;
    m.add_function(wrap_pyfunction!(py_has_selected_literals, m)?)?;
    Ok(())
}