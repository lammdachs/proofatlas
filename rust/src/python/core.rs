//! Python bindings for core logic types

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyList, PyTuple};
use crate::core::logic::Problem;
use crate::python::types;

/// Python wrapper for Problem
#[pyclass(name = "Problem")]
#[derive(Clone)]
pub struct PyProblem {
    pub inner: Problem,
}

#[pymethods]
impl PyProblem {
    /// Create a new problem from clauses
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(args: &PyTuple, kwargs: Option<&PyDict>) -> PyResult<Self> {
        // For now, create an empty problem
        // In a full implementation, we'd parse clauses from args
        let problem = Problem::new(Vec::new());
        
        // Handle conjecture_indices if provided
        if let Some(kwargs) = kwargs {
            if let Ok(Some(indices)) = kwargs.get_item("conjecture_indices") {
                // Would handle conjecture indices here
            }
        }
        
        Ok(PyProblem { inner: problem })
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("Problem({} clauses)", self.inner.clauses.len())
    }
    
    /// Get number of clauses
    fn __len__(&self) -> usize {
        self.inner.clauses.len()
    }
    
    /// Get clauses as a list (returns JSON representation)
    #[getter]
    fn clauses(&self, py: Python) -> PyResult<PyObject> {
        // Convert clauses to Python-friendly format
        let clauses_list = PyList::empty(py);
        for clause in &self.inner.clauses {
            let clause_str = format!("{}", clause);
            clauses_list.append(clause_str)?;
        }
        Ok(clauses_list.into())
    }
    
    /// Get conjecture indices
    #[getter]
    fn conjecture_indices(&self, py: Python) -> PyResult<PyObject> {
        let indices_list = PyList::new(py, self.inner.conjecture_indices.iter().cloned());
        Ok(indices_list.into())
    }
    
    /// Check if a clause is from a conjecture
    fn is_conjecture_clause(&self, index: usize) -> bool {
        self.inner.is_conjecture_clause(index)
    }
    
    /// Get all conjecture clauses
    fn get_conjecture_clauses(&self, py: Python) -> PyResult<PyObject> {
        let result = PyList::empty(py);
        for (idx, clause) in self.inner.get_conjecture_clauses() {
            let tuple = PyTuple::new(py, &[idx.into_py(py), format!("{}", clause).into_py(py)]);
            result.append(tuple)?;
        }
        Ok(result.into())
    }
    
    /// Count total literals
    fn count_literals(&self) -> usize {
        self.inner.count_literals()
    }
    
    /// Create from a dictionary (for deserialization)
    #[staticmethod]
    fn from_dict(py: Python, dict: &PyDict) -> PyResult<Self> {
        // Parse problem from dictionary representation
        let num_clauses = dict.get_item("num_clauses")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<usize>().ok())
            .unwrap_or(0);
            
        let clauses = dict.get_item("clauses")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<Vec<PyObject>>().ok())
            .unwrap_or_default();
            
        let conjecture_indices = dict.get_item("conjecture_indices")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<Vec<usize>>().ok())
            .unwrap_or_default();
        
        // For now, create empty problem
        // Full implementation would parse clauses
        let problem = Problem::with_conjectures(Vec::new(), conjecture_indices);
        
        Ok(PyProblem { inner: problem })
    }
    
    /// Convert to dictionary (for serialization)
    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        types::problem_to_dict(py, &self.inner).map(|dict| dict.into())
    }
}

/// Register the core module with Python
pub fn register_core_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyProblem>()?;
    Ok(())
}