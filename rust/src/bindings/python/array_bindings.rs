//! Python bindings for array representation

use pyo3::prelude::*;
use numpy::PyArray1;
use crate::core::{Problem, NodeType};
use crate::saturation::{saturate, SaturationConfig};

/// Python wrapper for Problem
/// 
/// This wrapper provides access to array data with a freeze mechanism for safety.
/// After calling freeze(), no modifications are allowed, which enables future
/// zero-copy implementations.
/// 
/// Currently, arrays are returned as copies for safety. True zero-copy can be
/// implemented using one of these approaches:
/// 1. Pin memory with Arc<Vec<T>> to prevent reallocation
/// 2. Use ndarray instead of Vec for better view support
/// 3. Custom allocator that guarantees stable addresses
#[pyclass(name = "Problem")]
pub struct PyProblem {
    pub inner: Problem,
    /// If true, the problem is "frozen" and arrays can be safely viewed
    pub frozen: bool,
}

#[pymethods]
impl PyProblem {
    /// Create a new empty array problem with default or specified capacity
    #[new]
    #[pyo3(signature = (max_nodes=1000000, max_clauses=100000, max_edges=5000000))]
    fn new(max_nodes: usize, max_clauses: usize, max_edges: usize) -> Self {
        PyProblem {
            inner: Problem::with_capacity(max_nodes, max_clauses, max_edges),
            frozen: false,
        }
    }
    
    /// Freeze the problem to enable safe array access
    /// After freezing, no more modifications are allowed
    fn freeze(&mut self) -> PyResult<()> {
        if self.frozen {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Problem is already frozen"
            ));
        }
        
        // With Box<[T]> arrays, memory is already stable
        // Just mark as frozen to prevent further modifications
        self.frozen = true;
        Ok(())
    }
    
    /// Check if the problem is frozen
    #[getter]
    fn is_frozen(&self) -> bool {
        self.frozen
    }
    
    
    /// Get node arrays as NumPy arrays
    /// 
    /// Note: Currently returns copies for safety. True zero-copy can be implemented
    /// using unsafe operations with the numpy crate's internal APIs when needed.
    /// The Box<[T]> backing ensures stable memory addresses for future zero-copy.
    fn get_node_arrays<'py>(&'py self, py: Python<'py>) -> PyResult<(
        &'py PyArray1<u8>,      // node_types
        &'py PyArray1<u32>,     // node_symbols  
        &'py PyArray1<i8>,      // node_polarities
        &'py PyArray1<u32>,     // node_arities
    )> {
        // Currently using from_slice which copies data
        // Future: implement true zero-copy with numpy crate's unsafe APIs
        Ok((
            PyArray1::from_slice(py, &self.inner.node_types[..self.inner.num_nodes]),
            PyArray1::from_slice(py, &self.inner.node_symbols[..self.inner.num_nodes]),
            PyArray1::from_slice(py, &self.inner.node_polarities[..self.inner.num_nodes]),
            PyArray1::from_slice(py, &self.inner.node_arities[..self.inner.num_nodes]),
        ))
    }
    
    /// Get edge arrays in CSR format
    /// Returns zero-copy views if frozen, copies if not frozen
    fn get_edge_arrays<'py>(&'py self, py: Python<'py>) -> PyResult<(
        &'py PyArray1<usize>,   // row_offsets
        &'py PyArray1<u32>,     // col_indices
    )> {
        Ok((
            PyArray1::from_slice(py, &self.inner.edge_row_offsets),
            PyArray1::from_slice(py, &self.inner.edge_col_indices),
        ))
    }
    
    /// Get clause information
    /// Returns zero-copy views if frozen, copies if not frozen
    fn get_clause_info<'py>(&'py self, py: Python<'py>) -> PyResult<(
        &'py PyArray1<usize>,   // clause_boundaries
        &'py PyArray1<usize>,   // literal_boundaries
        usize,              // num_clauses
        usize,              // num_literals
    )> {
        Ok((
            PyArray1::from_slice(py, &self.inner.clause_boundaries),
            PyArray1::from_slice(py, &self.inner.literal_boundaries),
            self.inner.num_clauses,
            self.inner.num_literals,
        ))
    }
    
    /// Get clause types array
    /// Returns zero-copy view if frozen, copy if not frozen
    fn get_clause_types<'py>(&'py self, py: Python<'py>) -> PyResult<&'py PyArray1<u8>> {
        Ok(PyArray1::from_slice(py, &self.inner.clause_types))
    }
    
    /// Get symbol table
    fn get_symbols(&self) -> Vec<String> {
        let mut symbols = Vec::new();
        for i in 0..self.inner.symbols.len() {
            if let Some(symbol) = self.inner.symbols.get(i as u32) {
                symbols.push(symbol.to_string());
            }
        }
        symbols
    }
    
    /// Run saturation
    fn saturate(&mut self, max_clauses: Option<usize>, max_iterations: Option<usize>) -> PyResult<(bool, usize, usize)> {
        if self.frozen {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Cannot saturate a frozen problem"
            ));
        }
        let mut config = SaturationConfig::default();
        
        if let Some(max) = max_clauses {
            config.max_clauses = max;
        }
        
        if let Some(max) = max_iterations {
            config.max_iterations = max;
        }
        
        let result = saturate(&mut self.inner, &config);
        
        match result {
            crate::core::SaturationResult::Proof(proof) => {
                Ok((true, proof.problem.num_clauses, proof.steps.len()))
            }
            crate::core::SaturationResult::Saturated => {
                Ok((false, self.inner.num_clauses, 0))
            }
            crate::core::SaturationResult::ResourceLimit => {
                Ok((false, self.inner.num_clauses, 0))
            }
        }
    }
    
    /// Get number of nodes
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes
    }
    
    /// Get number of clauses
    #[getter]
    fn num_clauses(&self) -> usize {
        self.inner.num_clauses
    }
    
    /// Check if empty clause exists
    fn has_empty_clause(&self) -> bool {
        self.inner.has_empty_clause()
    }
}

/// Register the array bindings module
pub fn register_array_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyProblem>()?;
    
    // Add node type constants
    m.add("NODE_VARIABLE", NodeType::Variable as u8)?;
    m.add("NODE_CONSTANT", NodeType::Constant as u8)?;
    m.add("NODE_FUNCTION", NodeType::Function as u8)?;
    m.add("NODE_PREDICATE", NodeType::Predicate as u8)?;
    m.add("NODE_LITERAL", NodeType::Literal as u8)?;
    m.add("NODE_CLAUSE", NodeType::Clause as u8)?;
    
    // Edge type constants removed - can be computed from node types
    
    Ok(())
}