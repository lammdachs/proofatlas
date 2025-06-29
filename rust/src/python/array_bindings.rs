//! Python bindings for array representation

use pyo3::prelude::*;
use numpy::{PyArray1, IntoPyArray};
use crate::array_repr::types::{ArrayProblem, NodeType, EdgeType};
use crate::array_repr::builder::ArrayBuilder;
use crate::array_repr::saturation::{saturate, SaturationConfig};

/// Python wrapper for ArrayProblem
#[pyclass(name = "ArrayProblem")]
pub struct PyArrayProblem {
    pub inner: ArrayProblem,
}

#[pymethods]
impl PyArrayProblem {
    /// Create a new empty array problem
    #[new]
    fn new() -> Self {
        PyArrayProblem {
            inner: ArrayProblem::new(),
        }
    }
    
    /// Create from a traditional problem
    #[staticmethod]
    fn from_problem(problem: &crate::python::core::PyProblem) -> Self {
        let mut array_problem = ArrayProblem::new();
        let mut builder = ArrayBuilder::new(&mut array_problem);
        
        // Convert all clauses
        for clause in &problem.inner.clauses {
            builder.add_clause(clause);
        }
        
        PyArrayProblem {
            inner: array_problem,
        }
    }
    
    /// Get node arrays as NumPy arrays
    fn get_node_arrays<'py>(&self, py: Python<'py>) -> PyResult<(
        &'py PyArray1<u8>,      // node_types
        &'py PyArray1<u32>,     // node_symbols  
        &'py PyArray1<i8>,      // node_polarities
        &'py PyArray1<u32>,     // node_arities
    )> {
        // Convert NodeType enum to u8
        let node_types_u8: Vec<u8> = self.inner.node_types.iter()
            .map(|&t| t as u8)
            .collect();
        
        Ok((
            node_types_u8.into_pyarray(py),
            self.inner.node_symbols.clone().into_pyarray(py),
            self.inner.node_polarities.clone().into_pyarray(py),
            self.inner.node_arities.clone().into_pyarray(py),
        ))
    }
    
    /// Get edge arrays in CSR format
    fn get_edge_arrays<'py>(&self, py: Python<'py>) -> PyResult<(
        &'py PyArray1<usize>,   // row_offsets
        &'py PyArray1<u32>,     // col_indices
        &'py PyArray1<u8>,      // edge_types
    )> {
        // Convert EdgeType enum to u8
        let edge_types_u8: Vec<u8> = self.inner.edge_types.iter()
            .map(|&t| t as u8)
            .collect();
        
        Ok((
            self.inner.edge_row_offsets.clone().into_pyarray(py),
            self.inner.edge_col_indices.clone().into_pyarray(py),
            edge_types_u8.into_pyarray(py),
        ))
    }
    
    /// Get clause information
    fn get_clause_info<'py>(&self, py: Python<'py>) -> PyResult<(
        &'py PyArray1<usize>,   // clause_boundaries
        &'py PyArray1<usize>,   // literal_boundaries
        usize,              // num_clauses
        usize,              // num_literals
    )> {
        Ok((
            self.inner.clause_boundaries.clone().into_pyarray(py),
            self.inner.literal_boundaries.clone().into_pyarray(py),
            self.inner.num_clauses,
            self.inner.num_literals,
        ))
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
        let mut config = SaturationConfig::default();
        
        if let Some(max) = max_clauses {
            config.max_clauses = max;
        }
        
        if let Some(max) = max_iterations {
            config.max_iterations = max;
        }
        
        let result = saturate(&mut self.inner, &config);
        
        Ok((
            result.found_empty_clause,
            result.num_clauses_generated,
            result.num_iterations,
        ))
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
    m.add_class::<PyArrayProblem>()?;
    
    // Add node type constants
    m.add("NODE_VARIABLE", NodeType::Variable as u8)?;
    m.add("NODE_CONSTANT", NodeType::Constant as u8)?;
    m.add("NODE_FUNCTION", NodeType::Function as u8)?;
    m.add("NODE_PREDICATE", NodeType::Predicate as u8)?;
    m.add("NODE_LITERAL", NodeType::Literal as u8)?;
    m.add("NODE_CLAUSE", NodeType::Clause as u8)?;
    
    // Add edge type constants
    m.add("EDGE_HAS_ARGUMENT", EdgeType::HasArgument as u8)?;
    m.add("EDGE_HAS_LITERAL", EdgeType::HasLiteral as u8)?;
    m.add("EDGE_HAS_PREDICATE", EdgeType::HasPredicate as u8)?;
    
    Ok(())
}