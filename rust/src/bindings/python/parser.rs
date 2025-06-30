//! Python bindings for the TPTP parser

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::parsing::tptp_parser::{parse_file as rust_parse_file, parse_string as rust_parse_string};
use crate::parsing::prescan::prescan_file as rust_prescan_file;
use crate::core::Problem;
use crate::bindings::python::array_bindings::PyProblem;
use std::path::Path;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

/// Register parser functions in the Python module
pub fn register_parser_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_file, m)?)?;
    m.add_function(wrap_pyfunction!(parse_string, m)?)?;
    m.add_function(wrap_pyfunction!(parse_file_to_dict, m)?)?;
    m.add_function(wrap_pyfunction!(parse_file_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(parse_string_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(prescan_file, m)?)?;
    m.add_function(wrap_pyfunction!(count_literals_in_file, m)?)?;
    m.add_function(wrap_pyfunction!(parse_and_saturate, m)?)?;
    m.add_class::<RustTPTPParser>()?;
    Ok(())
}

/// Parse a TPTP file and return array representation as a dictionary
#[pyfunction]
#[pyo3(signature = (file_path, include_path=None))]
fn parse_file(py: Python, file_path: String, include_path: Option<String>) -> PyResult<PyObject> {
    match rust_parse_file(&file_path, include_path.as_deref()) {
        Ok(problem) => {
            let dict = array_problem_to_dict(py, &problem)?;
            Ok(dict.into())
        },
        Err(e) => Err(PyValueError::new_err(format!("Parse error: {}", e))),
    }
}

/// Parse TPTP string content and return array representation as a dictionary
#[pyfunction]
fn parse_string(py: Python, content: String) -> PyResult<PyObject> {
    match rust_parse_string(&content) {
        Ok(problem) => {
            let dict = array_problem_to_dict(py, &problem)?;
            Ok(dict.into())
        },
        Err(e) => Err(PyValueError::new_err(format!("Parse error: {}", e))),
    }
}

/// Parse a TPTP file and return a PyProblem with specified capacity
#[pyfunction]
#[pyo3(signature = (file_path, max_nodes=1000000, max_clauses=100000, max_edges=5000000, include_path=None))]
fn parse_file_to_array(
    file_path: String, 
    max_nodes: usize,
    max_clauses: usize,
    max_edges: usize,
    include_path: Option<String>
) -> PyResult<PyProblem> {
    // First parse to get the initial problem
    let parsed = rust_parse_file(&file_path, include_path.as_deref())
        .map_err(|e| PyValueError::new_err(format!("Parse error: {}", e)))?;
    
    // Create a new problem with specified capacity
    let mut problem = Problem::with_capacity(max_nodes, max_clauses, max_edges);
    
    // Copy the parsed data into the pre-allocated problem
    // Note: This is a temporary solution until parser directly supports capacity
    problem.num_nodes = parsed.num_nodes;
    problem.num_clauses = parsed.num_clauses;
    problem.num_literals = parsed.num_literals;
    problem.num_edges = parsed.num_edges;
    problem.symbols = parsed.symbols;
    
    // Copy arrays (up to the used portion)
    problem.node_types[..parsed.num_nodes].copy_from_slice(&parsed.node_types[..parsed.num_nodes]);
    problem.node_symbols[..parsed.num_nodes].copy_from_slice(&parsed.node_symbols[..parsed.num_nodes]);
    problem.node_polarities[..parsed.num_nodes].copy_from_slice(&parsed.node_polarities[..parsed.num_nodes]);
    problem.node_arities[..parsed.num_nodes].copy_from_slice(&parsed.node_arities[..parsed.num_nodes]);
    problem.node_selected[..parsed.num_nodes].copy_from_slice(&parsed.node_selected[..parsed.num_nodes]);
    
    problem.edge_row_offsets[..=parsed.num_nodes].copy_from_slice(&parsed.edge_row_offsets[..=parsed.num_nodes]);
    problem.edge_col_indices[..parsed.num_edges].copy_from_slice(&parsed.edge_col_indices[..parsed.num_edges]);
    
    problem.clause_boundaries[..=parsed.num_clauses].copy_from_slice(&parsed.clause_boundaries[..=parsed.num_clauses]);
    problem.clause_types[..parsed.num_clauses].copy_from_slice(&parsed.clause_types[..parsed.num_clauses]);
    problem.literal_boundaries[..=parsed.num_literals].copy_from_slice(&parsed.literal_boundaries[..=parsed.num_literals]);
    
    Ok(PyProblem {
        inner: problem,
        frozen: false,
    })
}

/// Parse TPTP string content and return a PyProblem with specified capacity
#[pyfunction]
#[pyo3(signature = (content, max_nodes=1000000, max_clauses=100000, max_edges=5000000))]
fn parse_string_to_array(
    content: String,
    max_nodes: usize,
    max_clauses: usize,
    max_edges: usize,
) -> PyResult<PyProblem> {
    // First parse to get the initial problem
    let parsed = rust_parse_string(&content)
        .map_err(|e| PyValueError::new_err(format!("Parse error: {}", e)))?;
    
    // Create a new problem with specified capacity
    let mut problem = Problem::with_capacity(max_nodes, max_clauses, max_edges);
    
    // Copy the parsed data into the pre-allocated problem
    problem.num_nodes = parsed.num_nodes;
    problem.num_clauses = parsed.num_clauses;
    problem.num_literals = parsed.num_literals;
    problem.num_edges = parsed.num_edges;
    problem.symbols = parsed.symbols;
    
    // Copy arrays (up to the used portion)
    problem.node_types[..parsed.num_nodes].copy_from_slice(&parsed.node_types[..parsed.num_nodes]);
    problem.node_symbols[..parsed.num_nodes].copy_from_slice(&parsed.node_symbols[..parsed.num_nodes]);
    problem.node_polarities[..parsed.num_nodes].copy_from_slice(&parsed.node_polarities[..parsed.num_nodes]);
    problem.node_arities[..parsed.num_nodes].copy_from_slice(&parsed.node_arities[..parsed.num_nodes]);
    problem.node_selected[..parsed.num_nodes].copy_from_slice(&parsed.node_selected[..parsed.num_nodes]);
    
    problem.edge_row_offsets[..=parsed.num_nodes].copy_from_slice(&parsed.edge_row_offsets[..=parsed.num_nodes]);
    problem.edge_col_indices[..parsed.num_edges].copy_from_slice(&parsed.edge_col_indices[..parsed.num_edges]);
    
    problem.clause_boundaries[..=parsed.num_clauses].copy_from_slice(&parsed.clause_boundaries[..=parsed.num_clauses]);
    problem.clause_types[..parsed.num_clauses].copy_from_slice(&parsed.clause_types[..parsed.num_clauses]);
    problem.literal_boundaries[..=parsed.num_literals].copy_from_slice(&parsed.literal_boundaries[..=parsed.num_literals]);
    
    Ok(PyProblem {
        inner: problem,
        frozen: false,
    })
}

/// Parse a TPTP file and return it as a dictionary (for JSON serialization)
#[pyfunction]
#[pyo3(signature = (file_path, include_path=None))]
fn parse_file_to_dict(py: Python, file_path: String, include_path: Option<String>) -> PyResult<PyObject> {
    match rust_parse_file(&file_path, include_path.as_deref()) {
        Ok(problem) => {
            let dict = array_problem_to_dict(py, &problem)?;
            dict.set_item("source_file", &file_path)?;
            Ok(dict.into())
        }
        Err(e) => Err(PyValueError::new_err(format!("Parse error: {}", e))),
    }
}

/// Quick pre-scan to estimate literal count without full parsing
#[pyfunction]
#[pyo3(signature = (file_path, max_depth=None))]
fn prescan_file(file_path: String, max_depth: Option<i32>) -> PyResult<(usize, bool)> {
    let path = Path::new(&file_path);
    let max_depth = max_depth.unwrap_or(3);
    
    match rust_prescan_file(path, max_depth, &mut std::collections::HashSet::new()) {
        Ok((count, is_exact)) => Ok((count, is_exact)),
        Err(e) => Err(PyValueError::new_err(format!("Prescan error: {}", e))),
    }
}

/// Count literals in a TPTP file
#[pyfunction]
#[pyo3(signature = (file_path, include_path=None))]
fn count_literals_in_file(file_path: String, include_path: Option<String>) -> PyResult<usize> {
    match rust_parse_file(&file_path, include_path.as_deref()) {
        Ok(problem) => Ok(problem.num_literals),
        Err(e) => Err(PyValueError::new_err(format!("Parse error: {}", e))),
    }
}

/// Python class wrapper for TPTP parser
#[pyclass]
struct RustTPTPParser {
    include_path: Option<String>,
}

#[pymethods]
impl RustTPTPParser {
    #[new]
    #[pyo3(signature = (include_path=None))]
    fn new(include_path: Option<String>) -> Self {
        RustTPTPParser { include_path }
    }
    
    /// Parse a file and return a dictionary representation
    fn parse_file(&self, py: Python, file_path: String) -> PyResult<PyObject> {
        parse_file(py, file_path, self.include_path.clone())
    }
    
    /// Parse string content and return a dictionary representation
    fn parse_string(&self, py: Python, content: String) -> PyResult<PyObject> {
        parse_string(py, content)
    }
    
    /// Parse a file and return as dictionary
    fn parse_file_to_dict(&self, py: Python, file_path: String) -> PyResult<PyObject> {
        parse_file_to_dict(py, file_path, self.include_path.clone())
    }
    
    /// Prescan a file to estimate literal count
    #[pyo3(signature = (file_path, max_depth=None))]
    fn prescan_file(&self, file_path: String, max_depth: Option<i32>) -> PyResult<(usize, bool)> {
        prescan_file(file_path, max_depth)
    }
    
    /// Count literals in a file
    fn count_literals_in_file(&self, file_path: String) -> PyResult<usize> {
        count_literals_in_file(file_path, self.include_path.clone())
    }
}

/// Convert Problem to Python dictionary
fn array_problem_to_dict<'py>(py: Python<'py>, problem: &Problem) -> PyResult<&'py PyDict> {
    let dict = PyDict::new(py);
    
    // Basic counts
    dict.set_item("num_clauses", problem.num_clauses)?;
    dict.set_item("num_literals", problem.num_literals)?;
    dict.set_item("num_nodes", problem.num_nodes)?;
    
    // Arrays as Python lists - convert Box<[T]> to Vec<T> for PyO3
    let node_types_u8: Vec<u8> = problem.node_types[..problem.num_nodes].to_vec();
    dict.set_item("node_types", node_types_u8)?;
    dict.set_item("node_symbols", problem.node_symbols[..problem.num_nodes].to_vec())?;
    dict.set_item("node_polarities", problem.node_polarities[..problem.num_nodes].to_vec())?;
    dict.set_item("node_arities", problem.node_arities[..problem.num_nodes].to_vec())?;
    
    dict.set_item("edge_row_offsets", problem.edge_row_offsets[..=problem.num_nodes].to_vec())?;
    dict.set_item("edge_col_indices", problem.edge_col_indices[..problem.num_edges].to_vec())?;
    // edge_types removed - can be computed from node types
    
    dict.set_item("clause_boundaries", problem.clause_boundaries[..=problem.num_clauses].to_vec())?;
    dict.set_item("literal_boundaries", problem.literal_boundaries[..=problem.num_literals].to_vec())?;
    
    // Symbol table as list of strings
    let mut symbols_vec = Vec::new();
    for i in 0..problem.symbols.len() {
        if let Some(symbol) = problem.symbols.get(i as u32) {
            symbols_vec.push(symbol.to_string());
        }
    }
    let symbols_list = PyList::new(py, symbols_vec);
    dict.set_item("symbols", symbols_list)?;
    
    Ok(dict)
}

/// Parse TPTP and run saturation, returning statistics
#[pyfunction]
#[pyo3(signature = (input, max_clauses=None, max_iterations=None))]
pub fn parse_and_saturate(
    input: &str,
    max_clauses: Option<usize>,
    max_iterations: Option<usize>,
) -> PyResult<(bool, HashMap<String, usize>)> {
    use crate::saturation::{saturate, SaturationConfig};
    use crate::core::SaturationResult;
    
    // Parse the TPTP input
    let mut problem = rust_parse_string(input)
        .map_err(|e| PyValueError::new_err(format!("Parse error: {}", e)))?;
    
    // Configure saturation
    let mut config = SaturationConfig::default();
    if let Some(max) = max_clauses {
        config.max_clauses = max;
    }
    if let Some(max) = max_iterations {
        config.max_iterations = max;
    }
    
    let initial_clauses = problem.num_clauses;
    
    // Handle empty problem
    if initial_clauses == 0 {
        let mut stats = HashMap::new();
        stats.insert("iterations".to_string(), 0);
        stats.insert("final_clauses".to_string(), 0);
        stats.insert("generated_clauses".to_string(), 0);
        stats.insert("proof_found".to_string(), 0);
        return Ok((false, stats));
    }
    
    // Run saturation
    eprintln!("Python binding: About to call saturate with {} clauses", problem.num_clauses);
    let result = saturate(&mut problem, &config);
    eprintln!("Python binding: saturate returned");
    
    // Collect statistics based on result
    let mut stats = HashMap::new();
    
    match result {
        SaturationResult::Proof(proof) => {
            stats.insert("iterations".to_string(), proof.steps.len().max(1));
            stats.insert("final_clauses".to_string(), proof.problem.num_clauses);
            stats.insert("generated_clauses".to_string(), 
                proof.problem.num_clauses.saturating_sub(initial_clauses));
            stats.insert("proof_found".to_string(), 1);
            Ok((true, stats))
        }
        SaturationResult::Saturated => {
            // For saturated, we don't know exact iterations, but shouldn't be max
            stats.insert("iterations".to_string(), problem.num_clauses);
            stats.insert("final_clauses".to_string(), problem.num_clauses);
            stats.insert("generated_clauses".to_string(), 
                problem.num_clauses.saturating_sub(initial_clauses));
            stats.insert("proof_found".to_string(), 0);
            Ok((false, stats))
        }
        SaturationResult::ResourceLimit => {
            stats.insert("iterations".to_string(), config.max_iterations);
            stats.insert("final_clauses".to_string(), problem.num_clauses);
            stats.insert("generated_clauses".to_string(), 
                problem.num_clauses.saturating_sub(initial_clauses));
            stats.insert("proof_found".to_string(), 0);
            Ok((false, stats))
        }
    }
}