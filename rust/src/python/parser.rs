//! Python bindings for the TPTP parser

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::parser;
use crate::fileformats::tptp::TPTPFormat;
use crate::python::types;
use std::path::Path;

/// Register parser functions in the Python module
pub fn register_parser_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_file, m)?)?;
    m.add_function(wrap_pyfunction!(parse_string, m)?)?;
    m.add_function(wrap_pyfunction!(parse_file_to_dict, m)?)?;
    m.add_function(wrap_pyfunction!(prescan_file, m)?)?;
    m.add_function(wrap_pyfunction!(count_literals_in_file, m)?)?;
    m.add_class::<RustTPTPParser>()?;
    Ok(())
}

/// Parse a TPTP file and return a Python Problem object
#[pyfunction]
#[pyo3(signature = (file_path, include_path=None))]
fn parse_file(py: Python, file_path: String, include_path: Option<String>) -> PyResult<PyObject> {
    let parser = if let Some(inc_path) = include_path {
        TPTPFormat::with_include_path(inc_path)
    } else {
        TPTPFormat::new()
    };
    
    match parser.parse_file(Path::new(&file_path)) {
        Ok(problem) => types::problem_to_python(py, &problem),
        Err(e) => Err(PyValueError::new_err(format!("Parse error: {}", e))),
    }
}

/// Parse TPTP string content and return a Python Problem object
#[pyfunction]
fn parse_string(py: Python, content: String) -> PyResult<PyObject> {
    let parser = TPTPFormat::new();
    
    match parser.parse_string(&content) {
        Ok(problem) => types::problem_to_python(py, &problem),
        Err(e) => Err(PyValueError::new_err(format!("Parse error: {}", e))),
    }
}

/// Parse a TPTP file and return it as a dictionary (for JSON serialization)
#[pyfunction]
#[pyo3(signature = (file_path, include_path=None))]
fn parse_file_to_dict(py: Python, file_path: String, include_path: Option<String>) -> PyResult<PyObject> {
    let parser = if let Some(inc_path) = include_path {
        TPTPFormat::with_include_path(inc_path)
    } else {
        TPTPFormat::new()
    };
    
    match parser.parse_file(Path::new(&file_path)) {
        Ok(problem) => {
            let dict = types::problem_to_dict(py, &problem)?;
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
    
    match parser::prescan::prescan_file(path, max_depth, &mut std::collections::HashSet::new()) {
        Ok((count, is_exact)) => Ok((count, is_exact)),
        Err(e) => Err(PyValueError::new_err(format!("Prescan error: {}", e))),
    }
}

/// Count literals in a TPTP file
#[pyfunction]
#[pyo3(signature = (file_path, include_path=None))]
fn count_literals_in_file(file_path: String, include_path: Option<String>) -> PyResult<usize> {
    let parser = if let Some(inc_path) = include_path {
        TPTPFormat::with_include_path(inc_path)
    } else {
        TPTPFormat::new()
    };
    
    match parser.parse_file(Path::new(&file_path)) {
        Ok(problem) => Ok(problem.count_literals()),
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
    
    /// Parse a file and return a Problem object
    fn parse_file(&self, py: Python, file_path: String) -> PyResult<PyObject> {
        parse_file(py, file_path, self.include_path.clone())
    }
    
    /// Parse string content and return a Problem object
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