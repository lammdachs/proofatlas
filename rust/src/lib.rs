//! ProofAtlas Rust Components
//! 
//! This crate provides high-performance implementations of core ProofAtlas components,
//! with optional Python bindings via PyO3.

// Module declarations
pub mod core;
pub mod rules;
pub mod saturation;
pub mod parsing;
pub mod bindings;

// Re-exports for convenient access
pub use crate::parsing::tptp_parser as tptp;
pub use crate::parsing::tptp as tptp_format;
pub use crate::core::{Problem, NodeType};

// Main Python module entry point
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn proofatlas_rust(py: Python, m: &PyModule) -> PyResult<()> {
    // Add parser module
    let parser_module = PyModule::new(py, "parser")?;
    bindings::python::parser::register_parser_module(py, parser_module)?;
    m.add_submodule(parser_module)?;
    
    // Add array module
    let array_module = PyModule::new(py, "array_repr")?;
    bindings::python::array_bindings::register_array_module(py, array_module)?;
    m.add_submodule(array_module)?;
    
    // Add inference rules to main module
    bindings::python::inference_rules::add_inference_rules(m)?;
    
    // Add selection functions to main module
    bindings::python::selection::add_selection_functions(m)?;
    
    Ok(())
}