//! ProofAtlas Rust Components
//! 
//! This crate provides high-performance implementations of core ProofAtlas components,
//! with optional Python bindings via PyO3.

// Module declarations
pub mod core;
pub mod fileformats;
pub mod array_repr;

// Re-exports for convenient access
pub use crate::core::{logic, error};
pub use crate::fileformats::tptp_parser as tptp;
pub use crate::fileformats::tptp as tptp_format;

// Python bindings module (only compiled with python feature)
#[cfg(feature = "python")]
pub mod python;

// Main Python module entry point
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn proofatlas_rust(py: Python, m: &PyModule) -> PyResult<()> {
    // Add parser module
    let parser_module = PyModule::new(py, "parser")?;
    python::parser::register_parser_module(py, parser_module)?;
    m.add_submodule(parser_module)?;
    
    // Add core module
    let core_module = PyModule::new(py, "core")?;
    python::core::register_core_module(py, core_module)?;
    m.add_submodule(core_module)?;
    
    
    // Add array module
    let array_module = PyModule::new(py, "array_repr")?;
    python::array_bindings::register_array_module(py, array_module)?;
    m.add_submodule(array_module)?;
    
    Ok(())
}