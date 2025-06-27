//! ProofAtlas Rust Components
//! 
//! This crate provides high-performance implementations of core ProofAtlas components,
//! with optional Python bindings via PyO3.

// Module declarations
pub mod core;
pub mod parser;
pub mod fileformats;

// Future modules (uncomment as implemented)
// pub mod rules;
// pub mod proofs;
// pub mod loops;

// Re-exports for convenient access
pub use crate::core::{logic, error};
pub use crate::parser::tptp;
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
    // Add submodules
    let parser_module = PyModule::new(py, "parser")?;
    python::parser::register_parser_module(py, parser_module)?;
    m.add_submodule(parser_module)?;
    
    // Future: Add more submodules as they're implemented
    // let core_module = PyModule::new(py, "core")?;
    // python::core::register_core_module(py, core_module)?;
    // m.add_submodule(core_module)?;
    
    Ok(())
}