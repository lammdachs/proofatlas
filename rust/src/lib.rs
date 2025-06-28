//! ProofAtlas Rust Components
//! 
//! This crate provides high-performance implementations of core ProofAtlas components,
//! with optional Python bindings via PyO3.

// Module declarations
pub mod core;
pub mod parser;
pub mod fileformats;
pub mod proofs;

// Future modules (uncomment as implemented)
// pub mod rules;
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
    // Add parser module
    let parser_module = PyModule::new(py, "parser")?;
    python::parser::register_parser_module(py, parser_module)?;
    m.add_submodule(parser_module)?;
    
    // Add core module
    let core_module = PyModule::new(py, "core")?;
    python::core::register_core_module(py, core_module)?;
    m.add_submodule(core_module)?;
    
    // Add proofs module
    let proofs_module = PyModule::new(py, "proofs")?;
    python::proofs::register_proofs_module(py, proofs_module)?;
    m.add_submodule(proofs_module)?;
    
    Ok(())
}