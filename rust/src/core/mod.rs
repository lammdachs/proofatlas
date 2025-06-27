//! Core logic types and operations
//! 
//! This module mirrors the Python `proofatlas.core` module structure

pub mod logic;
pub mod error;

// Re-export commonly used types
pub use logic::{Term, Literal, Clause, Problem};
pub use error::{ProofAtlasError, Result};