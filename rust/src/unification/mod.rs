//! Unification algorithm for first-order terms

mod mgu;

pub use mgu::{unify, UnificationResult, UnificationError};

// Re-export commonly used functions
pub use mgu::{rename_variables, variables_in_term};