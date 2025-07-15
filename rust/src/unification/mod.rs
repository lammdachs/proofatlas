//! Unification algorithm for first-order terms

mod mgu;
mod r#match;

pub use mgu::{unify, UnificationResult, UnificationError};
pub use r#match::match_term;

// Re-export commonly used functions
pub use mgu::{rename_variables, variables_in_term};