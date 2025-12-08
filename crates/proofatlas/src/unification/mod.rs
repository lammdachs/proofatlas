//! Unification algorithm for first-order terms

mod r#match;
mod mgu;

pub use mgu::{unify, UnificationError, UnificationResult};
pub use r#match::match_term;

// Re-export commonly used functions
pub use mgu::{rename_variables, variables_in_term};
