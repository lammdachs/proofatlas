//! Unification, matching, and substitution for first-order terms

mod matching;
pub mod mgu;
pub mod substitution;

pub use matching::match_term;
pub use mgu::{unify, variable_ids_in_term, variables_in_term, UnificationError, UnificationResult};
pub use substitution::Substitution;
