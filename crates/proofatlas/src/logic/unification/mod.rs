//! Unification, matching, and substitution for first-order terms

mod matching;
pub mod mgu;
pub mod scoped;
pub mod substitution;

pub use matching::match_term;
pub use mgu::{unify, variable_ids_in_term, variables_in_term, UnificationError, UnificationResult};
pub use scoped::{ScopedSubstitution, ScopedTerm, ScopedVar, flatten, flatten_scoped, unify_scoped};
pub use substitution::Substitution;
