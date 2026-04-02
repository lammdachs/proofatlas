//! Unification, matching, and substitution for first-order terms

mod matching;
pub mod mgu;
pub mod scoped;
pub mod substitution;

#[cfg(test)]
mod proptest_tests;

pub use matching::match_term;
pub use mgu::{unify, variable_ids_in_term, variables_in_term, UnificationError, UnificationResult};
pub use scoped::{ScopedSubstitution, ScopedVar, flatten_scoped, unify_scoped, unify_scoped_terms, apply_scoped_term, lift};
pub use substitution::Substitution;
