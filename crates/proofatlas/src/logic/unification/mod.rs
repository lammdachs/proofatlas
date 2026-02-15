//! Unification, matching, and substitution for first-order terms

mod matching;
pub mod mgu;
pub mod scoped;
pub mod substitution;

pub use matching::match_term;
pub use mgu::{unify, variable_ids_in_term, variables_in_term, UnificationError, UnificationResult};
pub use scoped::{ScopedSubstitution, ScopedVar, flatten_scoped, unify_scoped, unify_scoped_extend};
pub use substitution::Substitution;
