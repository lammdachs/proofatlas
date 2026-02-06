//! Re-exports for backward compatibility.
//! Canonical location: crate::generating::common

pub use crate::generating::common::{
    rename_clause_variables, rename_variables, unify_atoms,
    remove_duplicate_literals, collect_literals_except, is_ordered_greater,
};
pub use crate::state::InferenceResult;
