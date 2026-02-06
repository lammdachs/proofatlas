//! Position references into clauses.
//!
//! A `Position` identifies a specific location within the clause store,
//! optionally pointing to a subterm within a literal.

use serde::{Deserialize, Serialize};

/// A reference to a clause, optionally with a path to a specific subterm.
///
/// - `clause`: index of the clause in the clause store
/// - `path`: path within the clause (`[literal_idx, arg_idx, ...]`), empty for whole-clause references
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    /// Index of the clause in the clause store
    pub clause: usize,
    /// Path within the clause: `[literal_idx, arg_idx, arg_idx, ...]`
    /// Empty for whole-clause references.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub path: Vec<usize>,
}

impl Position {
    /// Create a position referencing an entire clause (no subterm path).
    pub fn clause(idx: usize) -> Self {
        Position {
            clause: idx,
            path: vec![],
        }
    }

    /// Create a position referencing a specific location within a clause.
    pub fn with_path(clause: usize, path: Vec<usize>) -> Self {
        Position { clause, path }
    }
}
