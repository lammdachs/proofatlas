//! Literal selection strategies
//!
//! These strategies determine which literals in a clause are eligible
//! for inference rules like resolution and superposition.

use crate::core::Clause;
use std::collections::HashSet;

/// Trait for literal selection strategies
pub trait LiteralSelector: Send + Sync {
    /// Select eligible literals from a clause
    /// Returns indices of selected literals
    fn select(&self, clause: &Clause) -> HashSet<usize>;
    
    /// Get the name of this selection strategy
    fn name(&self) -> &str;
}

/// No selection - all literals are eligible
pub struct NoSelection;

impl LiteralSelector for NoSelection {
    fn select(&self, clause: &Clause) -> HashSet<usize> {
        (0..clause.literals.len()).collect()
    }
    
    fn name(&self) -> &str {
        "NoSelection"
    }
}