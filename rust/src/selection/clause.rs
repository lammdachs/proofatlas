//! Clause selection strategies for the given clause algorithm
//!
//! Available strategies:
//! - `AgeWeightSelector` - Classic age-weight ratio heuristic
//! - `BurnGcnSelector` / `BurnMlpSelector` - Burn-based ML selectors

use crate::core::Clause;
use std::collections::VecDeque;

/// Trait for clause selection strategies
///
/// Note: `Send` is required for passing selectors across thread boundaries.
/// `Sync` is not required since selectors are used mutably in a single-threaded context.
pub trait ClauseSelector: Send {
    /// Select the next clause from the unprocessed set
    /// Returns the index of the selected clause, or None if empty
    fn select(&mut self, unprocessed: &mut VecDeque<usize>, clauses: &[Clause]) -> Option<usize>;

    /// Get the name of this selection strategy
    fn name(&self) -> &str;
}
