//! Clause selection strategies for the given clause algorithm
//!
//! Available strategies:
//! - `AgeWeightSelector` - Classic age-weight ratio heuristic
//! - `GcnSelector` - GCN-based ML selector (requires `torch` feature)
//! - `SentenceSelector` - Sentence transformer selector (requires `torch` + `sentence` features)

use crate::fol::Clause;
use indexmap::IndexSet;
use std::time::Duration;

/// Accumulated statistics from a clause selector.
#[derive(Debug, Clone, Default)]
pub struct SelectorStats {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub embed_time: Duration,
    pub score_time: Duration,
}

/// Trait for clause selection strategies
///
/// Note: `Send` is required for passing selectors across thread boundaries.
/// `Sync` is not required since selectors are used mutably in a single-threaded context.
pub trait ClauseSelector: Send {
    /// Select the next clause from the unprocessed set
    /// Returns the index of the selected clause, or None if empty
    fn select(&mut self, unprocessed: &mut IndexSet<usize>, clauses: &[Clause]) -> Option<usize>;

    /// Get the name of this selection strategy
    fn name(&self) -> &str;

    /// Reset any internal state (e.g., caches) when starting a new problem
    fn reset(&mut self) {}

    /// Return accumulated selector statistics, if tracked.
    fn stats(&self) -> Option<SelectorStats> {
        None
    }
}
