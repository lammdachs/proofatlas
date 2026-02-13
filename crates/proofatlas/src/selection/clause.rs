//! Clause selection strategies for the given clause algorithm
//!
//! Available strategies:
//! - `AgeWeightSelector` - Classic age-weight ratio heuristic
//! - `GcnSelector` - GCN-based ML selector (requires `torch` feature)
//! - `SentenceSelector` - Sentence transformer selector (requires `torch` + `sentence` features)

use crate::logic::{Clause, Interner};
use indexmap::IndexSet;
use std::sync::Arc;
use std::time::Duration;

// =============================================================================
// ProverSink — signal-based interface for clause selection
// =============================================================================

/// Signal-based interface for data processing in the saturation loop.
///
/// The prover notifies the sink of clause lifecycle events (transfer, activate,
/// simplify) and requests clause selection via `select()`. Implementations
/// track their own U/P state from these signals, eliminating the need to pass
/// the full clause list on every selection call.
///
/// This trait replaces `ClauseSelector` for new implementations and supports
/// pipelined architectures where data processing runs concurrently.
pub trait ProverSink: Send {
    /// Clause entered U (survived forward simplification).
    fn on_transfer(&mut self, clause_idx: usize, clause: &Arc<Clause>);

    /// Clause moved U→P (activated as given clause).
    fn on_activate(&mut self, clause_idx: usize);

    /// Clause removed from U or P by simplification.
    fn on_simplify(&mut self, clause_idx: usize);

    /// Request clause selection. Returns the index of the selected clause.
    ///
    /// Implementations track their own unprocessed set from signals and
    /// remove the selected clause from their internal state during this call.
    /// The prover is responsible for removing it from `state.unprocessed`.
    fn select(&mut self) -> Option<usize>;

    /// Selector name for profiling.
    fn name(&self) -> &str;

    /// Reset internal state (e.g., for new problem).
    fn reset(&mut self) {}

    /// Return accumulated selector statistics, if tracked.
    fn stats(&self) -> Option<SelectorStats> { None }
}

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
    fn select(&mut self, unprocessed: &mut IndexSet<usize>, clauses: &[Arc<Clause>]) -> Option<usize>;

    /// Get the name of this selection strategy
    fn name(&self) -> &str;

    /// Reset any internal state (e.g., caches) when starting a new problem
    fn reset(&mut self) {}

    /// Return accumulated selector statistics, if tracked.
    fn stats(&self) -> Option<SelectorStats> {
        None
    }

    /// Notify the selector that a clause has been moved to the processed set (U→P).
    /// Used by cross-attention selectors to track the P set for context.
    fn on_clause_processed(&mut self, _clause_idx: usize) {}

    /// Provide the symbol interner for clause serialization.
    /// Called by the prover after parsing, before the saturation loop.
    /// ML selectors that need symbol names (e.g., sentence transformers) should
    /// store this and use it during embedding.
    fn set_interner(&mut self, _interner: Arc<Interner>) {}
}
