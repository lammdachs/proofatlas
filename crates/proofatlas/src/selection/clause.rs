//! Clause selection strategies for the given clause algorithm
//!
//! The `ProverSink` trait provides a signal-based interface for clause selection
//! in the saturation loop. Implementations track clause lifecycle events and
//! provide selection via `select()`.

use crate::logic::Clause;
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
/// Supports pipelined architectures where data processing runs concurrently.
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
