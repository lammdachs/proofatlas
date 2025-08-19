//! Enhanced proof tracking with clause selection events

use crate::core::Clause;
use crate::inference::InferenceResult;

/// Events that occur during proof search
#[derive(Debug, Clone)]
pub enum ProofEvent {
    /// A clause was selected as the given clause
    ClauseSelected {
        clause_idx: usize,
        clause: Clause,
        selection_type: SelectionType,
        step_number: usize,
    },
    /// A new clause was derived by inference
    ClauseDerived {
        inference: InferenceResult,
        clause_idx: usize,
    },
    /// A clause was discarded (redundant, subsumed, etc.)
    ClauseDiscarded {
        clause: Clause,
        reason: DiscardReason,
    },
}

/// How a clause was selected
#[derive(Debug, Clone, Copy)]
pub enum SelectionType {
    /// Selected by age (FIFO)
    ByAge,
    /// Selected by weight (smallest)
    ByWeight { weight: usize },
}

/// Why a clause was discarded
#[derive(Debug, Clone)]
pub enum DiscardReason {
    Tautology,
    Subsumed,
    Duplicate,
    TooLarge,
}
