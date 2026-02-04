//! Saturation-based theorem proving using the given clause algorithm

pub mod profile;
pub mod rule;
mod state;
pub mod subsumption;
pub mod trace;

pub use profile::SaturationProfile;
pub use rule::{ProofStateChange, SaturationEventLog};
pub use state::{LiteralSelectionStrategy, SaturationConfig, SaturationResult, SaturationState};
pub use trace::{
    extract_proof_from_events, BackwardSimplification, ClauseSimplification, EventLogReplayer,
    ForwardSimplification, GeneratingInference, SaturationStep, SaturationTrace,
    SimplificationOutcome,
};

use crate::fol::CNFFormula;
use crate::selection::ClauseSelector;

/// Run saturation on a CNF formula
pub fn saturate(
    formula: CNFFormula,
    config: SaturationConfig,
    clause_selector: Box<dyn ClauseSelector>,
) -> (SaturationResult, Option<SaturationProfile>, SaturationTrace) {
    let state = SaturationState::new(formula.clauses, config, clause_selector);
    state.saturate()
}
