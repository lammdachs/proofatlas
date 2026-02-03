//! Saturation-based theorem proving using the given clause algorithm

pub mod profile;
mod state;
pub mod subsumption;
pub mod trace;

pub use profile::SaturationProfile;
pub use state::{LiteralSelectionStrategy, SaturationConfig, SaturationResult, SaturationState};
pub use trace::{
    BackwardSimplification, ClauseSimplification, ForwardSimplification, GeneratingInference,
    SaturationStep, SaturationTrace, SimplificationOutcome,
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
